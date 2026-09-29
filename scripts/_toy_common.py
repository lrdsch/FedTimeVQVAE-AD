"""Shared building blocks for the toy_*_channel_anomalies generator family.

The 8 sibling scripts (build_toy_<family>_channel_anomalies.py) all consume:

  * the same 6-entity grid (variable period / n_channels / jitter level),
  * the same periodic + jitter signal generator for normals,
  * the same interval-choice + writer helpers.

Each family-specific script only defines:

  * VARIANTS               : tuple[str, ...]                 # anomaly variants in the family
  * ENTITY_VARIANT_PLAN    : dict[entity_id -> list[str]]    # which variants per entity (1-3 each)
  * apply_anomaly_variant  : function (test, spec, start, stop, variant, rng) -> dict

and then calls build_and_write(...) from this module.

Anomaly labels are written in three forms:
  * test_label/<entity>.npy   (T_test,) int64 0/1 - per-step
  * full_label/<entity>.npy   (T_train+T_val+T_test,) int64 0/1
  * events.csv                one row per anomaly  (entity, period, jitter,
                                                   start, stop, channels, variant, description)
  * labels.csv                one row per entity   (entity, period, n_channels, jitter, n_anomalies)
  * metadata.json             generator parameters + per-entity summary
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


# Canonical 6-entity cry
# This grid is REUSED IDENTICALLY across all 8 toy_*_channel_anomalies datasets,
# so per-entity behaviour is comparable across families.

ENTITIES: tuple[dict, ...] = (
    {"entity_id": "toy_00", "period":  32, "n_channels":  6, "jitter": "mild"},
    {"entity_id": "toy_01", "period":  32, "n_channels":  8, "jitter": "medium"},
    {"entity_id": "toy_02", "period":  64, "n_channels":  8, "jitter": "strong"},
    {"entity_id": "toy_03", "period":  64, "n_channels": 10, "jitter": "medium"},
    {"entity_id": "toy_04", "period": 128, "n_channels":  8, "jitter": "strong"},
    {"entity_id": "toy_05", "period": 128, "n_channels": 10, "jitter": "strong"},
)


# Default split lengths (user-confirmed: 3072 / 768 / 1536).
TRAIN_LENGTH_DEFAULT: int = 3072
VAL_LENGTH_DEFAULT:   int = 768
TEST_LENGTH_DEFAULT:  int = 1536


JITTER_LEVELS: dict[str, dict[str, float]] = {
    # phase_step  : per-step phase random-walk std (radians)
    # amp_sigma   : per-cycle multiplicative amplitude jitter std
    # period_pct  : period jitter as fraction of P (time-warp amplitude)
    # obs_sigma   : per-sample Gaussian observation noise std
    # trend_sigma : per-channel linear slope std (units per step)
    # Levels boosted by +50% over the v1 defaults so even "strong" produces
    # visibly noisy series rather than near-deterministic sinusoids.
    "mild":   {"phase_step": 0.0075, "amp_sigma": 0.030, "period_pct": 0.015, "obs_sigma": 0.038, "trend_sigma": 5.0e-6},
    "medium": {"phase_step": 0.0150, "amp_sigma": 0.075, "period_pct": 0.030, "obs_sigma": 0.060, "trend_sigma": 1.0e-5},
    "strong": {"phase_step": 0.0300, "amp_sigma": 0.150, "period_pct": 0.053, "obs_sigma": 0.105, "trend_sigma": 2.0e-5},
}


# Harmonic multipliers for each channel of an entity (relative to the entity
# BASE period). MUST be reciprocals of integers (= divisors of the base period
# in the time domain) so that LCM(channel_periods) = base_period exactly.
# This makes the multivariate entity period coincide with the base period,
# i.e. `period == base_period` for every entity in this grid.
#
# Frequency-domain interpretation: channel 0 is the fundamental (f), channel 2
# is the 2nd harmonic (2f, period P/2), channel-with-0.25 is the 4th harmonic
# (4f, period P/4). All harmonics are integer multiples of the fundamental
# frequency, so the composite signal repeats exactly every P samples.
HARMONIC_MULTIPLIERS: tuple[float, ...] = (
    1.0, 1.0, 0.5, 0.25, 1.0, 0.5, 0.25, 1.0, 0.5, 1.0, 0.25, 0.5,
)


# ─── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class EntitySpec:
    entity_id: str
    period: int
    n_channels: int
    jitter: str
    # Morphology of the normal signal. Only consulted by `generate_normal_waveform`
    # (the univariate/federated builder); the legacy `generate_normal_signal`
    # ignores it, so every pre-existing toy family keeps its exact output.
    waveform: str = "sine"
    # Free-form group tag ("machine type"). Carried into events/labels/metadata so a
    # federation can be scoped to one coherent group of clients.
    cluster: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "EntitySpec":
        return cls(
            entity_id=d["entity_id"],
            period=int(d["period"]),
            n_channels=int(d["n_channels"]),
            jitter=str(d["jitter"]),
            waveform=str(d.get("waveform", "sine")),
            cluster=str(d.get("cluster", "")),
        )


@dataclass
class AnomalyEvent:
    start: int
    stop: int
    channels: list[int]
    variant: str
    description: str


@dataclass
class ChannelHarmonics:
    """Per-channel periods (P_c) for one entity."""
    periods: list[float] = field(default_factory=list)


# Normal signal generator

def channel_periods(base_period: int, n_channels: int) -> list[float]:
    """Return per-channel periods derived from the entity's BASE period.

    Channel 0 always uses the base period. Other channels use the harmonic
    multipliers (looping if n_channels > len(HARMONIC_MULTIPLIERS))."""
    return [float(base_period) * HARMONIC_MULTIPLIERS[c % len(HARMONIC_MULTIPLIERS)]
            for c in range(n_channels)]


def multivariate_period(channel_periods_list: list[float]) -> int:
    """LCM of integer channel periods - the TRUE period of the multivariate
    signal (smallest T such that x(t+T) = x(t) for ALL channels simultaneously).

    Given the harmonic multipliers in HARMONIC_MULTIPLIERS = {0.25, 0.5, 1, 1.5, 2}
    and BASE periods drawn from {32, 48, 64, 96, 128, 192} (all divisible by 4),
    every channel period is an integer, so LCM is well-defined.
    """
    if not channel_periods_list:
        return 0
    ints = [int(round(p)) for p in channel_periods_list]
    # math.lcm is Python 3.9+; build it from math.gcd (3.5+) so the toy
    # builders also run under the Python 3.7 baseline envs (e.g. omni37).
    def _lcm2(a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return a * b // math.gcd(a, b)
    return reduce(_lcm2, ints)


def _lowpass_walk(length: int, kernel_size: int, rng: np.random.Generator) -> np.ndarray:
    """White noise lowpassed with a boxcar kernel - smooth slow drift in [-~1, ~1]."""
    if kernel_size < 1:
        return np.zeros(length, dtype=np.float32)
    raw = rng.normal(0.0, 1.0, size=length + kernel_size).astype(np.float32)
    k = np.ones(kernel_size, dtype=np.float32) / kernel_size
    smoothed = np.convolve(raw, k, mode="same")[:length]
    return smoothed.astype(np.float32)


def _cumulative_phase_walk(length: int, sigma: float, clip: float,
                           rng: np.random.Generator) -> np.ndarray:
    """Cumulative random walk over phase, clipped to ±clip radians."""
    steps = rng.normal(0.0, sigma, size=length).astype(np.float32)
    walk = np.cumsum(steps)
    return np.clip(walk, -clip, clip).astype(np.float32)


def _cycle_amplitude_envelope(length: int, P_c: float, sigma: float,
                              rng: np.random.Generator) -> np.ndarray:
    """Per-cycle multiplicative amplitude jitter, expanded back to length."""
    n_cycles = int(np.ceil(length / max(P_c, 1.0))) + 1
    cycle_amp = (1.0 + sigma * rng.normal(0.0, 1.0, size=n_cycles)).astype(np.float32)
    t = np.arange(length, dtype=np.float32)
    idx = np.minimum((t / max(P_c, 1.0)).astype(np.int64), n_cycles - 1)
    return cycle_amp[idx]


def generate_normal_signal(length: int, spec: EntitySpec,
                           rng: np.random.Generator) -> np.ndarray:
    """Multivariate periodic + jittered signal.

    Shape: (length, spec.n_channels). Each channel has its own harmonic
    period (P_c, all divisors of spec.period). Each channel is a weighted
    sum of 3 harmonics: f, 2f, 4f (i.e. periods P_c, P_c/2, P_c/4), with
    PER-CHANNEL RANDOMISED harmonic weights so different channels have
    different "timbres". On top of the harmonic stack we apply: phase
    random walk, time-warp jitter, cycle-level amplitude envelope,
    Gaussian observation noise, a small linear trend, plus a weak shared
    driver (correlates channels but doesn't dominate).
    """
    t = np.arange(length, dtype=np.float32)
    jp = JITTER_LEVELS[spec.jitter]
    periods = channel_periods(spec.period, spec.n_channels)

    # Weak shared driver at fundamental period (was 0.30, reduced to 0.15
    # so channels are less correlated and the model has to learn more
    # genuine per-channel structure).
    omega0 = 2.0 * np.pi / float(spec.period)
    driver = 0.15 * np.sin(omega0 * t + rng.uniform(0.0, 2.0 * np.pi)).astype(np.float32)

    data = np.zeros((length, spec.n_channels), dtype=np.float32)
    for c in range(spec.n_channels):
        P_c = periods[c]
        baseline = float(rng.uniform(-1.0, 1.0))
        amplitude = float(rng.uniform(0.7, 1.4))
        phase_1 = float(rng.uniform(0.0, 2.0 * np.pi))
        phase_2 = float(rng.uniform(0.0, 2.0 * np.pi))
        phase_3 = float(rng.uniform(0.0, 2.0 * np.pi))
        # Per-channel RANDOMISED harmonic weights. Sampled from U with
        # decreasing range so the fundamental usually dominates, but with
        # enough variation that some channels emphasise the 2nd or 3rd
        # harmonic and have a visibly different timbre.
        w1 = float(rng.uniform(0.5, 1.2))
        w2 = float(rng.uniform(0.1, 0.6))
        w3 = float(rng.uniform(0.05, 0.30))
        # Harmonic frequencies: f, 2f, 4f (periods P_c, P_c/2, P_c/4).
        # Power-of-2 step keeps every channel's signal strictly periodic
        # with period P_c.
        omega_1 = 2.0 * np.pi / max(P_c, 1.0)
        omega_2 = 2.0 * np.pi / max(P_c / 2.0, 1.0)
        omega_3 = 2.0 * np.pi / max(P_c / 4.0, 1.0)
        # Time-warp: \tilde t = t + (period_pct * P_c) * lowpass-noise(t).
        warp_amp = jp["period_pct"] * P_c
        warp = warp_amp * _lowpass_walk(length, kernel_size=max(5, int(P_c // 4)), rng=rng)
        # Phase random walk (clipped to keep instantaneous frequency sane).
        phase_walk = _cumulative_phase_walk(length, jp["phase_step"], clip=0.3, rng=rng)
        # Cycle-level amplitude envelope.
        amp_env = _cycle_amplitude_envelope(length, P_c, jp["amp_sigma"], rng)
        # Observation noise.
        obs = rng.normal(0.0, jp["obs_sigma"], size=length).astype(np.float32)
        # Tiny per-channel linear trend (slope per step, total drift over
        # a 5k-step series ~5-10% of unit amplitude).
        slope = float(rng.normal(0.0, jp["trend_sigma"]))
        trend = (slope * t).astype(np.float32)

        signal = (
            baseline
            + trend
            + amplitude * amp_env * (
                w1 * np.sin(omega_1 * (t + warp) + phase_1 + phase_walk)
              + w2 * np.cos(omega_2 * (t + warp) + phase_2)
              + w3 * np.sin(omega_3 * (t + warp) + phase_3)
            )
            + driver
            + obs
        ).astype(np.float32)
        data[:, c] = signal
    return data


def generate_normal_one_hot(length: int, spec: EntitySpec,
                            rng: np.random.Generator) -> np.ndarray:
    """One-hot rotation across `spec.n_channels` with period `spec.period`.

    Dwell pattern is even (period / n_channels) with leftovers distributed
    symmetrically at outer states. A small Bernoulli "category-flip" jitter
    is applied per step (only swaps consecutive states with prob = phase_step)
    so the rotation isn't strictly deterministic.
    """
    n = spec.n_channels
    period = spec.period
    jp = JITTER_LEVELS[spec.jitter]

    # Symmetric dwell pattern: each state holds for ~period/n steps.
    base = period // n
    extra = period - base * n
    dwells = [base] * n
    for i in range(extra):
        if i % 2 == 0:
            dwells[i // 2] += 1
        else:
            dwells[n - 1 - i // 2] += 1
    # One full period template (period, n)
    template = np.zeros((period, n), dtype=np.float32)
    pos = 0
    for state, d in enumerate(dwells):
        template[pos: pos + d, state] = 1.0
        pos += d
    # Tile to length
    n_full, rem = divmod(length, period)
    tiled = np.tile(template, (n_full, 1))
    if rem > 0:
        tiled = np.concatenate([tiled, template[:rem]], axis=0)
    data = tiled.astype(np.float32)

    # Apply mild jitter: with probability phase_step at each step, swap with
    # the next neighbour (an early/late transition). Bounded - the marginal
    # one-hot structure is preserved.
    flip_prob = float(jp["phase_step"])
    if flip_prob > 0.0:
        coins = rng.random(size=length)
        for t_idx in range(length - 1):
            if coins[t_idx] < flip_prob:
                data[t_idx], data[t_idx + 1] = data[t_idx + 1].copy(), data[t_idx].copy()
    return data


# ─── Machine-type waveforms (used by the univariate federated benchmark) ─────
# `generate_normal_signal` above is a *harmonic sine stack*: at C=1 every entity
# collapses to "a sinusoid at its base period", so the only surviving non-IID axes
# are period + jitter + rng. That is too little separation to define client GROUPS.
#
# The generators below give each group its own MORPHOLOGY ("machine type"). Within
# a group the waveform and the base period are FIXED — that is what makes the
# clients established enough to federate — and heterogeneity comes only from the
# jitter level, the rng realisation, and the per-entity shape parameters each
# waveform samples (duty cycle, ring decay, harmonic weights, …).
#
# Every waveform is strictly 1-periodic in the cycle position u ∈ [0, 1) and is
# normalised to unit std before the amplitude/noise stack, so `obs_sigma` means
# the same SNR for a sine and for a sparse pulse train.


def _smooth(x: np.ndarray, kernel_size: int) -> np.ndarray:
    """Boxcar smoothing that preserves length (used to round hard edges)."""
    if kernel_size < 2:
        return x.astype(np.float32)
    k = np.ones(int(kernel_size), dtype=np.float32) / float(kernel_size)
    # Wrap-pad so the cycle stays periodic across the seam.
    pad = int(kernel_size)
    padded = np.concatenate([x[-pad:], x, x[:pad]])
    return np.convolve(padded, k, mode="same")[pad: pad + len(x)].astype(np.float32)


def _cycle_position(length: int, P_c: float, jp: dict[str, float],
                    rng: np.random.Generator, jitter_scale: float = 1.0) -> np.ndarray:
    """Jittered cycle position u ∈ [0, 1): time-warp + phase random walk, mod 1.

    `jitter_scale` < 1 shrinks the temporal jitter. Both `period_pct` and
    `phase_step` are expressed relative to the CYCLE, but a QRS spike or a
    ringing burst has structure at P/20, so an unscaled warp smears it away: at
    `jitter="strong"` the lag-P autocorrelation of `pulse` drops to 0.27 vs
    ~0.91 for a sine. Per-waveform scales (see `WAVEFORM_JITTER_SCALE`) put every
    morphology back on the sine's periodicity curve (≈0.99/0.97/0.91 for
    mild/medium/strong), so `jitter` means the same thing across clusters.
    """
    t = np.arange(length, dtype=np.float32)
    s = float(jitter_scale)
    warp = (jp["period_pct"] * P_c * s) * _lowpass_walk(
        length, kernel_size=max(5, int(P_c // 4)), rng=rng)
    phase_walk = _cumulative_phase_walk(length, jp["phase_step"] * s, clip=0.3 * s, rng=rng)
    u = (t + warp) / max(P_c, 1.0) + phase_walk / (2.0 * np.pi)
    return np.mod(u, 1.0).astype(np.float32)


def _wave_sine(u: np.ndarray, P_c: float, rng: np.random.Generator) -> np.ndarray:
    """M1 — rotating machinery: fundamental + 2nd + 4th harmonic, random timbre."""
    w1, w2, w3 = rng.uniform(0.6, 1.1), rng.uniform(0.10, 0.50), rng.uniform(0.05, 0.25)
    p1, p2, p3 = rng.uniform(0.0, 2.0 * np.pi, size=3)
    return (w1 * np.sin(2 * np.pi * u + p1)
            + w2 * np.cos(4 * np.pi * u + p2)
            + w3 * np.sin(8 * np.pi * u + p3)).astype(np.float32)


def _wave_square(u: np.ndarray, P_c: float, rng: np.random.Generator) -> np.ndarray:
    """M2 — solenoid valve: two-state cycle with a random duty and rounded edges."""
    duty = float(rng.uniform(0.35, 0.55))
    raw = np.where(u < duty, 1.0, -1.0).astype(np.float32)
    return _smooth(raw, kernel_size=max(3, int(0.06 * P_c)))


def _wave_saw(u: np.ndarray, P_c: float, rng: np.random.Generator) -> np.ndarray:
    """M3 — reciprocating pump: slow charge ramp, fast discharge."""
    skew = float(rng.uniform(0.80, 0.95))          # fraction of the cycle spent charging
    r = np.where(u < skew, u / skew, 1.0 - (u - skew) / max(1e-6, 1.0 - skew))
    return _smooth(2.0 * r.astype(np.float32) - 1.0, kernel_size=max(3, int(0.03 * P_c)))


def _wave_pulse(u: np.ndarray, P_c: float, rng: np.random.Generator) -> np.ndarray:
    """M4 — cardiac-like: sparse QRS complex + T wave, flat baseline in between."""
    def g(centre: float, sigma: float, amp: float) -> np.ndarray:
        return amp * np.exp(-0.5 * ((u - centre) / sigma) ** 2)
    r_c = float(rng.uniform(0.18, 0.22))
    t_c = float(rng.uniform(0.42, 0.50))
    return (
        -g(r_c - 0.040, 0.012, float(rng.uniform(0.15, 0.30)))     # Q
        + g(r_c, 0.014, 1.00)                                       # R
        - g(r_c + 0.040, 0.018, float(rng.uniform(0.25, 0.45)))     # S
        + g(t_c, 0.060, float(rng.uniform(0.20, 0.35)))             # T
    ).astype(np.float32)


def _wave_ring(u: np.ndarray, P_c: float, rng: np.random.Generator) -> np.ndarray:
    """M5 — bearing impact: once-per-cycle shock followed by damped ringing."""
    u0 = float(rng.uniform(0.05, 0.15))
    n_ring = float(rng.uniform(6.0, 10.0))
    decay = float(rng.uniform(8.0, 16.0))
    d = np.clip(u - u0, 0.0, None)
    env = np.exp(-decay * d) * (u >= u0)
    ring = env * np.sin(2.0 * np.pi * n_ring * d)
    hum = 0.15 * np.sin(2.0 * np.pi * u + float(rng.uniform(0.0, 2.0 * np.pi)))
    return (ring + hum).astype(np.float32)


def _wave_am(u: np.ndarray, P_c: float, rng: np.random.Generator) -> np.ndarray:
    """M6 — variable-speed drive: carrier amplitude-modulated by a slow envelope.

    `k` (carriers per envelope cycle) is an integer so the product stays exactly
    1-periodic in u."""
    k = int(rng.integers(5, 8))
    m = float(rng.uniform(0.5, 0.9))
    ph = float(rng.uniform(0.0, 2.0 * np.pi))
    return ((1.0 + m * np.sin(2.0 * np.pi * u + ph))
            * np.sin(2.0 * np.pi * k * u)).astype(np.float32)


WAVEFORMS: dict[str, Callable[[np.ndarray, float, np.random.Generator], np.ndarray]] = {
    "sine":   _wave_sine,
    "square": _wave_square,
    "saw":    _wave_saw,
    "pulse":  _wave_pulse,
    "ring":   _wave_ring,
    "am":     _wave_am,
}

# Temporal-jitter scales for morphology — calibrated so lag-P autocorrelation at
# mild/medium/strong matches the sine's ≈0.99/0.97/0.91. Fine-detail waveforms
# need a smaller warp; see `_cycle_position`.
WAVEFORM_JITTER_SCALE: dict[str, float] = {
    "sine": 1.00, "square": 1.00, "saw": 1.00,
    "pulse": 0.20, "ring": 0.25, "am": 0.25,
}


def generate_normal_waveform(length: int, spec: EntitySpec,
                             rng: np.random.Generator) -> np.ndarray:
    """Normal signal with `spec.waveform` morphology. Shape `(length, n_channels)`.

    Works for any C (each channel gets its own harmonic period and its own shape
    draw), but the federated benchmark uses C=1: one series per client.
    """
    if spec.waveform not in WAVEFORMS:
        raise ValueError(f"Unknown waveform {spec.waveform!r}; known: {sorted(WAVEFORMS)}")
    wave_fn = WAVEFORMS[spec.waveform]
    jitter_scale = WAVEFORM_JITTER_SCALE[spec.waveform]
    jp = JITTER_LEVELS[spec.jitter]
    periods = channel_periods(spec.period, spec.n_channels)
    t = np.arange(length, dtype=np.float32)

    data = np.zeros((length, spec.n_channels), dtype=np.float32)
    for c in range(spec.n_channels):
        P_c = periods[c]
        u = _cycle_position(length, P_c, jp, rng, jitter_scale)
        base = wave_fn(u, P_c, rng)
        # Unit-std so `obs_sigma` is the same SNR across morphologies.
        base = base / max(float(base.std()), 1e-6)
        amp_env = _cycle_amplitude_envelope(length, P_c, jp["amp_sigma"], rng)
        obs = rng.normal(0.0, jp["obs_sigma"], size=length).astype(np.float32)
        baseline = float(rng.uniform(-1.0, 1.0))
        amplitude = float(rng.uniform(0.7, 1.4))
        slope = float(rng.normal(0.0, jp["trend_sigma"]))
        data[:, c] = (baseline + slope * t + amplitude * amp_env * base + obs).astype(np.float32)
    return data


# Anomaly interval placement

def choose_intervals(test_length: int, period: int, widths: list[int],
                     rng: np.random.Generator) -> list[tuple[int, int]]:
    """Place `len(widths)` non-overlapping anomaly windows on [margin, test_length-margin)
    with at least `period` gap between consecutive windows.

    Margins/gaps are 1·P (one full period of context on each side). This is the
    tightest spacing that still leaves visible normal context around each
    anomaly in the plots.

    Raises if the requested count cannot fit (caller should reduce anomalies).
    """
    margin = period
    gap = period
    n = len(widths)
    used: list[tuple[int, int]] = []
    # Reserve room: required = sum(widths) + (n-1)*gap + 2*margin.
    required = sum(widths) + (n - 1) * gap + 2 * margin
    if required > test_length:
        raise RuntimeError(
            f"Cannot fit {n} anomalies of total width {sum(widths)} with "
            f"period={period} in test_length={test_length} (need {required})."
        )

    # Greedy retry: place widest first.
    order = sorted(range(n), key=lambda i: -widths[i])
    placed: dict[int, tuple[int, int]] = {}
    for attempt in range(2000):
        placed.clear()
        ok = True
        for i in order:
            w = widths[i]
            for _ in range(200):
                start = int(rng.integers(margin, test_length - margin - w))
                stop = start + w
                if all(stop + gap <= s or start - gap >= e
                       for (s, e) in placed.values()):
                    placed[i] = (start, stop)
                    break
            else:
                ok = False
                break
        if ok:
            used = [placed[i] for i in range(n)]
            return used
    raise RuntimeError(
        f"Failed to place {n} anomalies after 2000 attempts "
        f"(test_length={test_length}, period={period}, widths={widths})."
    )


# ─── Writer ──────────────────────────────────────────────────────────────────

def build_and_write(
    *,
    dataset_name: str,
    family_description: str,
    output_dir: Path,
    entity_variant_plan: dict[str, list[str]],
    apply_variant: Callable[[np.ndarray, EntitySpec, int, int, str, np.random.Generator],
                            AnomalyEvent],
    make_normal: Callable[[int, EntitySpec, np.random.Generator], np.ndarray] | None = None,
    entities: tuple[dict, ...] = ENTITIES,
    train_length: int = TRAIN_LENGTH_DEFAULT,
    val_length: int = VAL_LENGTH_DEFAULT,
    test_length: int = TEST_LENGTH_DEFAULT,
    entity_lengths: dict[str, tuple[int, int, int]] | None = None,
    seed: int = 7,
    overwrite: bool = False,
    eps_mask: float = 1e-6,
    extra_meta: dict | None = None,
) -> None:
    """Common writer. The family-specific script only supplies:

    * dataset_name      : e.g. "toy_point_channel_anomalies"
    * family_description: one-line human description
    * entity_variant_plan: dict {entity_id -> list of variant names to inject}
                          (each entity gets 1-3 variants -> 1-3 anomalies in test)
    * apply_variant(test, spec, start, stop, variant, rng) -> AnomalyEvent
                          mutates test in-place, returns the (start, stop, channels,
                          variant, description) tuple.

    File layout: identical to the existing toy datasets (train/, val/, test/,
    test_label/, full/, full_label/, events.csv, labels.csv, metadata.json).
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output_dir} already exists and is not empty. Use --overwrite to replace."
        )
    if make_normal is None:
        make_normal = generate_normal_signal

    for sub in ("train", "val", "test", "test_label", "full", "full_label",
                "test_clean", "test_mask"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    rng_master = np.random.default_rng(seed)
    events_rows: list[dict] = []
    labels_rows: list[dict] = []
    meta_entities: list[dict] = []

    for ent_idx, ent in enumerate(entities):
        spec = EntitySpec.from_dict(ent)
        if spec.entity_id not in entity_variant_plan:
            raise KeyError(
                f"entity_variant_plan missing entry for {spec.entity_id} "
                f"(family={dataset_name})"
            )
        variants = list(entity_variant_plan[spec.entity_id])
        if not (1 <= len(variants) <= 3):
            raise ValueError(
                f"{spec.entity_id}: expected 1-3 variants, got {len(variants)}"
            )

        # Per-entity rng for reproducible per-entity output even if global order changes.
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))

        # Per-entity (train, val, test) lengths override the scalar defaults — lets a
        # builder mirror a real dataset's heterogeneous client lengths (the wsd twin
        # copies wsd's exact per-KPI (train, val, test) triples so the synthetic
        # federation has an IDENTICAL length distribution, only the signal differs).
        tl, vl, te = (entity_lengths or {}).get(
            spec.entity_id, (train_length, val_length, test_length))

        # Per-channel periods and TRUE multivariate period (LCM).
        ch_periods = channel_periods(spec.period, spec.n_channels)
        entity_lcm = multivariate_period(ch_periods)

        # Generate one long normal signal then split.
        full_normal = make_normal(
            tl + vl + te, spec, rng,
        )
        # (T, C) even at C=1. A (T,) array here would round-trip through np.save
        # and blow up in the loader (`x.shape[1]` -> IndexError) and in
        # federated_eval's test_mask indexing.
        if full_normal.ndim != 2 or full_normal.shape[1] != spec.n_channels:
            raise ValueError(
                f"{spec.entity_id}: make_normal must return (T, {spec.n_channels}); "
                f"got {full_normal.shape}"
            )
        train = full_normal[:tl].copy()
        val = full_normal[tl: tl + vl].copy()
        test = full_normal[tl + vl:].copy()

        # Pick anomaly widths.
        #   spike/bump/contextual_point → 1-3 steps (point-like).
        #   missed_beat/dropout         → P/2.
        #   everything else             → full period P (the default).
        # We deliberately keep widths ≤ P so 3 anomalies + margins/gaps fit
        # within test_length=1536 even for P=192 (longest period in the grid).
        widths: list[int] = []
        for v in variants:
            if ("spike" in v) or ("bump" in v) or v == "contextual_point":
                widths.append(int(rng.integers(1, 4)))
            elif "missed_beat" in v or "dropout" in v:
                widths.append(int(max(8, spec.period // 2)))
            else:
                widths.append(int(spec.period))

        # IMPORTANT: keep the (variant, interval) mapping intact across the
        # time-order sort — pair them first, then sort by start.
        intervals = choose_intervals(te, spec.period, widths, rng)
        pairs = sorted(zip(intervals, variants), key=lambda iv: iv[0][0])

        # Ground truth for explainability/federation: snapshot the CLEAN test
        # BEFORE any in-place injection. The per-channel-per-timestep anomaly
        # mask is the ACTUAL deviation introduced (not the declared channels,
        # which over-claim for some families) — so localization target, CF
        # repair target, and non-disturbance target are one lossless object.
        x_clean = test.copy()

        events: list[AnomalyEvent] = []
        test_label = np.zeros((te,), dtype=np.int64)
        for (start, stop), variant in pairs:
            ev = apply_variant(test, spec, start, stop, variant, rng)
            events.append(ev)
            test_label[ev.start: ev.stop] = 1

        test_mask = (np.abs(test - x_clean) > eps_mask).astype(np.int64)  # (T_test, C)

        train_with_val = np.concatenate([train, val], axis=0)
        full = np.concatenate([train, val, test], axis=0)
        full_label = np.concatenate([
            np.zeros((tl + vl,), dtype=np.int64),
            test_label,
        ], axis=0)

        np.save(output_dir / "train" / f"{spec.entity_id}.npy", train)
        np.save(output_dir / "val" / f"{spec.entity_id}.npy", val)
        np.save(output_dir / "test" / f"{spec.entity_id}.npy", test)
        np.save(output_dir / "test_label" / f"{spec.entity_id}.npy", test_label)
        np.save(output_dir / "full" / f"{spec.entity_id}.npy", full)
        np.save(output_dir / "full_label" / f"{spec.entity_id}.npy", full_label)
        np.save(output_dir / "test_clean" / f"{spec.entity_id}.npy", x_clean)
        np.save(output_dir / "test_mask" / f"{spec.entity_id}.npy", test_mask)

        for ev in events:
            events_rows.append({
                "entity_id": spec.entity_id,
                "cluster": spec.cluster,
                "waveform": spec.waveform,
                "period": entity_lcm,                       # LCM of channel periods (TRUE multivariate period)
                "base_period": spec.period,                 # smallest harmonic; drives anomaly width / jitter
                "n_channels": spec.n_channels,
                "jitter": spec.jitter,
                "variant": ev.variant,
                "test_anomaly_start": ev.start,
                "test_anomaly_stop_exclusive": ev.stop,
                "full_anomaly_start": tl + vl + ev.start,
                "full_anomaly_stop_exclusive": tl + vl + ev.stop,
                "anomaly_channels": json.dumps([int(c) for c in ev.channels]),
                "description": ev.description,
            })

        labels_rows.append({
            "entity_id": spec.entity_id,
            "cluster": spec.cluster,
            "waveform": spec.waveform,
            "period": entity_lcm,                           # LCM
            "base_period": spec.period,                     # generator base unit
            "n_channels": spec.n_channels,
            "jitter": spec.jitter,
            "n_anomalies": len(events),
            "variants": json.dumps(variants),
            # Convenience: collapse all anomaly intervals + channels for plot script
            "anomaly_channels": json.dumps(sorted({
                int(c) for ev in events for c in ev.channels
            })),
            "anomaly_intervals": json.dumps(
                [[ev.start, ev.stop] for ev in events]
            ),
        })

        meta_entities.append({
            "entity_id": spec.entity_id,
            "cluster": spec.cluster,
            "waveform": spec.waveform,
            "period": entity_lcm,                           # TRUE multivariate period
            "base_period": spec.period,                     # smallest harmonic / generator unit
            "n_channels": spec.n_channels,
            "jitter": spec.jitter,
            "train_length": int(tl),                        # per-entity (may differ across clients)
            "val_length": int(vl),
            "test_length": int(te),
            "channel_periods": ch_periods,
            "variants": variants,
            "events": [
                {"start": e.start, "stop": e.stop, "channels": e.channels,
                 "variant": e.variant, "description": e.description}
                for e in events
            ],
        })

    pd.DataFrame(events_rows).to_csv(output_dir / "events.csv", index=False)
    pd.DataFrame(labels_rows).to_csv(output_dir / "labels.csv", index=False)

    # cluster -> [entity_id, ...], in declaration order. Empty when no entity
    # declares a cluster (every pre-existing toy family). Consumed by
    # `federated_eval.py --cluster` to scope a federation to one machine type.
    clusters: dict[str, list[str]] = {}
    for e in meta_entities:
        if e["cluster"]:
            clusters.setdefault(e["cluster"], []).append(e["entity_id"])

    metadata = {
        "dataset": dataset_name,
        "family_description": family_description,
        "seed": seed,
        "n_series": len(entities),
        "clusters": clusters,
        "train_length": train_length,
        "val_length": val_length,
        "test_length": test_length,
        "jitter_levels": JITTER_LEVELS,
        "harmonic_multipliers": list(HARMONIC_MULTIPLIERS),
        "splits": {
            "train": "normal data only",
            "val": "normal data only",
            "test": "1-3 labelled anomalies per entity, all from this family",
            "full": "concatenation of train + val + test",
        },
        "entities": meta_entities,
    }
    # Optional top-level extras (e.g. metrics_tolerance, measured_period_min) so a
    # builder can mirror a real dataset's evaluation protocol via metadata.json,
    # which config.apply_dataset_overrides reads.
    if extra_meta:
        metadata.update(extra_meta)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if clusters:
        with (output_dir / "clusters.json").open("w", encoding="utf-8") as f:
            json.dump(clusters, f, indent=2)

    print(f"Wrote {dataset_name} -> {output_dir}")
    # `entities`, not the module-level ENTITIES grid — a builder that overrides
    # the grid (build_toy_fed*, ...) used to print a wrong client count here.
    print(f"  Entities: {len(entities)} | train={train_length} val={val_length} test={test_length}")
    if clusters:
        print(f"  Clusters: {len(clusters)} -> " +
              ", ".join(f"{k}({len(v)})" for k, v in clusters.items()))
    print(f"  Total anomalies: {len(events_rows)}")


# ─── Anomaly building blocks (callable from family-specific scripts) ─────────
# Each helper mutates `test` in-place over [start, stop) on `channels`,
# returns a one-line description string. The variant function in the family
# script picks which helpers to call and which channels to target.

def pick_channels(rng: np.random.Generator, n_channels: int, k: int) -> list[int]:
    """Pick k distinct channel indices uniformly at random."""
    k = max(1, min(k, n_channels))
    return sorted(int(c) for c in rng.choice(n_channels, size=k, replace=False))


def _require_channels(variant: str, channels: list[int], minimum: int) -> None:
    """Fail loudly when a cross-channel anomaly cannot be injected.

    These injectors used to return a "(no-op)" description string instead. That
    was silently catastrophic: `build_and_write` still stamped `test_label = 1`
    over the interval, so the dataset shipped a LABELLED anomaly with a
    bit-identical signal (and an all-zero `test_mask`, i.e. no explainability
    target). At C=1 every cross-channel family hit that path. Raising here means
    a builder that requests an impossible family fails at build time instead of
    poisoning every downstream metric.
    """
    if len(channels) < minimum:
        raise ValueError(
            f"anomaly variant {variant!r} needs >= {minimum} channels, got {len(channels)} "
            f"({channels}). Cross-channel families are undefined for univariate data — "
            f"drop this variant from the entity plan (see build_toy_fed_uni.py)."
        )


def inject_spike(test: np.ndarray, start: int, stop: int, channels: list[int],
                 rng: np.random.Generator, magnitude: float = 6.0) -> str:
    """Brief multiplicative spike on selected channels."""
    sign = rng.choice([-1.0, 1.0], size=(1, len(channels)))
    test[start: stop, channels] += sign * magnitude
    return f"point spike on {len(channels)} channels (magnitude={magnitude:.1f})"


def inject_level_shift(test: np.ndarray, start: int, stop: int, channels: list[int],
                       rng: np.random.Generator, magnitude: float = 3.0) -> str:
    test[start: stop, channels] += rng.uniform(magnitude * 0.8, magnitude * 1.2,
                                               size=(1, len(channels)))
    return f"sustained level shift on {len(channels)} channels"


def inject_ramp(test: np.ndarray, start: int, stop: int, channels: list[int],
                rng: np.random.Generator, peak: float = 3.0) -> str:
    width = stop - start
    ramp = np.linspace(0.0, peak, width, dtype=np.float32)[:, None]
    test[start: stop, channels] += ramp
    return f"linear ramp drift on {len(channels)} channels (peak={peak:.1f})"


def inject_frequency_shift(test: np.ndarray, start: int, stop: int, channels: list[int],
                           rng: np.random.Generator, new_period: float = 8.0) -> str:
    """Add an extra fast sinusoid on top of the existing signal (frequency contamination)."""
    width = stop - start
    t = np.arange(width, dtype=np.float32)
    extra = 1.8 * np.sin(2.0 * np.pi * t / new_period)
    test[start: stop, channels] = test[start: stop, channels] + extra[:, None]
    return f"frequency contamination on {len(channels)} channels (extra P={new_period})"


def inject_phase_jump(test: np.ndarray, start: int, stop: int, channels: list[int],
                      rng: np.random.Generator, period: int = 32) -> str:
    """Hard phase reset: drop a fresh sinusoidal segment of the entity's period."""
    width = stop - start
    t = np.arange(width, dtype=np.float32)
    new_phase = float(rng.uniform(np.pi / 2, 3 * np.pi / 2))
    # Replace by an offset sinusoid + retained baseline
    baseline = test[start: stop, channels].mean(axis=0, keepdims=True)
    test[start: stop, channels] = (
        baseline + 1.2 * np.sin(2.0 * np.pi * t[:, None] / period + new_phase)
    )
    return f"phase jump (Δφ≈{new_phase:.2f} rad) on {len(channels)} channels"


def inject_period_break(test: np.ndarray, start: int, stop: int, channels: list[int],
                        rng: np.random.Generator, period: int = 32) -> str:
    width = stop - start
    new_period = float(period) / float(rng.uniform(1.5, 2.2))
    t = np.arange(width, dtype=np.float32)
    baseline = test[start: stop, channels].mean(axis=0, keepdims=True)
    test[start: stop, channels] = (
        baseline + 1.0 * np.sin(2.0 * np.pi * t[:, None] / new_period)
    )
    return f"period break: {period} -> {new_period:.1f} on {len(channels)} channels"


def inject_amplitude_drop(test: np.ndarray, start: int, stop: int, channels: list[int],
                          rng: np.random.Generator, factor: float = 0.25) -> str:
    """Multiply selected channels by a small factor (attenuation)."""
    mean = test[start: stop, channels].mean(axis=0, keepdims=True)
    test[start: stop, channels] = mean + factor * (test[start: stop, channels] - mean)
    return f"amplitude drop (×{factor}) on {len(channels)} channels"


def inject_amplitude_burst(test: np.ndarray, start: int, stop: int, channels: list[int],
                           rng: np.random.Generator, factor: float = 3.0) -> str:
    """Multiply selected channels by a large factor (amplitude blow-up)."""
    mean = test[start: stop, channels].mean(axis=0, keepdims=True)
    test[start: stop, channels] = mean + factor * (test[start: stop, channels] - mean)
    return f"amplitude burst (×{factor}) on {len(channels)} channels"


def inject_noise_burst(test: np.ndarray, start: int, stop: int, channels: list[int],
                       rng: np.random.Generator, sigma: float = 1.5) -> str:
    width = stop - start
    test[start: stop, channels] += rng.normal(0.0, sigma, size=(width, len(channels)))
    return f"noise burst (σ+={sigma}) on {len(channels)} channels"


def inject_flatline(test: np.ndarray, start: int, stop: int, channels: list[int],
                    rng: np.random.Generator) -> str:
    """Freeze each channel at its value at `start - 1`."""
    if start == 0:
        held = test[0, channels]
    else:
        held = test[start - 1, channels]
    test[start: stop, channels] = held
    return f"flatline (sensor stuck) on {len(channels)} channels"


def inject_correlation_break(test: np.ndarray, start: int, stop: int, channels: list[int],
                             rng: np.random.Generator) -> str:
    """For each pair, copy the first onto the second negated - destroys
    whatever phase relationship existed."""
    _require_channels("correlation_break", channels, 2)
    a, b = channels[0], channels[1]
    test[start: stop, b] = -test[start: stop, a]
    if len(channels) >= 4:
        c, d = channels[2], channels[3]
        test[start: stop, d] = np.roll(test[start: stop, c], (stop - start) // 3)
    return f"correlation break between channels {channels}"


def inject_lag_shift(test: np.ndarray, start: int, stop: int, channels: list[int],
                     rng: np.random.Generator, period: int = 32) -> str:
    """Shift one channel by half a period relative to its normal phase."""
    _require_channels("lag_shift", channels, 1)
    width = stop - start
    shift = max(1, period // 2)
    src = test[start: stop, channels[0]]
    shifted = np.roll(src, shift)
    test[start: stop, channels[0]] = shifted
    return f"lag shift by {shift} on channel {channels[0]}"


def inject_sync_break(test: np.ndarray, start: int, stop: int, channels: list[int],
                      rng: np.random.Generator, period: int = 32) -> str:
    """Apply different per-channel phase shifts; destroys cross-channel sync
    while keeping per-channel statistics close to normal."""
    width = stop - start
    for c in channels:
        shift = int(rng.integers(period // 4, period))
        test[start: stop, c] = np.roll(test[start: stop, c], shift)
    return f"sync break (random shifts) across {len(channels)} channels"


def inject_dependency_break(test: np.ndarray, start: int, stop: int, channels: list[int],
                            rng: np.random.Generator) -> str:
    """Permute channels' values among themselves; preserves marginals but breaks
    the channel identity. Requires len(channels) >= 2."""
    _require_channels("dependency_break", channels, 2)
    perm = rng.permutation(len(channels))
    while np.all(perm == np.arange(len(channels))):
        perm = rng.permutation(len(channels))
    sub = test[start: stop, channels].copy()
    test[start: stop, channels] = sub[:, perm]
    return f"dependency break (permutation) on channels {channels}"


def inject_group_spike(test: np.ndarray, start: int, stop: int, channels: list[int],
                       rng: np.random.Generator, magnitude: float = 4.0) -> str:
    """Coherent spike across many channels at once."""
    test[start: stop, channels] += magnitude
    return f"coherent group spike on {len(channels)} channels"


def inject_group_drift(test: np.ndarray, start: int, stop: int, channels: list[int],
                       rng: np.random.Generator, peak: float = 2.0) -> str:
    width = stop - start
    drift = np.linspace(0.0, peak, width, dtype=np.float32)[:, None]
    test[start: stop, channels] += drift
    return f"coherent group drift on {len(channels)} channels (peak={peak:.1f})"


def inject_group_correlation(test: np.ndarray, start: int, stop: int, channels: list[int],
                             rng: np.random.Generator) -> str:
    """Replace all targeted channels with the first one (perfect correlation -
    a regime change in cross-channel structure)."""
    _require_channels("group_correlation", channels, 2)
    src = test[start: stop, channels[0]].copy()
    for c in channels[1:]:
        test[start: stop, c] = src
    return f"group correlation collapse on {len(channels)} channels"


def inject_contextual_point(test: np.ndarray, start: int, stop: int, channels: list[int],
                            rng: np.random.Generator) -> str:
    """Set the window to a value that is normal globally but anomalous locally
    (the channel mean over the whole test segment, applied during a high-amplitude
    crest)."""
    for c in channels:
        global_mean = float(test[:, c].mean())
        test[start: stop, c] = global_mean
    return f"contextual point (global mean held in window) on {len(channels)} channels"


def inject_missed_beat(test: np.ndarray, start: int, stop: int, channels: list[int],
                       rng: np.random.Generator) -> str:
    """Replace the window by a constant baseline (a missed cycle)."""
    for c in channels:
        test[start: stop, c] = float(test[start - 1, c]) if start > 0 else 0.0
    return f"missed beat (dropout) on {len(channels)} channels"


def inject_waveform_swap(test: np.ndarray, start: int, stop: int, channels: list[int],
                         rng: np.random.Generator, period: int = 32) -> str:
    """Replace the window by a sawtooth waveform (morphological change vs the
    sinusoidal normal)."""
    width = stop - start
    t = np.arange(width, dtype=np.float32)
    saw = (2.0 * ((t / period) - np.floor(0.5 + t / period))).astype(np.float32)
    saw = 1.4 * saw
    for c in channels:
        baseline = float(test[start: stop, c].mean())
        test[start: stop, c] = baseline + saw
    return f"waveform swap (sine -> sawtooth) on {len(channels)} channels"


def inject_double_beat(test: np.ndarray, start: int, stop: int, channels: list[int],
                       rng: np.random.Generator, period: int = 32) -> str:
    """Insert a double-frequency oscillation (extra-systole-like)."""
    width = stop - start
    t = np.arange(width, dtype=np.float32)
    extra = 1.5 * np.sin(4.0 * np.pi * t / period)            # double frequency
    for c in channels:
        test[start: stop, c] = test[start: stop, c] + extra
    return f"double-beat (extra-systole-like) on {len(channels)} channels"


def inject_invalid_two_active(test: np.ndarray, start: int, stop: int, channels: list[int],
                              rng: np.random.Generator) -> str:
    """For one-hot-like discrete channels: force two channels to be 1 at once."""
    # Treat the channels as binary; force two of them high, rest low.
    _require_channels("invalid_two_active", channels, 2)
    n_total = test.shape[1]
    test[start: stop, :] = 0.0
    test[start: stop, channels[0]] = 1.0
    test[start: stop, channels[1]] = 1.0
    return f"invalid one-hot: two channels active ({channels[0]}, {channels[1]})"


def inject_invalid_all_zero(test: np.ndarray, start: int, stop: int, channels: list[int],
                            rng: np.random.Generator) -> str:
    test[start: stop, :] = 0.0
    return "invalid one-hot: all channels zero"


def inject_stuck_category(test: np.ndarray, start: int, stop: int, channels: list[int],
                          rng: np.random.Generator) -> str:
    """Force the one selected channel to 1, all others to 0, for the window."""
    _require_channels("stuck_category", channels, 1)
    test[start: stop, :] = 0.0
    test[start: stop, channels[0]] = 1.0
    return f"stuck one-hot category at channel {channels[0]}"
