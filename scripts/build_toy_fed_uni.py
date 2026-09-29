"""Build toy_fed_uni: the UNIVARIATE federated benchmark (C=1, clustered clients).

Companion to ``build_toy_fed.py`` (fixed C=8). Here every client owns exactly ONE
univariate series, and clients are partitioned into **6 clusters = 6 machine
types**A cluster is what you federate over: within it the morphology and the
base period are FIXED, so the clients are coherent enough to share a codebook and
a prior body; across clusters they are not.

Why univariate at all
---------------------
At C=8 the encoder is a grouped conv with ``groups=C``, so a federation is only
architecturally possible under a fixed-C regime. At C=1 ``groups=1``: every client
is structurally identical regardless of its domain, and the whole stack (encoder,
decoder, prior) becomes federable — not just the codebook. The cost is that every
channel-attribution metric degenerates (see `metrics_core.channel_localization_at_k`,
which now refuses to score C=1) — that part of the multivariate story is gone.

Cluster design (the non-IID axes)
---------------------------------
  * BETWEEN clusters : waveform morphology + base period. Strong separation.
  * WITHIN a cluster : jitter level, rng realisation, and the per-entity shape
    parameters each waveform samples (duty cycle, ring decay, harmonic weights,
    QRS geometry, …). "Same type of machine, different unit."

Amplitude and offset are deliberately NOT non-IID axes: the pipeline's
per-entity standardiser (``scaling="per_entity_standard"``) removes them.
Keep ``cfg.dataset.window_normalization = "none"`` — a per-window z-score would
erase ``level_shift`` outright, since at C=1 the offset is the only signal.

Anomaly families (all univariate-legitimate; width = base period P, i.e. at least
one latent time-token's receptive field):
    level_shift   - sustained additive offset
    amp_burst     - multiplicative amplitude blow-up around the window mean
    period_break  - the cycle is replaced by an off-period oscillation
``period_break`` is HELD OUT of ~1/3 of clients (unseen-anomaly-family test).

NOTE the C=8 builder used ``corr_break`` as its third family. At C=1 that injector
is a no-op, and it used to return a description string while the writer stamped a
label anyway — a labelled anomaly with zero signal deviation. `_toy_common` now
raises instead of silently no-op'ing, so that class of bug cannot recur.

Outputs (per build_and_write): train/ val/ test/ test_label/ full/ full_label/
test_clean/ test_mask/ events.csv labels.csv metadata.json clusters.json, PLUS:
  * probe/<cluster>.npy — a PUBLIC normal series per cluster (held-out seed, so
    data-independent of every client) for server-side codebook init and the
    cross-client token-agreement diagnostic. Use the cluster probe when you
    federate one cluster.
  * probe/probe.npy     — all cluster probes concatenated; the default global probe.

Usage:
    python scripts/build_toy_fed_uni.py --tier dev
    python scripts/build_toy_fed_uni.py --tier full --train-length 2048
    python scripts/build_toy_fed_uni.py --print-clusters      # comma lists for --clients
    python scripts/build_toy_fed_uni.py --wsd-twin            # synthetic twin of wsd_fed (see below)
    python scripts/build_toy_fed_uni.py --ucr-twin            # lengths sampled from UCR (see below)

The `--wsd-twin` mode builds `toy_fed_uni_wsdlike`: a synthetic dataset with the
SAME setting as the real wsd_fed benchmark — 4 clusters, cluster sizes (5/11/9/6)
and per-client (train, val, test) lengths COPIED verbatim from wsd, val=1440,
metrics_tolerance=14, window=256 (no top-level `period`). Cluster names c0..c3
match wsd, so the same `--clusters c0,c2,c3` driver invocations run on it verbatim.
Only the signal is synthetic — the federation topology and per-client data budget
are identical to wsd, so you can say "same setting, synthetic (controlled) data".

The `--ucr-twin` mode builds `toy_fed_uni_ucrlike`: the standard 6 machine-type
clusters (47 clients in dev), but each client's (train, test) length is SAMPLED
from the UCR Anomaly Archive 2021 (one series drawn per client, paired train:test),
val=1440. Sampled train is capped at `--ucr-cap` (default 32000) — UCR runs to 250k
train steps, untrainable for a federated toy — so series above the cap are dropped
from the pool and sampled tests are clipped to it. Only the signal is synthetic;
the length DISTRIBUTION comes from UCR (a wider/longer regime than wsd).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _toy_common import (
    AnomalyEvent,
    EntitySpec,
    build_and_write,
    generate_normal_waveform,
    inject_amplitude_burst,
    inject_level_shift,
    inject_period_break,
    pick_channels,
)

C_UNIVARIATE = 1
FAMILIES = ("level_shift", "amp_burst", "period_break")
HELD_OUT_FAMILY = "period_break"

# ── The 6 machine types. `n_dev` / `n_full` = clients in that cluster per tier ──
# Base periods are all divisible by 4 (see `_toy_common.multivariate_period`) and
# <= window_length/2 = 128, so a 256-step training window always covers >= 2 cycles.
#
# Client counts (2026-07-13): every cluster now has >= 6 clients, variable across
# clusters (6..11), mirroring the real wsd_fed spread (5..11). Two clients was too
# few to federates — FedAvg over 2 is hardly an average and amplifies noise. Within-
# cluster diversity is GENUINE at any count: `_toy_common.build_and_write` draws an
# independent rng per entity and every waveform resamples its shape params (duty
# cycle, ring decay, harmonic weights, QRS geometry) from it, so client 6..11 is a
# distinct unit, not a reseeded duplicate. The real scarcity knob stays train_length
# (see the _scarce/_scarcer builds), NOT client count — so n_full mirrors n_dev.
CLUSTERS: tuple[dict, ...] = (
    {"cluster": "M1_rotary",  "waveform": "sine",   "period":  32, "n_dev":  6, "n_full":  6},
    {"cluster": "M2_valve",   "waveform": "square", "period":  64, "n_dev":  7, "n_full":  7},
    {"cluster": "M3_pump",    "waveform": "saw",    "period":  48, "n_dev":  9, "n_full":  9},
    {"cluster": "M4_cardiac", "waveform": "pulse",  "period": 128, "n_dev": 11, "n_full": 11},
    {"cluster": "M5_bearing", "waveform": "ring",   "period":  64, "n_dev":  8, "n_full":  8},
    {"cluster": "M6_drive",   "waveform": "am",     "period":  96, "n_dev":  6, "n_full":  6},
)

_JITTERS = ("mild", "medium", "strong")

# ── The wsd twin (`--wsd-twin`): a synthetic dataset with the SAME setting as the
# real wsd_fed benchmark — 4 clusters, C=1, cluster sizes + per-client (train,val,
# test) lengths COPIED verbatim from wsd, val=1440, metrics_tolerance=14, window=256
# (no top-level `period`, exactly like wsd). Only the SIGNAL is synthetic. Cluster
# names c0..c3 match wsd so the same `--clusters c0,c2,c3` driver calls work verbatim.
# Sizes are NOT set here — they are read from wsd's metadata (5/11/9/6). Each wsd
# cluster is given one toy morphology (period <= 128 so window=256 covers >= 2 cycles).
WSD_TWIN_CLUSTERS: tuple[dict, ...] = (
    {"cluster": "c0", "waveform": "sine",   "period":  32},
    {"cluster": "c1", "waveform": "square", "period":  64},
    {"cluster": "c2", "waveform": "saw",    "period":  48},
    {"cluster": "c3", "waveform": "pulse",  "period": 128},
)


def make_entities(tier: str, n_channels: int = C_UNIVARIATE) -> tuple[dict, ...]:
    """Flat entity list, cluster-major. Client i of a cluster takes jitter i % 3,
    so a 4th/5th client is a reseed of an earlier jitter with a different rng."""
    key = "n_dev" if tier == "dev" else "n_full"
    entities: list[dict] = []
    for spec in CLUSTERS:
        for i in range(spec[key]):
            entities.append({
                "entity_id": f"uni_{len(entities):02d}",
                "cluster": spec["cluster"],
                "waveform": spec["waveform"],
                "period": spec["period"],
                "n_channels": n_channels,
                "jitter": _JITTERS[i % len(_JITTERS)],
            })
    return tuple(entities)


def make_wsd_twin_entities(wsd_meta: dict, n_channels: int = C_UNIVARIATE
                           ) -> tuple[tuple[dict, ...], dict[str, tuple[int, int, int]]]:
    """Build the wsd-twin entity grid: cluster sizes and per-client (train,val,test)
    lengths are taken verbatim from wsd's metadata, one synthetic client per real KPI.

    Returns (entities, entity_lengths). Client i within a cluster takes jitter i%3,
    so the within-cluster axis (jitter + rng + resampled shape params) still varies.
    """
    ents = wsd_meta.get("entities", [])
    if not ents:
        raise SystemExit("wsd metadata has no `entities` — cannot copy per-client lengths.")
    by_cluster: dict[str, list[dict]] = {}
    for e in ents:
        by_cluster.setdefault(e["cluster"], []).append(e)

    spec_of = {c["cluster"]: c for c in WSD_TWIN_CLUSTERS}
    missing = set(by_cluster) - set(spec_of)
    if missing:
        raise SystemExit(f"wsd clusters {sorted(missing)} have no WSD_TWIN_CLUSTERS morphology; "
                         f"known: {sorted(spec_of)}")

    entities: list[dict] = []
    entity_lengths: dict[str, tuple[int, int, int]] = {}
    for cs in WSD_TWIN_CLUSTERS:                              # declaration order = c0..c3
        members = by_cluster.get(cs["cluster"], [])
        for i, we in enumerate(members):
            eid = f"uni_{len(entities):02d}"
            entities.append({
                "entity_id": eid,
                "cluster": cs["cluster"],
                "waveform": cs["waveform"],
                "period": cs["period"],
                "n_channels": n_channels,
                "jitter": _JITTERS[i % len(_JITTERS)],
            })
            entity_lengths[eid] = (int(we["train_length"]),
                                   int(we["val_length"]),
                                   int(we["test_length"]))
    return tuple(entities), entity_lengths


# ── The ucr twin (`--ucr-twin`): the 6 machine-type CLUSTERS — i.e. the SAME
# structure the standard builder now emits (47 clients in dev, `make_entities`) —
# but each client's (train, test) length is SAMPLED from the UCR Anomaly Archive
# 2021 instead of the uniform --train-length. One UCR series is drawn per client
# (paired train/test, so real UCR train:test ratios are preserved); val is fixed
# at 1440 (UCR has no validation split). The draw is bounded by `--ucr-cap` (default
# 32k) because the UCR tail runs to 250k train steps — untrainable for a federated
# toy: series with train > cap are rejected from the pool (the sub-cap shape is
# kept intact), and each sampled test is clipped to the cap. Only the SIGNAL is
# synthetic (toy morphologies); the length DISTRIBUTION comes from UCR.
DEFAULT_UCR_DIR = Path(
    "preprocessing/dataset/AnomalyDatasets_2021/"
    "UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData"
)
UCR_NAME_RE = re.compile(r"^(\d{3})_UCR_Anomaly_(.+?)_(\d+)_(\d+)_(\d+)\.txt$")
UCR_TWIN_VAL_LEN = 1440


def _ucr_total_len(path: Path) -> int:
    """Number of values in a UCR .txt. Most are one value per line, but 9 store
    the whole series on a single whitespace-separated line — count tokens, robust
    to both layouts."""
    with open(path, "r") as f:
        return sum(len(line.split()) for line in f)


def ucr_length_pool(ucr_dir: Path, cap: int) -> list[tuple[int, int]]:
    """Read the UCR archive and return the pool of (train_len, test_len) pairs
    eligible for sampling: train from the filename (train_stop), test = total −
    train_stop. Series with train_len > cap are dropped; test is clipped to cap."""
    files = sorted(ucr_dir.glob("*.txt"))
    if not files:
        raise SystemExit(
            f"No UCR .txt files under {ucr_dir}. Download the archive first "
            f"(UCR_TimeSeriesAnomalyDatasets2021) or pass --ucr-src."
        )
    pool: list[tuple[int, int]] = []
    for p in files:
        m = UCR_NAME_RE.match(p.name)
        if not m:
            continue
        train_len = int(m.group(3))              # train_stop
        if train_len > cap:
            continue                              # reject the long tail (keep sub-cap shape)
        test_len = _ucr_total_len(p) - train_len
        if test_len <= 0:
            continue
        pool.append((train_len, min(test_len, cap)))
    if not pool:
        raise SystemExit(f"UCR length pool empty at cap={cap} (all series exceed it?).")
    return pool


def make_ucr_twin_entities(tier: str, n_channels: int, ucr_dir: Path, cap: int,
                           seed: int) -> tuple[tuple[dict, ...], dict[str, tuple[int, int, int]], list[tuple[int, int]]]:
    """The standard `make_entities` grid (6 clusters), with each client assigned a
    (train, val=1440, test) triple sampled from the UCR length pool. Returns
    (entities, entity_lengths, pool)."""
    entities = make_entities(tier, n_channels)
    pool = ucr_length_pool(ucr_dir, cap)
    rng = np.random.default_rng(seed + 4242)     # length-sampling stream (independent of signal rng)
    idx = rng.integers(0, len(pool), size=len(entities))
    entity_lengths: dict[str, tuple[int, int, int]] = {}
    for e, j in zip(entities, idx):
        tr, te = pool[int(j)]
        entity_lengths[e["entity_id"]] = (tr, UCR_TWIN_VAL_LEN, te)
    return entities, entity_lengths, pool


def cluster_map(entities: tuple[dict, ...]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for e in entities:
        out.setdefault(e["cluster"], []).append(e["entity_id"])
    return out


def make_variant_plan(entities: tuple[dict, ...]) -> dict[str, list[str]]:
    """Each client gets 2 families. The held-out family is absent from ~1/3 of
    clients — and the rotation is over the WITHIN-CLUSTER index, so no cluster
    ends up with every client blind to it."""
    plan: dict[str, list[str]] = {}
    seen: dict[str, int] = {}
    for e in entities:
        i = seen.get(e["cluster"], 0)
        seen[e["cluster"]] = i + 1
        if i % 3 == 0:
            plan[e["entity_id"]] = ["level_shift", "amp_burst"]        # held-out family absent
        elif i % 3 == 1:
            plan[e["entity_id"]] = ["amp_burst", HELD_OUT_FAMILY]
        else:
            plan[e["entity_id"]] = ["level_shift", HELD_OUT_FAMILY]
    return plan


def apply_variant(test: np.ndarray, spec: EntitySpec, start: int, stop: int,
                  variant: str, rng: np.random.Generator) -> AnomalyEvent:
    # At C=1 this is always [0]; written generically so the builder also runs at C>1.
    chans = pick_channels(rng, spec.n_channels, k=max(1, spec.n_channels // 3))
    if variant == "level_shift":
        desc = inject_level_shift(test, start, stop, chans, rng, magnitude=3.0)
    elif variant == "amp_burst":
        desc = inject_amplitude_burst(test, start, stop, chans, rng, factor=3.0)
    elif variant == "period_break":
        desc = inject_period_break(test, start, stop, chans, rng, period=spec.period)
    else:
        raise ValueError(f"Unknown variant {variant!r} (univariate families: {FAMILIES})")
    return AnomalyEvent(start=start, stop=stop, channels=chans,
                        variant=variant, description=desc)


def write_public_probes(output_dir: Path, entities: tuple[dict, ...], length: int,
                        seed: int, n_channels: int,
                        cluster_specs: tuple[dict, ...] = CLUSTERS) -> dict[str, str]:
    """One PUBLIC normal series per cluster, from a held-out seed (no client's
    private data). `probe/probe.npy` is their concatenation — the global default.

    Federating a single cluster? Point the diagnostics at that cluster's probe;
    a probe drawn from a different machine type would make the cross-client
    token-agreement number meaningless.

    `cluster_specs` defaults to the 6 machine-type CLUSTERS; the wsd twin passes
    its own 4-cluster spec so the probe set matches its clusters.
    """
    probe_dir = output_dir / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    per_cluster: dict[str, str] = {}
    chunks: list[np.ndarray] = []
    for i, cs in enumerate(cluster_specs):
        spec = EntitySpec(entity_id=f"probe_{cs['cluster']}", period=cs["period"],
                          n_channels=n_channels, jitter="medium",
                          waveform=cs["waveform"], cluster=cs["cluster"])
        rng = np.random.default_rng(seed + i)
        probe = generate_normal_waveform(length, spec, rng).astype(np.float32)
        np.save(probe_dir / f"{cs['cluster']}.npy", probe)
        per_cluster[cs["cluster"]] = f"probe/{cs['cluster']}.npy"
        chunks.append(probe)
    np.save(probe_dir / "probe.npy", np.concatenate(chunks, axis=0))
    return per_cluster


def build_wsd_twin(args) -> None:
    """Build `toy_fed_uni_wsdlike`: the synthetic twin of wsd_fed (same setting)."""
    wsd_meta_path = Path(args.wsd_src) / "metadata.json"
    if not wsd_meta_path.exists():
        raise SystemExit(f"wsd metadata not found: {wsd_meta_path} (set --wsd-src)")
    wsd_meta = json.loads(wsd_meta_path.read_text(encoding="utf-8"))
    tol = int(wsd_meta.get("metrics_tolerance", 14))

    entities, entity_lengths = make_wsd_twin_entities(wsd_meta, args.n_channels)
    clusters = cluster_map(entities)
    name = args.dataset_name or "toy_fed_uni_wsdlike"
    out = args.output_dir or Path("data/raw") / name
    plan = make_variant_plan(entities)

    if args.print_clusters:
        print(f"wsd-twin  clients={len(entities)}  clusters={len(clusters)}  "
              f"metrics_tolerance={tol}  (window stays 256 — no top-level period)")
        for cs in WSD_TWIN_CLUSTERS:
            ids = clusters.get(cs["cluster"], [])
            tls = [entity_lengths[i][0] for i in ids]
            print(f"  {cs['cluster']:4s} waveform={cs['waveform']:6s} P={cs['period']:3d}  "
                  f"n={len(ids):2d}  train_len {min(tls)}..{max(tls)} (val 1440)  "
                  f"--clients {','.join(ids)}")
        return

    # Fit guard against the SHORTEST client's test length (lengths are per-client now).
    longest = max(c["period"] for c in WSD_TWIN_CLUSTERS)
    min_test = min(te for (_, _, te) in entity_lengths.values())
    if 5 * longest > min_test:
        raise SystemExit(f"shortest client test_length={min_test} too short for P={longest} "
                         f"(needs {5 * longest} for 2 anomalies + margins/gaps).")

    # Nominal top-level lengths = wsd medians; the per-client values in entities[] win.
    med_tr = int(statistics.median([v[0] for v in entity_lengths.values()]))
    med_te = int(statistics.median([v[2] for v in entity_lengths.values()]))
    sizes = ", ".join(f"{k}={len(v)}" for k, v in clusters.items())

    build_and_write(
        dataset_name=name,
        family_description=(
            f"wsd twin — synthetic univariate benchmark with the SAME setting as wsd_fed: "
            f"{len(entities)} clients in {len(clusters)} clusters ({sizes}), C={args.n_channels}, "
            f"per-client (train,val,test) lengths COPIED verbatim from wsd_fed, val=1440, "
            f"metrics_tolerance={tol}, window=256 (no top-level period). Between-cluster non-IID "
            f"= waveform morphology + base period; within-cluster = jitter + rng. Anomaly mix "
            f"{set(FAMILIES)}, {HELD_OUT_FAMILY} held out of ~1/3 of the clients. Only the signal "
            f"is synthetic — topology and per-client data budget are identical to wsd_fed."
        ),
        output_dir=out,
        entity_variant_plan=plan,
        apply_variant=apply_variant,
        make_normal=generate_normal_waveform,
        entities=entities,
        train_length=med_tr, val_length=1440, test_length=med_te,
        entity_lengths=entity_lengths,
        seed=args.seed,
        overwrite=args.overwrite,
        extra_meta={"metrics_tolerance": tol, "wsd_twin_of": "wsd_fed"},
    )

    per_cluster = write_public_probes(out, entities, args.probe_length,
                                      seed=args.seed + 10_000, n_channels=args.n_channels,
                                      cluster_specs=WSD_TWIN_CLUSTERS)
    (out / "probe" / "index.json").write_text(json.dumps(per_cluster, indent=2))
    print(f"  Public probes -> {out / 'probe'} "
          f"({len(per_cluster)} per-cluster + probe.npy global)")
    print(f"  Cluster map   -> {out / 'clusters.json'}")


def build_ucr_twin(args) -> None:
    """Build `toy_fed_uni_ucrlike`: the standard 6-cluster grid, but per-client
    (train, test) lengths sampled from the UCR Anomaly Archive 2021."""
    ucr_dir = args.ucr_src if args.ucr_src.is_absolute() else Path.cwd() / args.ucr_src
    entities, entity_lengths, pool = make_ucr_twin_entities(
        args.tier, args.n_channels, ucr_dir, args.ucr_cap, args.seed)
    clusters = cluster_map(entities)
    name = args.dataset_name or "toy_fed_uni_ucrlike"
    out = args.output_dir or Path("data/raw") / name
    plan = make_variant_plan(entities)

    tls = sorted(v[0] for v in entity_lengths.values())
    tes = sorted(v[2] for v in entity_lengths.values())
    if args.print_clusters:
        pool_tr = sorted(t for t, _ in pool)
        print(f"ucr-twin  clients={len(entities)}  clusters={len(clusters)}  "
              f"cap={args.ucr_cap}  pool={len(pool)}/250 UCR series "
              f"(pool train {pool_tr[0]}..{pool_tr[-1]})  val={UCR_TWIN_VAL_LEN}")
        for cs in CLUSTERS:
            ids = clusters[cs["cluster"]]
            ct = [entity_lengths[i][0] for i in ids]
            print(f"  {cs['cluster']:12s} waveform={cs['waveform']:6s} P={cs['period']:3d}  "
                  f"n={len(ids):2d}  train_len {min(ct)}..{max(ct)}  --clients {','.join(ids)}")
        print(f"  sampled train_len: min={tls[0]} med={int(statistics.median(tls))} max={tls[-1]}")
        print(f"  sampled test_len : min={tes[0]} med={int(statistics.median(tes))} max={tes[-1]}")
        return

    # Fit guard against the SHORTEST client's test length (lengths are per-client).
    longest = max(c["period"] for c in CLUSTERS)
    if 5 * longest > tes[0]:
        raise SystemExit(f"shortest client test_length={tes[0]} too short for P={longest} "
                         f"(needs {5 * longest} for 2 anomalies + margins/gaps).")

    med_tr = int(statistics.median(tls))
    med_te = int(statistics.median(tes))
    sizes = ", ".join(f"{k}={len(v)}" for k, v in clusters.items())

    build_and_write(
        dataset_name=name,
        family_description=(
            f"ucr twin — synthetic univariate benchmark: {len(entities)} clients in "
            f"{len(clusters)} machine-type clusters ({sizes}), C={args.n_channels}, one series "
            f"per client. Per-client (train,test) lengths are SAMPLED from the UCR Anomaly "
            f"Archive 2021 (one series drawn per client, paired train:test; val={UCR_TWIN_VAL_LEN}; "
            f"train capped at {args.ucr_cap}, pool={len(pool)}/250). Between-cluster non-IID = "
            f"waveform morphology + base period; within-cluster = jitter + rng. Anomaly mix "
            f"{set(FAMILIES)}, {HELD_OUT_FAMILY} held out of ~1/3 of the clients. Only the signal "
            f"is synthetic — the length distribution comes from UCR."
        ),
        output_dir=out,
        entity_variant_plan=plan,
        apply_variant=apply_variant,
        make_normal=generate_normal_waveform,
        entities=entities,
        train_length=med_tr, val_length=UCR_TWIN_VAL_LEN, test_length=med_te,
        entity_lengths=entity_lengths,
        seed=args.seed,
        overwrite=args.overwrite,
        extra_meta={"ucr_twin_of": "UCR_Anomaly_2021",
                    "ucr_train_cap": int(args.ucr_cap),
                    "ucr_pool_size": len(pool),
                    "length_sampling": "per-client (train,test) sampled from UCR, val fixed 1440"},
    )

    per_cluster = write_public_probes(out, entities, args.probe_length,
                                      seed=args.seed + 10_000, n_channels=args.n_channels)
    (out / "probe" / "index.json").write_text(json.dumps(per_cluster, indent=2))
    print(f"  Public probes -> {out / 'probe'} "
          f"({len(per_cluster)} per-cluster + probe.npy global)")
    print(f"  Cluster map   -> {out / 'clusters.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", choices=["dev", "full"], default="dev")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default: data/raw/toy_fed_uni (dev) or data/raw/toy_fed_uni_full (full)")
    p.add_argument("--dataset-name", type=str, default=None)
    p.add_argument("--n-channels", type=int, default=C_UNIVARIATE,
                   help="1 = univariate (the point of this builder). >1 runs the same "
                        "cluster design multivariate, for an apples-to-apples C ablation.")
    # Scarcity driver (the FL difficulty knob). The sweep builds several of these.
    p.add_argument("--train-length", type=int, default=1024)
    p.add_argument("--val-length", type=int, default=512)
    p.add_argument("--test-length", type=int, default=1536)
    p.add_argument("--probe-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--print-clusters", action="store_true",
                   help="print the cluster -> client mapping (and --clients strings) and exit")
    p.add_argument("--wsd-twin", action="store_true",
                   help="build the wsd twin: 4 clusters, cluster sizes + per-client (train,val,test) "
                        "lengths copied from wsd_fed, val=1440, metrics_tolerance=14, window=256. "
                        "Overrides --tier/--train-length. Default name: toy_fed_uni_wsdlike.")
    p.add_argument("--wsd-src", type=Path, default=Path("data/raw/wsd_fed"),
                   help="wsd dataset dir to copy the topology + per-client lengths from (--wsd-twin).")
    p.add_argument("--ucr-twin", action="store_true",
                   help="build the ucr twin: the standard 6-cluster grid but per-client "
                        "(train,test) lengths SAMPLED from the UCR Anomaly Archive 2021, val=1440. "
                        "Overrides --train-length. Default name: toy_fed_uni_ucrlike.")
    p.add_argument("--ucr-src", type=Path, default=DEFAULT_UCR_DIR,
                   help="UCR_Anomaly_FullData dir to sample per-client lengths from (--ucr-twin).")
    p.add_argument("--ucr-cap", type=int, default=32000,
                   help="max sampled train length (--ucr-twin); UCR series above it are dropped "
                        "from the pool and sampled tests are clipped to it (default: %(default)s).")
    args = p.parse_args()

    if args.wsd_twin:
        return build_wsd_twin(args)
    if args.ucr_twin:
        return build_ucr_twin(args)

    entities = make_entities(args.tier, args.n_channels)
    clusters = cluster_map(entities)

    if args.print_clusters:
        print(f"tier={args.tier}  clients={len(entities)}  clusters={len(clusters)}")
        for cs in CLUSTERS:
            ids = clusters[cs["cluster"]]
            print(f"  {cs['cluster']:12s} waveform={cs['waveform']:6s} P={cs['period']:3d}  "
                  f"--clients {','.join(ids)}")
        return

    name = args.dataset_name or ("toy_fed_uni" if args.tier == "dev" else "toy_fed_uni_full")
    out = args.output_dir or Path("data/raw") / name
    plan = make_variant_plan(entities)

    # `period_break` replaces the window with an oscillation at P / U(1.5, 2.2).
    # A window narrower than one base period would make that indistinguishable
    # from a phase glitch, so the writer's default width (= base period) is right;
    # the longest period (128) must still fit 2 anomalies + margins in test_length:
    #   2*128 (widths) + 1*128 (gap) + 2*128 (margins) = 640 <= test_length.
    longest = max(c["period"] for c in CLUSTERS)
    if 5 * longest > args.test_length:
        raise SystemExit(
            f"--test-length {args.test_length} too short: cluster with P={longest} needs "
            f"{5 * longest} steps for 2 anomalies + margins/gaps."
        )

    build_and_write(
        dataset_name=name,
        family_description=(
            f"Univariate federated benchmark ({args.tier}): {len(entities)} clients in "
            f"{len(clusters)} machine-type clusters, C={args.n_channels}, one series per "
            f"client. Between-cluster non-IID = waveform morphology + base period; "
            f"within-cluster = jitter + rng. Anomaly mix {set(FAMILIES)}, "
            f"{HELD_OUT_FAMILY} held out of ~1/3 of the clients in every cluster. "
            f"train_length={args.train_length} (scarcity driver)."
        ),
        output_dir=out,
        entity_variant_plan=plan,
        apply_variant=apply_variant,
        make_normal=generate_normal_waveform,
        entities=entities,
        train_length=args.train_length,
        val_length=args.val_length,
        test_length=args.test_length,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    # Public probes (held-out seed, offset from the dataset seed so no client's
    # rng stream can coincide with a probe's).
    per_cluster = write_public_probes(out, entities, args.probe_length,
                                      seed=args.seed + 10_000, n_channels=args.n_channels)
    (out / "probe" / "index.json").write_text(json.dumps(per_cluster, indent=2))
    print(f"  Public probes -> {out / 'probe'} "
          f"({len(per_cluster)} per-cluster + probe.npy global)")
    print(f"  Cluster map   -> {out / 'clusters.json'}")


if __name__ == "__main__":
    main()
