# The configurations of Table I, as commands

Every row of Table I is one `--arms` value plus a set of `--extra` flags, launched over a cohort.
The recipes below are the ones the paper's runs were dispatched with, recovered from the
orchestrator logs of those runs — not reconstructed from the code. `scripts/paper_runs.sh` prints
the same list and can run it.

Every reported training command explicitly enables per-window normalization:

```
--window-normalization zscore     each window is z-normalized before the transform
                                  (original method, §5.2/§A.1; passed explicitly, never ablated)
```
plus the window length `W = 2 × period`, which the **cohort** pins per series, not the command
line. The launcher refuses an `--extra` that touches a cohort-pinned axis.

The standard federated recipe is: `--protocol converged`, ceiling
`--s1-rounds 300 --s2-rounds 300`, `--fed-patience-rounds 6`, `--local-epochs 10`, `--batch 64`,
seed 0, fp16. Rows (h) and (i) override the stage-2 update rhythm and patience.
Local and centralized baselines instead use step ceilings and step-based patience, with
warmup and cosine decay. Their capped runs are retained in the paper; the federated
implementation flags a stage when its best round is its last round.

## Development set (cohort `ucr2p_10`, 10 series)

| Paper row | `--arms` | additional `--extra` | tag in `artifacts/runs/` |
|---|---|---|---|
| (a) local | `local` | — | `zn_main` |
| (b) centralized | `centralized` | — | `zn_main` |
| (c) codebook only | `federated_cb_only` | — | `zn_main` |
| (d) codebook only, averaged dictionary | `federated_fedavg_cb_only` | — | `zn_main` |
| (e) codebook and shared prior body | `federated` | — | `zn_main` |
| (f) aligned tokenizer, local prior (A1) | `federated_enc_fedavg` | `--fed-enc-cb suffstat --fed-enc-prior local --fed-enc-bn shared` | `zn_a1` |
| **(g) aligned tokenizer, shared prior body (A2)** | `federated_enc_fedavg` | `--fed-enc-cb suffstat --fed-enc-prior partial --fed-enc-bn shared` | `zn_a2` |
| (h) (g) with a step-counted rhythm | `federated_enc_fedavg` | (g) `+ --fed-s2-val fixed --tau-steps 64 --fed-s2-patience 29 --resume-from <(g) stage-1 ckpt>` | `zn_a2s2_tau64` |
| (i) (h) with a mobile dictionary | `federated_enc_fedavg` | (h) with `--fed-enc-cb fedavg` | `zn_cbfa_tau64` |
| (j) encoder FedAvg, local BN stats | `federated_enc_fedavg` | `--fed-enc-cb suffstat --fed-enc-prior local` | `zn_enc` |
| (k) (j) with FedProx | `federated_enc_fedprox` | idem (μ = 0.01, the CLI default) | `zn_enc` |
| (l) (j) with FedProto | `federated_enc_fedproto` | idem `+ --fedproto-agg uniform` | `zn_enc` |
| (m) (j) with a shared initialization only | `federated_enc_commoninit` | idem | `zn_enc` |
| (n) score fusion | — | not trained: `scripts/fusion_probe.py` over (a) | — |

Rows (j)–(m) differ from (f) by **one** thing: no `--fed-enc-bn shared`, so the BN running
statistics stay local. They are matched to one another and *not* to (f)/(g)/(h) — their baseline
is (j). This is the comparison that isolates the aggregation rule, and it is also where the κ
measurement lands at 0.77 instead of 1.00.

### The two diagnostics (not federation-legal)

| Row | `--arms` | `--extra` | tag |
|---|---|---|---|
| *pooled tokenizer* (pooled stage 1, federated stage 2) | `centraltok_fedprior` | `--fed-s2-val fixed --fedtokcp-stage1-root <centralized ckpt root>` | `zn_170_ctfp` |
| *pooled prior* (federated stage 1 of (g), stage 2 on pooled tokens) | `fedtok_centralprior` | `--fed-s2-val fixed --fedtokcp-stage1-root <(g) ckpt root>` | `zn_fedtokcp` |

Both take their stage 1 from an existing run's checkpoint tree, and both carry the fixed-mode
validation oracle (`--fed-s2-val fixed`). Their matched reference is therefore the arm that also
carries it (`zn_a2s2_ctrl`), not the (g) of Table III — the paper says so, and so does this file,
because getting it wrong changes the numbers.

### The stage-2 rhythm and its control

`--tau-steps` aggregates stage 2 every τ optimizer steps instead of every local epoch. Small τ
needs a low-noise validation signal, so (h) also carries `--fed-s2-val fixed` (fp32 validation with the recorded RNG handling), which is intended to reduce selection noise. The RNG limitation of the
current implementation is documented in [REPRODUCE.md](REPRODUCE.md#protocol-and-implementation-notes). The paired reference for
(h) is therefore `zn_a2s2_ctrl` = (g) + `--fed-s2-val fixed`, and about 60 % of the apparent τ
effect belongs to the oracle rather than to τ. On the confirmation sample that control was not
run, so (h) is compared with (g) there and the oracle travels with τ; the paper reports both
readings separately.

The less-frequent-aggregation probe of §VI-C is the same control at `--local-epochs 31`
(tag `zn_a2s2_lep`, also 16 and 24).

### The overtrained baseline

`zn_ot` is (a) and (b) again with early stopping disabled and a four-to-five times larger step
budget (`KEEP_LAST_WEIGHTS=1`). It exists because "federation beats local" must state *which*
local: against the patience-stopped baseline the count is 9/10, against the overtrained one it is
6/4 with an essentially unchanged median. `zn_es` is the matching early-stopped half.

### The floor

```bash
bash scripts/launch.sh --cohort ucr2p_10 --tag floor --engine floor --heads ma_c --modes local
```

The parameter-free reference of §VI-B: the squared residual of a centered moving average of
length 10, pushed through the same detection path. It has no parameters, so it admits no
federated variant.

## Confirmation sample (cohort `c50`, 50 series drawn at random, 48 analyzed)

Three arms only, all pre-registered before any of their numbers existed
([PREREGISTRATION.md](PREREGISTRATION.md)):

| Row | `--arms` | `--extra` | tag |
|---|---|---|---|
| local | `local` | — | `c50_local` |
| centralized | `centralized` | — | `c50_central` |
| (g) | `federated_enc_fedavg` | `--fed-enc-cb suffstat --fed-enc-prior partial --fed-enc-bn shared` | `c50_a2` |

`c50_tau64` is (h) on the same sample, added after the pre-registration and reported as a null.

Two series (`ucr_190`, `ucr_240`) were discarded from **all** arms by the pre-registered rule: the
federated arm aborted on non-finite entries in the aggregated codebook.
inside the aggregation, not codebook collapse, and it tracks half-precision exposure rather than
detection difficulty.

## Reading the on-disk names

The result json uses the **bare** arm name, the checkpoint directory carries the knobs:

```
artifacts/runs/zn_a2/ucr_split_w2p/ucr_011__federated_enc_fedavg.json
artifacts/runs/zn_a2/ckpt/ucr_split_w2p/ucr_011/seed0/federated_enc_fedavg_bn-shared_prior-partial/
```

That is how two configurations of one arm avoid colliding on disk, and it is a standing trap:
`latent_probe.py` wants the directory name with the knobs, `paper_numbers.py` reads the bare one.

⚠️ The cohort fingerprint covers the series list, the window, the tolerance and the seeds — it
does **not** cover `--window-normalization`, `--fed-enc-*` or the AMP dtype. Two cells can share a
fingerprint and still be different models. The protection is a distinct `--tag` per configuration,
which is why the table above lists one.
