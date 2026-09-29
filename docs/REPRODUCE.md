# Reproducing the paper

Three levels, cheapest first:

1. **Check the tables** from the shipped per-cell results — seconds, no GPU, no data.
2. **Check the code** — unit tests plus a smoke run of every arm at collapsed budgets, minutes.
3. **Re-run the campaign** — build the benchmark and launch the cohorts, GPU-weeks.

---

## 1. Check the tables (no GPU, no download)

`artifacts/runs/` ships the per-cell output of every run the paper reports: the per-series summary
json, the per-client `report.json`, and — for the `local` arm on the development set — the
per-timestep score profiles that the fusion analysis consumes. Checkpoints and token caches are
not shipped (54 GB).

```bash
python scripts/paper_numbers.py                      # Sections VI-A .. VI-E, claim by claim
python scripts/paper_numbers.py --json /tmp/n.json   # ... and dump the raw values
python scripts/c50_table.py                           # Table IV (confirmation sample, n=48)
python scripts/fusion_probe.py --run artifacts/runs/zn_main --all-series   # configuration (n)
```

`paper_numbers.py` prints one line per implemented numerical check — `[OK ]` or `[FAIL]`, with the
value it computed next to the value in the text — and exits non-zero if anything disagrees. It
starts with a gate: its own top-1 rule must reproduce the `local` column of Table III to the last
digit, because the fusion rows use that rule on profiles the pipeline never scored as an arm. If
the gate fails, the fusion numbers are declared unvalidated rather than printed as results.

`c50_table.py` recomputes Table IV — win/loss/tie, median, mean and a two-sided sign test with
ties excluded, on both endpoints — and checks it against the published table.

What the shipped results do **not** support: the κ measurement (`latent_probe.py`), the
counterfactual table (`cf_quality.py`) and `fig_overview.py` all read checkpoints, so they need a
run of their own. Their commands are in §4 below.

## 2. Check the code

All of it at once, with a PASS/FAIL line per check and logs under `artifacts/_tests/`:

```bash
bash scripts/run_tests.sh            # layers 1-3: static, unit, shipped results (CPU)
bash scripts/run_tests.sh --smoke    # + layer 4: every arm trained on a GPU
```

The layers separately. First the aggregation math — CPU, seconds each. The last of them trains
two clients for three rounds, so generate the synthetic dataset first (it takes under a second
and needs no download):

```bash
python scripts/build_toy_fed_uni.py            # 47 synthetic series, 6 clusters, generated locally

python scripts/fed_codebook_unittest.py        # the merged codebook equals the pooled
                                               # k-means M-step, and collecting stats does not
                                               # mutate the codebook
python scripts/fed_regression_unittest.py      # count-weighted FedAvg, the seed schedule, and
                                               # BN running stats never entering the linear mean
python scripts/fed_enc_algo_unittest.py        # FedAvg / FedProx / FedProto, incl. the FedProto
                                               # degeneracy: count-weighted prototypes ARE the
                                               # merged codebook; uniform ones are not
python scripts/fed_val_selection_unittest.py   # restoring a best round that is not the last one
python scripts/fed_cb_server_ema_unittest.py   # the server-side codebook EMA arm
```

Then one GPU, minutes — every reporting arm trains, saves and scores at collapsed budgets:

```bash
bash scripts/smoke_arms.sh            # toy dataset, 13 cases
bash scripts/smoke_arms.sh --full     # + one real UCR series (needs §3 first)
ARMS=federated_enc_fedavg bash scripts/smoke_arms.sh    # one arm
```

`TVQ_SMOKE=1` collapses the budgets but keeps the *converged* code path — validation selection,
patience, best-round restore — so what is exercised is what runs for real. The metrics a smoke
produces are meaningless and every line of its output says so.

## 3. Build the benchmark

Download the [UCR Time Series Anomaly
Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip)
and unzip it so that this exists:

```
preprocessing/dataset/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData/
```

Then:

```bash
python scripts/build_ucr_split.py --name ucr_split_w2p --window-mode 2p --shares 10,10,20,20,30 --val-pct 10
python scripts/check_ucr_split.py --root data/raw/ucr_split_w2p   # verify the bytes on disk
```

Each series becomes one federation of five clients holding contiguous, chronological, disjoint
slices of its own training segment (10/10/20/20/30 %), with the last 10 % held out for validation
and the archive's test segment and labels shared and never trained on. The split is deliberately homogeneous: clients observe the same sensor and regime.
Chronological shards are not guaranteed to be independent and identically distributed.

`--window-mode 2p` sets each series' window to twice its measured period, read from
`preprocessing/UCR_anomaly_dataset_periods.csv`. With that rule 180 of the 250 archive series are
usable: eligible series need a measured period, at least two window lengths of samples
in every client shard, one in validation and two in test.
The build asserts the partition property in memory; `check_ucr_split.py` re-asserts it on disk by
reading the source `.txt` back — the shards plus the validation tail must concatenate to the
original training segment exactly.

## 4. Re-run the campaign

A **cohort** pins which series, which window, which tolerance and which seeds a run covers, and
stamps a fingerprint into the manifest of every tag that names it. The fingerprint does not include normalization, precision or all training settings;
equal fingerprints alone do not establish matched experimental conditions. The two cohorts of the paper ship in `cohorts/`:

```bash
python scripts/cohort.py show ucr2p_10        # the 10 development series
python scripts/cohort.py verify c50           # the 50 of the confirmation sample
```

Launch one configuration over one cohort (see [CONFIGURATIONS.md](CONFIGURATIONS.md) for all of
them, or `bash scripts/paper_runs.sh` to print the whole list):

```bash
LAUNCH_GPUS="0 1" SLOTS_PER_GPU=3 bash scripts/launch.sh \
    --cohort ucr2p_10 --tag a2 --arms federated_enc_fedavg \
    --extra "--window-normalization zscore --fed-enc-cb suffstat --fed-enc-prior partial --fed-enc-bn shared"
```

Add `--dry` to see the job list without launching. The launcher skips a job whose output JSON
already exists. This does not validate its configuration or automatically resume a checkpoint. Results
land in `artifacts/runs/<tag>/` and logs in `logs/runs/<tag>/`.

The launcher ends with a convergence audit and prints any cell whose best round was its last one.
The paper marks these federated stages as truncated. This check is not a universal
convergence guarantee. Local and centralized runs that reach their step ceilings are retained
in the reported results. Reuse a tag only for an unchanged configuration; use a new tag when
changing budgets, precision, normalization or other settings.


One cell is one `pipeline/federated_eval.py` invocation and can be run directly:

```bash
FEDVQ_AMP=fp16 CUDA_VISIBLE_DEVICES=0 python pipeline/federated_eval.py \
    --dataset ucr_split_w2p --cluster ucr_011 --arms federated_enc_fedavg \
    --protocol converged --s1-rounds 300 --s2-rounds 300 --local-epochs 10 \
    --fed-patience-rounds 6 --batch 64 --seeds 0 --window-length 182 --metrics-tolerance 64 \
    --window-normalization zscore --fed-enc-cb suffstat --fed-enc-prior partial --fed-enc-bn shared \
    --out-dir artifacts/runs/a2/ckpt/ucr_split_w2p --out-json artifacts/runs/a2/ucr_split_w2p/ucr_011__federated_enc_fedavg.json
```

### The measurements that need a checkpoint tree

Once a tag has run, its checkpoints support the three analyses the shipped results cannot:

```bash
# kappa: do the five clients assign the same code to the same latent position?
python scripts/latent_probe.py --run-dir artifacts/runs/a2 --all-complete \
       --arm federated_enc_fedavg_bn-shared_prior-partial

# counterfactual quality: plausibility and specificity, paired on window and rewritten set
python scripts/cf_quality.py --series ucr_011,ucr_014,ucr_043,ucr_170,ucr_001 \
       --arms A2,local,centralized --tag-baselines main --tag-federated a2 --out cf_quality.json --dump cf_arrays/

# the counterfactual as the method actually produces it at inference: the mask is whatever the
# client's OWN threshold declares, no label in the loop. This is the procedure used in panel (c) of Fig. 2.
python scripts/cf_explainable.py --tag a2 --series ucr_011 --client 1 --starts 1850 --out cf_ucr_011.npz

# the two figures of the paper
python scripts/fig_framework.py                       # Fig. 1: the arrangement of configuration (g)
python scripts/fig_overview.py --series ucr_011 --tag a2 --cf-npz cf_ucr_011.npz --out fig_overview.pdf
python scripts/cf_figure.py --npz cf_arrays/A2_ucr_043.npz --out fig_counterfactual.pdf  # standalone
```

`fig_overview.py` trains nothing: it reads the built dataset, the `scores.npz` already on disk
under the tag, and a counterfactual dump. It accepts either dump — the label-derived one from
`cf_quality.py` or the label-free one from `cf_explainable.py` — and the published figure uses
the label-free one, as its caption says.

κ is a measured property of the checkpoints, not a label attached to the federated arm: under (g)
it is 1.00 on all ten series (the clients tokenize identically), under (e) the median is 0.07, and
the codebook is bit-identical in both — what separates them is whether the clients *use* it the
same way.

## Protocol and implementation notes

The [README](../README.md#evaluation-and-reporting) records the submitted reporting
scope, stopping rules, exclusions and statistical unit. Pass
`--window-normalization zscore` explicitly: the cohort does not enforce it. The launcher
pins fp16; direct invocations must set `FEDVQ_AMP=fp16` themselves.

The step-counted prior comparisons use the `--fed-s2-val fixed` implementation from the
recorded runs. Its validation helper scopes PyTorch's RNG, while MaskGIT also draws mask
ratios from Python's RNG. Repeated validation is therefore not fully deterministic in
the current implementation. The distributed result files retain the recorded outcomes.

Compiling the included manuscript requires `pdflatex`, `bibtex` and IEEEtran. The
canonical build is maintained in the research workspace; the exported `paper/` files
contain the same source, bibliography and figures.
