# Federated TimeVQVAE-AD

Code and experimental results for **Federated TimeVQVAE Anomaly Detection:
Prior Sharing Requires an Aligned Tokenizer** by Leonardo Schiavo, Donato Cerciello,
Ángel Panizo-Lledot, Javier Huertas Tato, Stefano Izzo, Fabio Giampaolo and David Camacho.

Read the [paper](paper/paper.pdf) or its [LaTeX source](paper/paper.tex).
The paper distributed here is the submitted manuscript; there is no separate anonymous edition.

## What the paper establishes

TimeVQVAE-AD first learns a VQ-VAE tokenizer and then fits a MaskGIT prior over its
discrete tokens. The experiments test which components of this two-stage detector
can be shared across clients.

The main configuration, **(g), also called A2**, averages the encoder, pools BatchNorm
running statistics, merges codebook assignment statistics, and shares the prior body.
The decoder, channel embedding and output bias remain local. The matched development
ablations find a benefit from prior sharing after tokenizer alignment; a shared
codebook with local encoders does not provide the same benefit.

The confirmation study draws 50 additional UCR series and analyzes 48 complete
paired tasks. Configuration (g) improves over local training on **32 of 48 series
in AUPRC** (two-sided sign test, **p = 0.029**). This is the secondary endpoint.
**No comparison is significant on the preregistered primary endpoint**, localization
accuracy at tolerance 64, including centralized versus local training. Matching
centralized localization on 37 of 48 series does not establish statistical equivalence.

The paper also analyzes a failure on `ucr_170`, where averaging propagates a false
alarm. Averaging standardized local score profiles repairs that observed case.
This fusion requires a common test stream and all five local models at inference;
it is not the same deployment setting as one federated model per client.

## Repository contents

| Path | Purpose |
|---|---|
| `paper/` | Submitted paper, bibliography and figures |
| `config.py`, `data.py`, `utils.py` | Configuration, window extraction, scaling, paths and checkpoints |
| `model/` | STFT, encoder, quantizer, decoder and MaskGIT prior |
| `pipeline/federated.py` | Client updates and server aggregation |
| `pipeline/federated_eval.py` | One experimental cell: series, configuration and seed |
| `pipeline/detect.py`, `metrics_core.py` | Score assembly and evaluation |
| `scripts/` | Dataset construction, launches, result analysis and checks |
| `cohorts/` | Development and confirmation cohort definitions |
| `artifacts/runs/` | Bundled summaries, client reports and local score profiles |
| `docs/` | Reproduction instructions, configurations and preregistration |

Raw datasets, model checkpoints and token caches are not distributed.
The implementation simulates client training and server aggregation within one process.
Raw training windows stay local in the federated algorithms; the centralized and
pooled-stage diagnostics explicitly combine data. The aggregation itself provides
no formal privacy guarantee.

## Install and check the results

Use Python 3.10. Training the full study requires a CUDA GPU; the result checks run on CPU.

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

python scripts/paper_numbers.py
python scripts/c50_table.py
python scripts/fusion_probe.py --run artifacts/runs/zn_main --all-series
bash scripts/run_tests.sh
```

`requirements.txt` pins the recorded environment, including a CUDA 12.8 PyTorch build.
For a different CUDA version or a CPU-only installation, use a local requirements copy
without the `torch==...` and `--extra-index-url` lines, install the appropriate PyTorch
build, and install that requirements copy. Installing the original file afterward
would reapply its PyTorch pin.

`paper_numbers.py` checks the implemented numerical claims from the development
results; `c50_table.py` recomputes the confirmation table. Neither replaces the
checkpoint-based alignment and counterfactual analyses. The test runner reports
which checks pass, fail or are skipped; `--smoke` additionally trains short GPU jobs.

## Build the benchmark

Download the [UCR Time Series Anomaly Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip)
and extract it under:

```text
preprocessing/dataset/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData/
```

```bash
python scripts/build_ucr_split.py --name ucr_split_w2p --window-mode 2p \
    --shares 10,10,20,20,30 --val-pct 10
python scripts/check_ucr_split.py --root data/raw/ucr_split_w2p
```

Each series is a separate federated task. Its training segment is divided
chronologically into five disjoint client shards of 10/10/20/20/30%, with the final
10% reserved for validation. All clients use the original common test segment.
The split represents one sensor and regime; chronological shards are not guaranteed
to be independent and identically distributed.

The window length is twice the tabulated period, `T = 2P`. Eligibility requires
each client shard to contain at least `2T` samples, validation at least `T`, and
test at least `2T`; 180 of the 250 archive series are usable. The development cohort
contains 10 series. The disjoint confirmation cohort contains the 50 preregistered draws.

## Run a paper configuration

Print the launch recipes before running them:

```bash
bash scripts/paper_runs.sh
python scripts/cohort.py show ucr2p_10
python scripts/cohort.py verify c50
```

For configuration (g):

```bash
LAUNCH_GPUS="0" SLOTS_PER_GPU=1 bash scripts/launch.sh \
    --cohort ucr2p_10 --tag a2 --arms federated_enc_fedavg \
    --extra "--window-normalization zscore --fed-enc-cb suffstat --fed-enc-prior partial --fed-enc-bn shared"
```

The launcher pins `FEDVQ_AMP=fp16`. For direct Python invocations, set it explicitly
to reproduce that precision. **Pass `--window-normalization zscore` explicitly**:
the Python default is `none`, and the cohort files do not enforce normalization.
The cohort fingerprint identifies its dataset settings and seeds; it does not
cover the full experiment, including normalization and precision. Compare the
recorded configurations and use distinct tags for different settings.

Results are written under `artifacts/runs/<tag>/`; logs are under `logs/runs/<tag>/`.
The launcher skips cells whose output JSON already exists. This is completion-based
skipping, not a validation of existing results or automatic checkpoint recovery.
An existing tag should only be reused for the same configuration.

See [CONFIGURATIONS.md](docs/CONFIGURATIONS.md) for the mapping from paper rows to
flags and [REPRODUCE.md](docs/REPRODUCE.md) for the full workflow.

## Evaluation and reporting

- **Primary confirmation endpoint:** localization accuracy at tolerance 64.
  Each client returns its score maximum; client outcomes are averaged within a series.
- **Secondary endpoint:** AUPRC of the per-timestep anomaly score.
- **Statistical unit:** the series, with paired comparisons across configurations.
  The five clients are correlated shards of the same signal, not five independent tasks.
- **One training seed per cell:** seed 0. The confirmation draw uses seed 20260809.
- **Validation selection:** local and centralized training use step-based patience
  within fixed step ceilings; federated training uses round-based patience within
  fixed round ceilings. The baselines use warmup and cosine decay; the reported
  federated rounds use a constant learning rate.
- **Budget limits:** baseline runs that reach their step ceiling are retained in
  the paper. Federated stages whose selected best round is the last round are
  marked as truncated. The launcher reports those flags; it does not justify a
  blanket claim that every run reached convergence or every capped run was excluded.
- **Confirmation exclusions:** `ucr_190` and `ucr_240` are removed from every arm
  because the federated runs failed with non-finite codebook entries. The analysis
  follows the complete-pair rule described in [PREREGISTRATION.md](docs/PREREGISTRATION.md).

The library computes additional metrics, but the paper does not report AUROC,
threshold-based scores or range-aware scores as its comparison endpoints.
Counterfactual evaluation uses matched label-derived masks; the deployed explanation
uses the client's own train-fitted threshold. These are separate evaluation procedures.

## Credits and license

The detector is based on TimeVQVAE-AD by Lee, Malacarne and Aune and the TimeVQVAE
generative backbone. The implementation retains the upstream attribution and is
distributed under the [MIT license](LICENSE).
