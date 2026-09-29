#!/usr/bin/env bash
# paper_runs.sh — the campaign behind the paper, as commands.
#
#   bash scripts/paper_runs.sh                 # print every configuration (nothing runs)
#   bash scripts/paper_runs.sh --run g         # launch configuration (g) over the dev cohort
#   bash scripts/paper_runs.sh --run dev       # launch (a)..(m) in dependency order
#   bash scripts/paper_runs.sh --run c50       # launch the confirmation sample
#   LAUNCH_GPUS="0 1" SLOTS_PER_GPU=3 bash scripts/paper_runs.sh --run g
#
# The recipes are the ones the paper's runs were dispatched with. Table I of the paper and
# docs/CONFIGURATIONS.md name the same rows; this file is the executable copy, so a row that
# changes here must change there.
#
# ORDER MATTERS. (h) and (i) resume stage 1 from an earlier tag's checkpoints, and the two
# diagnostics read a whole checkpoint tree, so they cannot run before the tag they read exists.
# --run dev respects that; a single --run <row> does not check it and will fail loudly if the
# tree it needs is missing.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

ZN='--window-normalization zscore'
A2X="--fed-enc-cb suffstat --fed-enc-prior partial --fed-enc-bn shared"
ENCX="--fed-enc-cb suffstat --fed-enc-prior local"
TAU="--fed-s2-val fixed --tau-steps 64 --fed-s2-patience 29"
DEV_SERIES="ucr_001 ucr_011 ucr_014 ucr_043 ucr_082 ucr_083 ucr_086 ucr_170 ucr_222 ucr_229"

usage() {
cat <<'EOF'
rows:
  a,b,c,d,e   local · centralized · codebook only (suff.stat.) · codebook only (FedAvg dict) ·
              codebook + shared prior body, all with LOCAL encoders          -> tag zn_main
  f           (A1) aligned tokenizer, local prior                            -> tag zn_a1
  g           (A2) aligned tokenizer, shared prior body  [the main arm]      -> tag zn_a2
  h           (g) with a step-counted stage-2 rhythm, tau=64                 -> tag zn_a2s2_tau64
  ctrl        (g) + the frozen-mask validation oracle -- the PAIRED REFERENCE of h
  i           (h) with a mobile (FedAvg'd) dictionary                        -> tag zn_cbfa_tau64
  j,k,l,m     encoder aggregation rule at a local prior and LOCAL BN stats   -> tag zn_enc
  ot          the overtrained local/centralized baseline                     -> tag zn_ot
  floor       the parameter-free moving-average floor                        -> tag zn_floor
  diag        the two pooled-stage diagnostics (not federation-legal)
  c50-*       the three pre-registered arms of the confirmation sample
groups:
  dev         a,b,c,d,e -> f -> g -> ctrl -> h -> j,k,l,m -> ot -> floor
  c50         the three c50 rows
EOF
}

launch() {   # launch <cohort> <tag> <arms> <extra>
  echo "+ launch.sh --cohort $1 --tag $2 --arms $3 --extra \"$4\""
  [[ $RUN -eq 1 ]] || return 0
  bash scripts/launch.sh --cohort "$1" --tag "$2" --arms "$3" --extra "$4"
}

# Rows that resume stage 1 from an existing tag. One command PER SERIES, because --resume-from
# names one checkpoint directory: the series cannot be batched into a single launch the way the
# from-scratch rows can. In print mode only the first is shown -- the rest differ by the series
# name alone.
resume_row() {   # resume_row <tag> <from-tag> <ckpt-dirname> <extra> [series...]
  local tag="$1" from="$2" dirname="$3" extra="$4"; shift 4
  local list="${*:-$DEV_SERIES}" s n=0
  for s in $list; do
    local ck="$REPO/artifacts/runs/$from/ckpt/ucr_split_w2p/$s/seed0/$dirname"
    if [[ $RUN -eq 0 ]]; then
      n=$((n+1))
      [[ $n -gt 1 ]] && continue
      echo "+ LAUNCH_ONLY_CLUSTERS=$s launch.sh --cohort ucr2p_10 --tag $tag --arms federated_enc_fedavg \\"
      echo "    --extra \"$extra --resume-from $ck\""
      continue
    fi
    [[ -d "$ck" ]] || { echo "  !! missing $ck -- run tag '$from' first"; return 2; }
    LAUNCH_ONLY_CLUSTERS="$s" bash scripts/launch.sh --cohort ucr2p_10 --tag "$tag" \
        --arms federated_enc_fedavg --extra "$extra --resume-from $ck"
  done
  [[ $RUN -eq 0 ]] && echo "  # ... one such command per series, $n in all: $list"
  return 0
}

row() {
  case "$1" in
    a|b|c|d|e|a,b,c,d,e) launch ucr2p_10 zn_main \
        "centralized,local,federated,federated_cb_only,federated_fedavg_cb_only" "$ZN";;
    f)    launch ucr2p_10 zn_a1 federated_enc_fedavg \
              "$ZN --fed-enc-cb suffstat --fed-enc-prior local --fed-enc-bn shared";;
    g)    launch ucr2p_10 zn_a2 federated_enc_fedavg "$ZN $A2X";;
    ctrl) resume_row zn_a2s2_ctrl zn_a2 federated_enc_fedavg_bn-shared_prior-partial \
              "$ZN $A2X --fed-s2-val fixed --fed-s2-agg-penalty --fed-s2-snapshot-every 4";;
    h)    resume_row zn_a2s2_tau64 zn_a2 federated_enc_fedavg_bn-shared_prior-partial \
              "$ZN $A2X $TAU";;
    i)    # Three series only, and that is what Table I reports as n=3: (i) is the factorial
          # cell that answers "does the mobile dictionary help where the frozen one binds?",
          # which is decided on ucr_011 and ucr_043 (where it loses) and ucr_170 (where it wins).
          echo "# (i) needs its own stage 1 first: the mobile dictionary changes stage 1."
          launch ucr2p_10 zn_170_cbfa federated_enc_fedavg \
              "$ZN --fed-s2-val fixed --fed-enc-prior partial --fed-enc-bn shared --fed-enc-cb fedavg"
          resume_row zn_cbfa_tau64 zn_170_cbfa \
              federated_enc_fedavg_bn-shared_cb-fedavg_prior-partial \
              "$ZN --fed-enc-cb fedavg --fed-enc-prior partial --fed-enc-bn shared $TAU" \
              ucr_011 ucr_043 ucr_170;;
    j|k|l|m|j,k,l,m) launch ucr2p_10 zn_enc \
        "federated_enc_fedavg,federated_enc_fedprox,federated_enc_fedproto,federated_enc_commoninit" \
        "$ZN $ENCX --fedproto-agg uniform";;
    ot)   # EARLY_STOPPING=0 alone is NOT enough: the converged loop restores best-on-val at
          # the end anyway, so it would return the same model hours later. KEEP_LAST_WEIGHTS=1
          # is what makes it a different condition. The ceiling (25k/100k steps) leaves ~15k
          # beyond the latest early stop ever observed among converged clients (step 9552).
          echo "# the overtrained baseline: patience off, weights of the LAST step, 4-5x budget"
          echo "+ S1_MAX_STEPS=25000 S2_MAX_STEPS=100000 EARLY_STOPPING=0 KEEP_LAST_WEIGHTS=1 \\"
          echo "    launch.sh --cohort ucr2p_10 --tag zn_ot --arms local,centralized --extra \"$ZN\""
          [[ $RUN -eq 1 ]] && S1_MAX_STEPS=25000 S2_MAX_STEPS=100000 EARLY_STOPPING=0 \
              KEEP_LAST_WEIGHTS=1 bash scripts/launch.sh --cohort ucr2p_10 \
              --tag zn_ot --arms local,centralized --extra "$ZN";;
    floor) echo "+ launch.sh --cohort ucr2p_10 --tag zn_floor --engine floor --heads ma_c --modes local"
          [[ $RUN -eq 1 ]] && bash scripts/launch.sh --cohort ucr2p_10 --tag zn_floor \
              --engine floor --heads ma_c --modes local;;
    diag) echo "# pooled tokenizer (pooled stage 1, federated stage 2) -- diagnostic, not legal"
          launch ucr2p_10 zn_170_ctfp centraltok_fedprior \
              "$ZN --fed-s2-val fixed --fedtokcp-stage1-root $REPO/artifacts/runs/zn_main/ckpt"
          echo "# pooled prior (federated stage 1 of (g), stage 2 on pooled tokens)"
          launch ucr2p_10 zn_fedtokcp fedtok_centralprior \
              "$ZN --fed-s2-val fixed --fedtokcp-stage1-root $REPO/artifacts/runs/zn_a2/ckpt";;
    c50-local)   launch c50 c50_local   local        "$ZN";;
    c50-central) launch c50 c50_central centralized  "$ZN";;
    c50-a2)      launch c50 c50_a2      federated_enc_fedavg "$ZN $A2X";;
    *) echo "unknown row '$1'"; usage; exit 2;;
  esac
}

RUN=0; WHAT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN=1; WHAT="${2:-dev}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown flag $1"; usage; exit 2;;
  esac
done

if [[ $RUN -eq 0 ]]; then
  echo "# every configuration of the paper (nothing is launched; --run <row> to launch)"
  echo "# see docs/CONFIGURATIONS.md for what each row means"
  for r in a,b,c,d,e f g ctrl h i j,k,l,m ot floor diag c50-local c50-central c50-a2; do
    echo; echo "## ($r)"; row "$r"
  done
  echo; usage
  exit 0
fi

case "$WHAT" in
  dev) for r in a,b,c,d,e f g ctrl h j,k,l,m ot floor; do echo "== ($r)"; row "$r" || exit $?; done;;
  c50) for r in c50-local c50-central c50-a2; do echo "== ($r)"; row "$r" || exit $?; done;;
  *)   row "$WHAT";;
esac
