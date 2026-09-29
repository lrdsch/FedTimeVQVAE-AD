#!/usr/bin/env bash
# smoke_arms.sh — prove every reporting arm still trains, saves and scores. Minutes, not hours.
#
#   bash scripts/smoke_arms.sh                 # 13 cases (8 paper arms + the encoder trio and
#                                              # its null) on one toy cluster
#   bash scripts/smoke_arms.sh --full          # + wsd_fed (real) and ucr_split_w2p (W per series)
#   ARMS=federated_fedavg_cb_only bash scripts/smoke_arms.sh    # one arm
#
# WHAT IT IS FOR. Every launch of the real sweep costs days. The failures worth catching
# before paying that are not subtle -- an arm that raises, an arm whose json never lands, an
# arm whose `--protocol converged` is not plumbed so `--fed-patience-rounds` is silently
# dead. All three have happened in this repo. This catches all three in minutes.
#
# TVQ_SMOKE=1 collapses the converged-loop budgets (config.apply_env_overrides) so the
# CONVERGED path -- select_on_val, patience, best-on-val restore -- is the one exercised,
# rather than being skipped for speed. The metrics produced are meaningless by construction
# and the run says so on every line of its own output.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PY:-python3}"
export TVQ_SMOKE=1

FULL=0; [[ "${1:-}" == "--full" ]] && FULL=1
OUT="${OUT:-$REPO/artifacts/_smoke}"; LOGDIR="$OUT/logs"
mkdir -p "$OUT" "$LOGDIR"

# The reporting table, read from the registry so this file cannot drift from it.
# `ARMS=<one arm>` must smoke EXACTLY that arm: the text arms are appended only when the
# caller did not choose. (Before 2026-07-30 they were appended unconditionally, so
# `ARMS=x bash smoke_arms.sh` silently ran x plus the text arms — and after fedprox joined
# them, `ARMS=federated_enc_fedprox` ran fedprox twice.)
_ARMS_FROM_ENV="${ARMS:+1}"
ARMS="${ARMS:-$("$PY" -c "
import ast
t=ast.parse(open('pipeline/federated_eval.py').read())
g={n.targets[0].id:n.value for n in t.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
print(' '.join(e.value for e in g['PAPER_ARMS'].elts))")}"
# The encoder-federation arms, reported in TEXT rather than as table rows (LAUNCH_RUNBOOK
# §5.2b). fedavg and fedprox are what step 4 of §5.0 launches verbatim, so both must smoke;
# fedprox was added 2026-07-30 when it was promoted out of the cut list (the reason it had
# been cut, "solver no-op", was refuted). fedproto and its null control commoninit joined the
# same day: they complete the trio of §5.2, they are one `--arms` edit away from being
# launched at full budget, and until then NOTHING here had ever executed their code. fedproto
# is the branch with the most code nothing else in this repo touches — per-code prototypes,
# the live assignment mask, the stage-0-only restriction on a Residual-VQ, and the round-0
# seeding of the prototypes from the broadcast codebook — so a crash in any of it surfaced
# only after hours of GPU. commoninit is nearly free and shares the same dispatch branch (it
# IS fedproto with lambda=0 and no averaging), but it takes the OTHER side of every
# `enc_proto_on` gate, which makes it the cheapest cover for that half of the branch.
#
# A smoke case is an arm name optionally followed by `@<variant>`: the variant never reaches
# `--arms`, it only gives a second CONFIGURATION of the same arm its own json, its own log and
# its own arm_extra entry. `ARMS=<name>` from the environment never carries one, so the
# override above is untouched by the convention.
[[ -z "$_ARMS_FROM_ENV" ]] && ARMS="$ARMS federated_enc_fedavg federated_enc_fedprox \
    federated_enc_fedproto federated_enc_fedproto@count federated_enc_commoninit"

# Per-case extra flags. The smoke must exercise the flags the REAL launch uses, or it green-
# lights a configuration nobody runs. fedprox is the live case: the CLI default is mu=0.01,
# but every measurement and every claim is at the paper value mu=0.1 (form 'loss'), and
# federated.py itself warns below 0.1. Keyed by SMOKE-CASE id, i.e. arm[@variant].
#
# fedproto passes --fedproto-agg EXPLICITLY even though 'uniform' is already the parser
# default, because the two readings run different aggregation code and only one of them is
# the method:
#   uniform  one vote per client. The canonical reading, the parser default, and what the
#            authors' released `proto_aggregation` actually computes (Eq. 6 of Tan et al.
#            2022 is WRITTEN count-weighted; their code is not).
#   count    count-weighted. Degenerate BY CONSTRUCTION and federated.py says so at runtime:
#            the prototype target is the Prop.1 merged codebook to a relative 1e-8, so the arm
#            carries no cross-client information beyond federated_cb_only + common init.
# Both are smoked because at TVQ_SMOKE budgets the second run costs seconds, and 'count' is a
# branch a real sweep would otherwise reach for the first time hours in.
#
# NOT passed here, deliberately: `--fed-enc-cb local`. With enc_fed_algo=fedproto and
# lambda>0, federated.py REFUSES merge != 'suffstat' (federated.py:1426 — with per-client
# dictionaries "code k" denotes a different region on every client, so averaging per-code
# prototypes averages unrelated things). That combination would fail the smoke by
# construction, and it is not what step 4 launches: the default --fed-enc-cb suffstat is.
arm_extra() {
  case "$1" in
    federated_enc_fedprox)        echo "--fedprox-mu 0.1";;
    federated_enc_fedproto)       echo "--fedproto-agg uniform";;
    federated_enc_fedproto@count) echo "--fedproto-agg count";;
    *)                            echo "";;
  esac
}

# (dataset, cluster, window, tolerance, batch). Toy first: 6 clients x 897 windows, the
# cheapest cluster that still has more than two clients. --full adds the real dataset and
# the per-series-window build, which is where the window plumbing can break.
CASES=("toy_fed_uni M1_rotary 128 64 64")
if [[ $FULL -eq 1 ]]; then
  CASES+=("ucr_split_w2p ucr_113 46 64 64")   # W=46 -> only 15 encoder tensors are shared
fi

stamp() { date +'%F %T'; }
PASS=0; FAIL=0; RESULTS=()

for case in "${CASES[@]}"; do
  read -r ds cl win tol batch <<<"$case"
  for arm_id in $ARMS; do
    # `arm` is what the CLI is allowed to see; `arm_id` is what names the artifacts and picks
    # the extras, so the two --fedproto-agg readings do not overwrite each other's json.
    arm="${arm_id%%@*}"
    name="${ds}__${cl}__${arm_id/@/_}"
    json="$OUT/$name.json"
    rm -f "$json"
    echo "[$(stamp)] smoke $name"
    CUDA_VISIBLE_DEVICES="${GPU:-0}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    timeout "${SMOKE_TIMEOUT:-1800}" "$PY" -u pipeline/federated_eval.py \
        --dataset "$ds" --cluster "$cl" --arms "$arm" \
        --protocol converged --s1-rounds 2 --s2-rounds 2 --local-epochs 1 \
        --fed-patience-rounds 2 --batch "$batch" --seeds 0 \
        --window-length "$win" --metrics-tolerance "$tol" \
        --out-dir "$OUT/ckpt" --out-json "$json" $(arm_extra "$arm_id") > "$LOGDIR/$name.log" 2>&1
    rc=$?

    # THREE checks, because each one has been a real failure in this repo:
    #  1. the process exited 0                     (an arm that raises)
    #  2. the result json exists                   (an arm that trains but never saves)
    #  3. federated arms signed for finite `val=` (--protocol not plumbed === select_on_val
    #     off => --fed-patience-rounds silently dead => 300 rounds with no stop)
    why=""
    [[ $rc -ne 0 ]] && why="exit=$rc"
    [[ -z "$why" && ! -f "$json" ]] && why="no json"
    if [[ -z "$why" && "$arm" == federated* ]]; then
      grep -qE '^\[fed:[a-z]+\] round [0-9]+:.* val=' "$LOGDIR/$name.log" \
        || why="stage-1 logged no val= (select_on_val OFF -> patience is dead)"
      grep -q 'Convergence stop DISABLED' "$LOGDIR/$name.log" \
        && why="patience explicitly DISABLED in the log"
    fi
    if [[ -z "$why" ]]; then
      PASS=$((PASS+1)); RESULTS+=("PASS  $name")
    else
      FAIL=$((FAIL+1)); RESULTS+=("FAIL  $name   [$why]  -> $LOGDIR/$name.log")
    fi
  done
done

echo
echo "════════ SMOKE ════════"
printf '%s\n' "${RESULTS[@]}"
echo "═══════════════════════"
echo "$PASS passed, $FAIL failed   (logs: $LOGDIR)"
echo "REMINDER: TVQ_SMOKE=1 was set. These metrics are meaningless -- code path only."
exit $(( FAIL > 0 ))
