#!/bin/bash
# =============================================================================
# ONE-SHOT server run: everything for exp10/exp11 on rhea, GPU 0, fire and forget.
#
#   db                 factory-K per-image feature database        (~25 min)
#   exp10 (factoryK)   mocap + factory-K hand-eye as HARD prior     (~10 min COLMAP + ~40 min 3DGS)
#   exp11 (factoryK)   mocap centres as SOFT prior, pose_prior_mapper (~15 min + ~40 min)
#   exp10 (sfmrefit)   same hard route with the SfM-refitted hand-eye prior (~10 + ~40 min)
#   [exp11 (sfmrefit)] only with EXTRA=1
#
# Every launch gets its own tag (default factoryK_<MMDD_HHMM>), so re-launching
# never collides with or overwrites earlier results. A failing experiment is
# logged and the chain continues. Per-stage logs + a final summary land in
# $ROOT/logs/. All defaults: 30k iterations, full res, TRAIN_SCALE=100 (cm).
#
# Before launching, from the Mac:
#   git push the tools (Masked-3DGS main) and rsync gs_upload/{intrinsics_factory.csv,
#   sparse_mocap_factoryK,sparse_mocap_sfmrefit} to $ROOT/data/   (see chat)
#
# Launch on rhea (git pull first, then this script, then log out):
#   cd /home/robertsk/3dgs-masked && git pull origin main && \
#   ROOT=/media/white/nanodrones/roberts.kalvitis/3dgs/popillia/362img_01_09_masked && mkdir -p $ROOT/logs && \
#   nohup bash tools/run_all_exp10_11_on_rhea.sh > $ROOT/logs/run_all_$(date +%m%d_%H%M).log 2>&1 < /dev/null &
# =============================================================================
set -u
ROOT=${ROOT:-/media/white/nanodrones/roberts.kalvitis/3dgs/popillia/362img_01_09_masked}
CODE=${CODE:-/home/robertsk/3dgs-masked}
export GPU=${GPU:-0} RES=${RES:-1} ITERS=${ITERS:-30000} TRAIN_SCALE=${TRAIN_SCALE:-100} PRIOR_STD=${PRIOR_STD:-0.015}
export ROOT CODE
TAG=${TAG:-$(date +%m%d_%H%M)}
EXTRA=${EXTRA:-0}
SCRIPT=$CODE/tools/run_exp10_11_factoryK.sh
LOGS=$ROOT/logs; mkdir -p "$LOGS"
SUMMARY=$LOGS/summary_exp10_11_$TAG.txt
T0=$(date +%s)
log() { echo "[$(date '+%F %T')] $*"; }
note() { echo "$*" | tee -a "$SUMMARY"; }

note "=== exp10/exp11 one-shot run, tag $TAG, started $(date '+%F %T') on $(hostname) ==="
note "GPU $GPU, res $RES, iters $ITERS, train_scale $TRAIN_SCALE, prior std $PRIOR_STD m"
[ -f "$SCRIPT" ] || { note "ABORT: $SCRIPT missing (git pull the tools first)"; exit 1; }
for f in "$ROOT/data/intrinsics_factory.csv" "$ROOT/data/sparse_mocap_factoryK/0/images.txt" "$ROOT/data/sparse_mocap_sfmrefit/0/images.txt"; do
    [ -f "$f" ] || { note "ABORT: $f missing (rsync gs_upload/ to $ROOT/data/ first)"; exit 1; }
done
nvidia-smi -i "$GPU" --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /' | tee -a "$SUMMARY"

run_stage() {   # run_stage <label> <stage> [ENV=... ...]
    local label=$1 stage=$2; shift 2
    local logf=$LOGS/${label}_$TAG.log
    log "=== $label: stage $stage ($*) -> $logf ==="
    local t=$(date +%s)
    if env "$@" bash "$SCRIPT" "$stage" > "$logf" 2>&1; then
        note "$label: OK  ($(( ($(date +%s) - t) / 60 )) min)  log $logf"
    else
        note "$label: FAILED ($(( ($(date +%s) - t) / 60 )) min) — see $logf (last lines below)"
        tail -5 "$logf" | sed 's/^/    /' | tee -a "$SUMMARY"
        return 1
    fi
    # keep the informative lines
    grep -E "near-cull patch|prior:|database:|model:|focal length after|result vs|PRIOR vs|RESULT vs|Principal point|FAILED|ABORT" "$logf" | sed 's/^/    /' | tee -a "$SUMMARY"
}

RUN_A=factoryK_$TAG
RUN_B=sfmrefit_$TAG
if run_stage db db RUN=$RUN_A PRIOR=sparse_mocap_factoryK; then
    run_stage exp10_factoryK exp10 RUN=$RUN_A PRIOR=sparse_mocap_factoryK || true
    run_stage exp11_factoryK exp11 RUN=$RUN_A PRIOR=sparse_mocap_factoryK || true
    run_stage exp10_sfmrefit exp10 RUN=$RUN_B DBRUN=$RUN_A PRIOR=sparse_mocap_sfmrefit || true
    [ "$EXTRA" = 1 ] && { run_stage exp11_sfmrefit exp11 RUN=$RUN_B DBRUN=$RUN_A PRIOR=sparse_mocap_sfmrefit || true; }
else
    note "database stage failed — experiments skipped"
fi

note ""
note "=== results.json ==="
for n in exp4_colmap_perimage exp6_colmap_guided_r1 exp8_colmap_guided_freefocal_r1 \
         exp10_guided_${RUN_A}_r1 exp11_pose_prior_${RUN_A}_r1 exp10_guided_${RUN_B}_r1 exp11_pose_prior_${RUN_B}_r1; do
    if [ -f "$ROOT/output/$n/results.json" ]; then
        note "--- $n: $(tr -d '\n ' < "$ROOT/output/$n/results.json")"
    fi
done
note ""
note "outputs: data/colmap_work_factoryK_$RUN_A (database), data/colmap_work_exp1{0,1}_$RUN_A, data/colmap_work_exp10_$RUN_B,"
note "         data/exp10_guided_$RUN_A, data/exp11_pose_prior_$RUN_A, data/exp10_guided_$RUN_B, output/<same names>_r1"
note "=== finished $(date '+%F %T'), total $(( ($(date +%s) - T0) / 60 )) min; summary: $SUMMARY ==="
