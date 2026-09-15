#!/bin/bash
# =============================================================================
# ONE-SHOT server run: everything for exp10/exp11 on rhea, GPU 0, fire and forget.
#
#   db                 factory-K per-image feature database (~5 min; skipped with
#                      DBRUN=<tag> of an existing complete one, e.g. factoryK_0911_1241)
#   exp10 (factoryK)   mocap + factory-K hand-eye as HARD prior     (~2 min COLMAP + 3DGS)
#   exp11 (factoryK)   mocap centres as SOFT prior, pose_prior_mapper (~15 min + 3DGS)
#   exp10 (sfmrefit)   same hard route with the SfM-refitted hand-eye prior
#   exp11 (sfmrefit)   soft prior with the SfM-refitted hand-eye prior (EXTRA=0 skips it)
#
# Every launch gets its own tag (default <MMDD_HHMM>), so re-launching never
# collides with or overwrites earlier results. A failing experiment is logged
# and the chain continues. Per-stage logs + a final summary land in $ROOT/logs/.
# Defaults: 100k iterations full res (~5-6 h per experiment on the RTX 4080,
# i.e. ~1 day for all four), TRAIN_SCALE=100 (cm).
#
# Before launching, from the Mac:
#   git push the tools (Masked-3DGS main) and rsync gs_upload/{intrinsics_factory.csv,
#   sparse_mocap_factoryK,sparse_mocap_sfmrefit} to $ROOT/data/   (see chat)
#
# Launch on rhea inside screen (everything is shown live AND logged):
#   screen -S exp10_11
#   cd /home/robertsk/3dgs-masked && git pull origin main && DBRUN=factoryK_0911_1241 \
#     bash /home/robertsk/3dgs-masked/tools/run_all_exp10_11_on_rhea.sh 2>&1 | tee \
#     /media/white/nanodrones/roberts.kalvitis/3dgs/popillia/362img_01_09_masked/logs/run_all_$(date +%m%d_%H%M).log
#   (detach: Ctrl-A D, re-attach: screen -r exp10_11)
# =============================================================================
set -u
ROOT=${ROOT:-/media/white/nanodrones/roberts.kalvitis/3dgs/popillia/362img_01_09_masked}
CODE=${CODE:-/home/robertsk/3dgs-masked}
export GPU=${GPU:-0} RES=${RES:-1} ITERS=${ITERS:-100000} TRAIN_SCALE=${TRAIN_SCALE:-100} PRIOR_STD=${PRIOR_STD:-0.015}
export ROOT CODE
TAG=${TAG:-$(date +%m%d_%H%M)}
EXTRA=${EXTRA:-1}      # 1 = also exp11 with the SfM-refit prior (all four experiments)
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
    # full stage output goes to its log file AND to stdout (visible live in screen/tmux)
    env "$@" bash "$SCRIPT" "$stage" 2>&1 | tee "$logf"
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then
        note "$label: OK  ($(( ($(date +%s) - t) / 60 )) min)  log $logf"
    else
        note "$label: FAILED ($(( ($(date +%s) - t) / 60 )) min) — see $logf (last lines below)"
        tail -5 "$logf" | sed 's/^/    /' | tee -a "$SUMMARY"
        return 1
    fi
    # keep the informative lines
    grep -E "near-cull patch|prior:|database:|model:|focal length after|result vs|PRIOR vs|RESULT vs|Principal point|Evaluating test|attempt|FAILED|ABORT" "$logf" | sed 's/^/    /' | tee -a "$SUMMARY"
}

RUN_A=factoryK_$TAG
RUN_B=sfmrefit_$TAG
# DBRUN=<tag>: reuse an existing, complete factory-K database instead of building one
DB_OK=0
if [ -n "${DBRUN:-}" ]; then
    DBF=$ROOT/data/colmap_work_factoryK_$DBRUN/database.db
    if [ -f "$DBF" ] && singularity exec --cleanenv --contain --bind "$ROOT:$ROOT" "${SIF_PY:-$HOME/containers/3dgs.sif}" python - "$DBF" <<'PYEOF'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
n = c.execute('SELECT COUNT(*) FROM images').fetchone()[0]
kp = c.execute('SELECT COUNT(*) FROM keypoints WHERE rows > 0').fetchone()[0]
pairs = c.execute('SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0').fetchone()[0]
print(f'existing database: {n} images, {kp} with keypoints, {pairs} verified pairs')
sys.exit(0 if n >= 362 and kp >= 362 and pairs >= 3000 else 1)
PYEOF
    then note "reusing complete database colmap_work_factoryK_$DBRUN"; DB_OK=1; RUN_DB=$DBRUN
    else note "DBRUN=$DBRUN given but that database is missing/incomplete — building a new one"
    fi
fi
if [ $DB_OK = 0 ]; then
    RUN_DB=$RUN_A
    run_stage db db RUN=$RUN_A PRIOR=sparse_mocap_factoryK && DB_OK=1
fi
if [ $DB_OK = 1 ]; then
    run_stage exp10_factoryK exp10 RUN=$RUN_A DBRUN=$RUN_DB PRIOR=sparse_mocap_factoryK || true
    run_stage exp11_factoryK exp11 RUN=$RUN_A DBRUN=$RUN_DB PRIOR=sparse_mocap_factoryK || true
    run_stage exp10_sfmrefit exp10 RUN=$RUN_B DBRUN=$RUN_DB PRIOR=sparse_mocap_sfmrefit || true
    [ "$EXTRA" = 1 ] && { run_stage exp11_sfmrefit exp11 RUN=$RUN_B DBRUN=$RUN_DB PRIOR=sparse_mocap_sfmrefit || true; }
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
note "outputs: data/colmap_work_factoryK_$RUN_DB (database), data/colmap_work_exp1{0,1}_$RUN_A, data/colmap_work_exp10_$RUN_B,"
note "         data/exp10_guided_$RUN_A, data/exp11_pose_prior_$RUN_A, data/exp10_guided_$RUN_B, output/<same names>_r1"
note "=== finished $(date '+%F %T'), total $(( ($(date +%s) - T0) / 60 )) min; summary: $SUMMARY ==="
