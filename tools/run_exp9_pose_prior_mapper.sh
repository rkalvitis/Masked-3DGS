#!/bin/bash
# =============================================================================
# ring362 — exp9_colmap_pose_prior
#
# Mocap poses as SOFT position priors inside COLMAP's own incremental SfM
# (`colmap pose_prior_mapper`, COLMAP >= 3.11), instead of as a hard
# initialisation for point_triangulator + bundle_adjuster (exp6 / exp8).
#
# Why: audited against the free per-image SfM (perimage_362), the mocap poses
# are ~10 mm / 12 deg off (p50) = 400-1100 px of reprojection error at this
# working distance. Triangulating from them builds wrong 2-view tracks and the
# global BA then settles in a wrong minimum (exp6: cameras ended 17 mm from the
# free SfM, worse than the 10 mm prior; exp8: per-image focal lengths diverged
# to 4000 px). No triangulator/BA option changes that — they are local refiners.
#
# Here the geometry comes from the image matches exactly as in the free run;
# the mocap only pulls every camera centre towards its measured position with a
# Gaussian prior of PRIOR_STD (default 0.015 m ~ the measured mocap error, robust
# loss on). Result: the free SfM's geometry, but METRIC and in the INSECT frame,
# with no post-hoc sim3 alignment.
#
# REPRODUCIBILITY CONTRACT: this script never deletes or overwrites anything.
# Every run is tagged (RUN, default <prior std>mm_<prior model name>) and writes
# only into folders named after the tag; if a stage's output already exists the
# stage aborts and tells you to pick another RUN. Inputs are read-only: the
# feature database is COPIED before priors are written into the copy.
#
# Inputs (read-only):
#   $DB_SRC         feature database with one PINHOLE camera per image:
#                   data/colmap_work2/database.db (free perimage run), else
#                   data/colmap_work_guided_freefocal/database.db (exp8 prep),
#                   else data/colmap_work/database.db
#   $PRIOR_MODEL    COLMAP TXT model whose camera centres are the priors
#                   (default data/sparse/0 = raw mocap; the corrected variant has
#                   the same centres; point it at any other pose set to redo the
#                   experiment with modified data)
#   data/colmap_work2/sfm/<largest>          free perimage model, for the audit (optional)
#   data/exp6_colmap_guided/{images,masks}   linked into the experiment folder
#
# Outputs (new, per RUN):
#   data/colmap_work_pose_prior_<RUN>/       database copy + priors, sfm/, tri/ (TXT), run_params.txt
#   data/exp9_colmap_pose_prior_<RUN>/       experiment folder (sparse/0 + images/masks links)
#   output/exp9_colmap_pose_prior_<RUN>[_r1] trained model, renders, metrics
#
#   bash tools/run_exp9_pose_prior_mapper.sh [prep|colmap|train|eval|all]   (default all)
#   PRIOR_STD=0.015   prior std in metres (x=y=z)
#   RUN=<tag>         output tag (default derived from PRIOR_STD and PRIOR_MODEL)
#   RES=1 (default) -> *_r1 (full res) ; RES=2 -> half res
#
# Launch on rhea:
#   ROOT=/media/white/nanodrones/roberts.kalvitis/3dgs/popillia/362img_01_09_masked
#   nohup bash /home/robertsk/3dgs-masked/tools/run_exp9_pose_prior_mapper.sh > $ROOT/run_exp9_$(date +%Y%m%d_%H%M).log 2>&1 &
# =============================================================================
set -u
STAGE=${1:-all}
ROOT=${ROOT:-/media/white/nanodrones/roberts.kalvitis/3dgs/popillia/362img_01_09_masked}
DATA=$ROOT/data; OUT=$ROOT/output
GPU=${GPU:-0}
RES=${RES:-1}
PRIOR_STD=${PRIOR_STD:-0.015}
PRIOR_MODEL=${PRIOR_MODEL:-$DATA/sparse/0}
CODE=${CODE:-/home/robertsk/3dgs-masked}
TORCH_CACHE=${TORCH_CACHE:-/media/white/nanodrones/roberts.kalvitis/3dgs/torch_cache}
SIF_GS=${SIF_GS:-$HOME/containers/masked-3dgs.sif}
SIF_PY=${SIF_PY:-$HOME/containers/3dgs.sif}; [ -f "$SIF_PY" ] || SIF_PY=$SIF_GS
SIF_COL=${SIF_COL:-$HOME/containers/colmap-cuda.sif}
STD_MM=$(python3 -c "print(int(round($PRIOR_STD*1000)))" 2>/dev/null || echo "${PRIOR_STD}m")
RUN=${RUN:-${STD_MM}mm_$(basename "$(dirname "$PRIOR_MODEL")")}
EXP=exp9_colmap_pose_prior_$RUN
NAME=$EXP; [ "$RES" = 1 ] && NAME=${EXP}_r1
WORK=$DATA/colmap_work_pose_prior_$RUN
DB=$WORK/database.db
unset SINGULARITYENV_CUDA_VISIBLE_DEVICES
PY="singularity exec --cleanenv --contain --bind $ROOT:$ROOT --bind $CODE:/workspace $SIF_PY"
COL="singularity exec --nv --cleanenv --contain --bind $ROOT:$ROOT $SIF_COL"
GS="singularity exec --nv --cleanenv --contain --bind $CODE:/workspace --bind $DATA:/data \
    --bind $OUT:/outroot --bind $ROOT:$ROOT --bind $TORCH_CACHE:/torch_cache $SIF_GS"
log() { echo "[$(date '+%F %T')] $*"; }
want() { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }
fresh() { [ -e "$1" ] && { log "ABORT: $1 already exists — previous results are kept; choose another RUN=<tag>"; exit 1; }; return 0; }
findopt() { $COL colmap "$1" -h 2>&1 | grep -o -- "--[A-Za-z_.]*\.\?$2\b" | head -1; }

# ── 0. sanity ────────────────────────────────────────────────────────────────
[ -f "$PRIOR_MODEL/images.txt" ] || { log "ABORT: $PRIOR_MODEL/images.txt (prior model) missing"; exit 1; }
$COL colmap pose_prior_mapper -h >/dev/null 2>&1 || { log "ABORT: this COLMAP has no pose_prior_mapper (needs >= 3.11): $SIF_COL"; exit 1; }
DB_SRC=""
for cand in "$DATA/colmap_work2/database.db" "$DATA/colmap_work_guided_freefocal/database.db" "$DATA/colmap_work/database.db"; do
    [ -f "$cand" ] && { DB_SRC=$cand; break; }
done
[ -n "$DB_SRC" ] || { log "ABORT: no feature database found (run the free SfM or exp8 prep first)"; exit 1; }
if want train || want eval; then
    grep -q "Principal point honoured" "$CODE/scene/dataset_readers.py" || { log "ABORT: trainer lacks the principal-point fix"; exit 1; }
fi
log "RUN=$RUN  prior model $PRIOR_MODEL  std $PRIOR_STD m  database $DB_SRC  res $RES"

# ── 1. prep: copy the database, write the prior camera centres as pose priors ─
if want prep; then
    fresh "$WORK"; mkdir -p "$WORK"
    printf 'run=%s\nprior_model=%s\nprior_std_m=%s\ndb_src=%s\nres=%s\ndate=%s\n' \
        "$RUN" "$PRIOR_MODEL" "$PRIOR_STD" "$DB_SRC" "$RES" "$(date '+%F %T')" > "$WORK/run_params.txt"
    cp "$DB_SRC" "$DB"
    log "=== $EXP: database copy $DB_SRC -> $DB; writing priors (std $PRIOR_STD m) from $PRIOR_MODEL ==="
    $PY python - "$DB" "$PRIOR_MODEL" "$PRIOR_STD" <<'PYEOF'
import sqlite3, sys
import numpy as np
db, model, std = sys.argv[1], sys.argv[2], float(sys.argv[3])
con = sqlite3.connect(db)
cols = [r[1] for r in con.execute('PRAGMA table_info(pose_priors)')]
assert cols, 'database has no pose_priors table (COLMAP < 3.11 created it) — re-extract features with a newer COLMAP'
imgs = {n: (i, c) for i, n, c in con.execute('SELECT image_id, name, camera_id FROM images')}
cams = dict(con.execute('SELECT camera_id, model FROM cameras'))
use = {}
for _, c in imgs.values(): use[c] = use.get(c, 0) + 1
print(f'database: {len(imgs)} images, {len(cams)} cameras, per-image cameras: {all(n == 1 for n in use.values())}')
def q2R(w, x, y, z):
    n = (w*w + x*x + y*y + z*z) ** 0.5; w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
cov = (np.eye(3) * std * std).astype(np.float64)              # Eigen Matrix3d; symmetric, so layout irrelevant
nan3 = np.full(3, np.nan, dtype=np.float64)
new_schema = 'corr_data_id' in cols                             # COLMAP >= 4: prior keyed by (data_id, sensor)
print('pose_priors schema:', 'COLMAP>=4 (corr_data_id/corr_sensor_id/corr_sensor_type)' if new_schema else 'COLMAP 3.11-3.12 (image_id)')
assert con.execute('SELECT COUNT(*) FROM pose_priors').fetchone()[0] == 0, 'the source database already has pose priors — not touching them; pick a clean database'
n, missing = 0, []
for line in open(f'{model}/images.txt'):
    t = line.split()
    if len(t) != 10 or t[0].startswith('#') or not t[9].lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    if t[9] not in imgs:
        missing.append(t[9]); continue
    R = q2R(*map(float, t[1:5])); tv = np.array(list(map(float, t[5:8])))
    C = (-R.T @ tv).astype(np.float64)                          # camera centre in the insect frame [m]
    iid, cid = imgs[t[9]]
    if new_schema:
        # SensorType.CAMERA = 0, PosePrior::CoordinateSystem::CARTESIAN = 1 (colmap/util/types.h, geometry/pose_prior.h)
        con.execute('INSERT INTO pose_priors (corr_data_id, corr_sensor_id, corr_sensor_type, position, '
                    'position_covariance, gravity, coordinate_system) VALUES (?,?,?,?,?,?,?)',
                    (iid, cid, 0, C.tobytes(), cov.tobytes(), nan3.tobytes(), 1))
    else:
        con.execute('INSERT INTO pose_priors (image_id, position, coordinate_system, position_covariance) '
                    'VALUES (?,?,?,?)', (iid, C.tobytes(), 1, cov.tobytes()))
    n += 1
con.commit()
print(f'{n} pose priors written (CARTESIAN, std {std*1000:.0f} mm)' + (f'; not in database: {missing}' if missing else ''))
assert n >= 3, 'fewer than 3 priors written'
PYEOF
    [ $? -eq 0 ] || { log "ABORT: writing priors failed"; exit 1; }
fi

# ── 2. pose-prior mapper: free SfM from matches + soft mocap position priors ──
if want colmap; then
    [ -f "$DB" ] || { log "ABORT: $DB missing (run prep with the same RUN)"; exit 1; }
    fresh "$WORK/sfm"; fresh "$WORK/tri"; fresh "$DATA/$EXP"
    P_STDX=$(findopt pose_prior_mapper prior_position_std_x); P_STDY=$(findopt pose_prior_mapper prior_position_std_y)
    P_STDZ=$(findopt pose_prior_mapper prior_position_std_z); P_OVR=$(findopt pose_prior_mapper overwrite_priors_covariance)
    P_ROB=$(findopt pose_prior_mapper use_robust_loss_on_prior_position)
    M_FL=$(findopt pose_prior_mapper ba_refine_focal_length); M_PP=$(findopt pose_prior_mapper ba_refine_principal_point)
    M_EP=$(findopt pose_prior_mapper ba_refine_extra_params)
    mkdir -p "$WORK/sfm"
    log "=== $EXP: pose_prior_mapper (prior std $PRIOR_STD m, robust loss, focal FREE per image, pp fixed) ==="
    log "options: $P_STDX $P_STDY $P_STDZ $P_OVR $P_ROB $M_FL $M_PP $M_EP"
    $COL colmap pose_prior_mapper \
        --database_path "$DB" --image_path "$DATA/images" --output_path "$WORK/sfm" \
        ${P_STDX:+$P_STDX $PRIOR_STD} ${P_STDY:+$P_STDY $PRIOR_STD} ${P_STDZ:+$P_STDZ $PRIOR_STD} \
        ${P_OVR:+$P_OVR 1} ${P_ROB:+$P_ROB 1} \
        ${M_FL:+$M_FL 1} ${M_PP:+$M_PP 0} ${M_EP:+$M_EP 0} \
        || { log "$EXP pose_prior_mapper FAILED"; exit 1; }
    BEST=""; BEST_N=0
    for d in "$WORK"/sfm/*/; do
        [ -f "$d/images.bin" ] || [ -f "$d/images.txt" ] || continue
        c="$WORK/cand_$(basename "$d")"; mkdir -p "$c"
        $COL colmap model_converter --input_path "$d" --output_path "$c" --output_type TXT >/dev/null 2>&1
        n=$(grep -c '\.jpg' "$c/images.txt" 2>/dev/null || echo 0)
        log "model $d: $n images"
        [ "$n" -gt "$BEST_N" ] && { BEST=$c; BEST_N=$n; }
    done
    [ -n "$BEST" ] || { log "ABORT: pose_prior_mapper produced no model"; exit 1; }
    mkdir -p "$WORK/tri"; cp "$BEST"/{cameras,images,points3D}.txt "$WORK/tri/"
    log "$EXP model: $BEST_N images, $(grep -vc '^#' "$WORK/tri/points3D.txt") points  (expect 362 — fewer means the priors were too tight or the free run also lost them)"
    log "focal length after BA (fx px): $(awk '!/^#/{print $5}' "$WORK/tri/cameras.txt" | sort -n | awk '{a[NR]=$1} END{printf "min %.1f  median %.1f  max %.1f", a[1], a[int((NR+1)/2)], a[NR]}')"

    # audit: result vs the prior centres (no alignment: the result IS in the prior frame)
    # and vs the free perimage model (sim3) if it is on disk
    FREE=""; FREE_N=0
    for d in "$DATA"/colmap_work2/sfm/*/; do
        [ -f "$d/images.bin" ] || [ -f "$d/images.txt" ] || continue
        c="$WORK/free_$(basename "$d")"; mkdir -p "$c"
        $COL colmap model_converter --input_path "$d" --output_path "$c" --output_type TXT >/dev/null 2>&1
        n=$(grep -c '\.jpg' "$c/images.txt" 2>/dev/null || echo 0)
        [ "$n" -gt "$FREE_N" ] && { FREE=$c; FREE_N=$n; }
    done
    $PY python - "$WORK/tri" "$PRIOR_MODEL" "${FREE:-none}" <<'PYEOF'
import sys
import numpy as np
res_dir, prior_dir, free_dir = sys.argv[1:4]
def q2R(w, x, y, z):
    n = (w*w + x*x + y*y + z*z) ** 0.5; w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def read(path):
    out = {}
    for line in open(path):
        t = line.split()
        if len(t) >= 10 and not t[0].startswith('#') and t[9].lower().endswith(('.jpg', '.png', '.jpeg')):
            R = q2R(*map(float, t[1:5])); tv = np.array(list(map(float, t[5:8])))
            out[t[9]] = (-R.T @ tv, R.T)
    return out
def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0); xs, xd = src - ms, dst - md
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src)); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt; s = np.trace(np.diag(D) @ S) * len(src) / (xs ** 2).sum()
    return s, R, md - s * R @ ms
def pct(x): return f'p50 {np.percentile(x,50):.2f}  p90 {np.percentile(x,90):.2f}  max {x.max():.2f}'
res, prior = read(f'{res_dir}/images.txt'), read(f'{prior_dir}/images.txt')
common = sorted(set(res) & set(prior))
d = np.array([np.linalg.norm(res[n][0] - prior[n][0]) for n in common]) * 1000
print(f'result vs prior centres (NO alignment; = how far the images pulled the cameras off the prior): {len(common)} cams, {pct(d)} mm')
if free_dir != 'none':
    free = read(f'{free_dir}/images.txt')
    common = sorted(set(res) & set(free))
    A = np.array([free[n][0] for n in common]); B = np.array([res[n][0] for n in common])
    s, R, t = umeyama(A, B)
    d = np.linalg.norm((s * (R @ A.T).T + t) - B, axis=1) * 1000
    rot = np.array([np.degrees(np.arccos(np.clip((np.trace((R @ free[n][1]).T @ res[n][1]) - 1) / 2, -1, 1))) for n in common])
    print(f'result vs free perimage SfM (sim3, scale {s:.5f}): {len(common)} cams, pos {pct(d)} mm, rot {pct(rot)} deg  (expect a few mm / <1 deg)')
else:
    print('free perimage model not found under data/colmap_work2/sfm — skipped that comparison')
PYEOF

    mkdir -p "$DATA/$EXP/sparse/0"
    cp "$WORK/tri"/{cameras,images,points3D}.txt "$DATA/$EXP/sparse/0/"
    cp "$WORK/run_params.txt" "$DATA/$EXP/"
    for d in images masks; do
        src=$(readlink -f "$DATA/exp6_colmap_guided/$d"); [ -d "$src" ] || src=$DATA/$d
        rel=$($PY python -c "import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" "$src" "$DATA/$EXP")
        ln -s "$rel" "$DATA/$EXP/$d"
    done
    log "experiment folder ready: $DATA/$EXP (images -> $(readlink "$DATA/$EXP/images"))"
fi

# ── 3. train (as exp8: attempt 2 caps densification on OOM) ──────────────────
train_once() {   # train_once <model dir> [extra train args]
    local m=$1; shift
    mkdir -p "$m"
    $GS env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
        python /workspace/train.py -s "/data/$EXP" -m "/outroot/$(basename "$m")" \
        --masks masks --lambda_mask 0.1 --eval --disable_viewer \
        --data_device cpu --device "$GPU" -r "$RES" \
        --random_background --densify_until_iter 25000 \
        --iterations 100000 --save_iterations 30000 60000 90000 100000 "$@" \
        2>&1 | tee "$m/train.log"
    [ -d "$m/point_cloud/iteration_100000" ]
}
if want train; then
    [ -f "$DATA/$EXP/sparse/0/images.txt" ] || { log "ABORT: $DATA/$EXP not prepared (run colmap stage with the same RUN)"; exit 1; }
    fresh "$OUT/$NAME"
    log "=== TRAIN $NAME (res $RES) attempt 1 ==="
    if train_once "$OUT/$NAME"; then
        log "$NAME: attempt 1 succeeded"
    else
        # keep the failed attempt for the record, train again into the real name
        mv "$OUT/$NAME" "$OUT/${NAME}_attempt1_failed"
        log "$NAME: attempt 1 FAILED (OOM?) — kept as ${NAME}_attempt1_failed; attempt 2 with --densify_grad_threshold 0.0004"
        train_once "$OUT/$NAME" --densify_grad_threshold 0.0004 && log "$NAME: attempt 2 succeeded (NOTE: densify_grad_threshold 0.0004)" \
            || { log "$NAME: attempt 2 FAILED — giving up"; exit 1; }
    fi
fi

# ── 4. render test views + metrics ───────────────────────────────────────────
if want eval; then
    [ -d "$OUT/$NAME/point_cloud/iteration_100000" ] || { log "ABORT: $OUT/$NAME has no trained model"; exit 1; }
    fresh "$OUT/$NAME/results.json"
    log "=== RENDER $NAME (test views) ==="
    $GS python /workspace/render.py -m "/outroot/$NAME" --device "$GPU" --skip_train || log "render $NAME failed"
    log "=== METRICS $NAME ==="
    $GS env TORCH_HOME=/torch_cache PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
        python /workspace/metrics.py -m "/outroot/$NAME" --device "$GPU" 2>&1 | tee "$OUT/$NAME/metrics_v2.log"
    log "=== SUMMARY ==="
    for n in exp4_colmap_perimage exp6_colmap_guided_r1 exp8_colmap_guided_freefocal_r1 "$NAME"; do
        [ -f "$OUT/$n/results.json" ] && { echo "--- $n"; cat "$OUT/$n/results.json"; echo; } || echo "--- $n: no results.json"
    done
fi
log "all done."
