#!/bin/bash
# =============================================================================
# ring362 — exp8_colmap_guided_freefocal
#
# Same as exp6_colmap_guided (corrected mocap poses as prior -> point_triangulator
# -> bundle_adjuster, 200 px then 20 px rounds) but with ONE PINHOLE CAMERA PER
# IMAGE and the focal length (fx, fy) FREE in bundle adjustment, as in
# exp4_colmap_perimage. Principal point stays fixed at the rig calibration.
# This isolates the effect of the mocap prior from the intrinsics model.
#
# Reuses from disk:
#   data/sparse_mocap_corrected/0   corrected mocap poses + rig camera (from run_mocap_exps.sh)
#   data/colmap_work/database.db    masked SIFT features + matches, IF it has one camera per image
#   data/exp6_colmap_guided/{images,masks}   linked into the new experiment folder
#
# COLMAP >= 3.12 stores rigs/frames in the database and refuses a model whose
# cameras do not match them (Check failed: existing_frame.RigId() == frame.RigId()).
# The free-run database has ONE shared camera, so per-image cameras cannot be
# triangulated against it. In that case the prep stage builds a NEW database
# in colmap_work_guided_freefocal/ with --ImageReader.single_camera 0: masked
# SIFT (GPU) + exhaustive guided matching, same settings as the free run
# (max_image_size 3200, peak 0.004, 16384 features, ratio 0.85). ~20-30 min.
# FRESH=1 forces re-extraction even if that database already exists.
#
#   bash tools/run_exp8_guided_freefocal.sh [prep|colmap|train|eval|all]   (default all)
#
#   RES=1 (default) -> output exp8_colmap_guided_freefocal_r1 (full res, like exp6_colmap_guided_r1)
#   RES=2           -> output exp8_colmap_guided_freefocal     (half res, like exp6_colmap_guided)
#
# Launch on rhea:
#   ROOT=/media/white/nanodrones/roberts.kalvitis/3dgs/popillia/362img_01_09_masked
#   nohup bash /home/robertsk/3dgs-masked/tools/run_exp8_guided_freefocal.sh > $ROOT/run_exp8.log 2>&1 &
#   tail -f $ROOT/run_exp8.log
# =============================================================================
set -u
STAGE=${1:-all}
ROOT=${ROOT:-/media/white/nanodrones/roberts.kalvitis/3dgs/popillia/362img_01_09_masked}
DATA=$ROOT/data; OUT=$ROOT/output
GPU=${GPU:-0}
RES=${RES:-1}
CODE=${CODE:-/home/robertsk/3dgs-masked}
TORCH_CACHE=${TORCH_CACHE:-/media/white/nanodrones/roberts.kalvitis/3dgs/torch_cache}
SIF_GS=${SIF_GS:-$HOME/containers/masked-3dgs.sif}
SIF_PY=${SIF_PY:-$HOME/containers/3dgs.sif}; [ -f "$SIF_PY" ] || SIF_PY=$SIF_GS
SIF_COL=${SIF_COL:-$HOME/containers/colmap-cuda.sif}
EXP=exp8_colmap_guided_freefocal
NAME=$EXP; [ "$RES" = 1 ] && NAME=${EXP}_r1
WORK=$DATA/colmap_work_guided_freefocal
DB_FREE=$DATA/colmap_work/database.db
DB=$WORK/database.db          # copy of DB_FREE, or a fresh per-image-camera database
FRESH=${FRESH:-0}
unset SINGULARITYENV_CUDA_VISIBLE_DEVICES
PY="singularity exec --cleanenv --contain --bind $ROOT:$ROOT --bind $CODE:/workspace $SIF_PY"
COL="singularity exec --nv --cleanenv --contain --bind $ROOT:$ROOT $SIF_COL"
GS="singularity exec --nv --cleanenv --contain --bind $CODE:/workspace --bind $DATA:/data \
    --bind $OUT:/outroot --bind $ROOT:$ROOT --bind $TORCH_CACHE:/torch_cache $SIF_GS"
log() { echo "[$(date '+%F %T')] $*"; }
want() { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }

# ── 0. sanity ────────────────────────────────────────────────────────────────
[ -f "$DATA/sparse_mocap_corrected/0/images.txt" ] || { log "ABORT: $DATA/sparse_mocap_corrected/0 missing (run run_mocap_exps.sh step 1 first)"; exit 1; }
[ -f "$DB_FREE" ] || { log "ABORT: $DB_FREE missing"; exit 1; }
grep -q "Principal point honoured" "$CODE/scene/dataset_readers.py" || { log "ABORT: trainer lacks the principal-point fix"; exit 1; }
grep -q masked_ssim "$CODE/metrics.py" || { log "ABORT: metrics.py is not the mask-aware version (sync the code)"; exit 1; }

# ── 1. prior: corrected mocap poses, one PINHOLE camera per image ────────────
db_is_per_image() {   # db_is_per_image <db>: exit 0 if every image has its own PINHOLE camera
    $PY python - "$1" <<'PYEOF'
import sqlite3, sys
con = sqlite3.connect(sys.argv[1])
imgs = con.execute('SELECT image_id, camera_id FROM images').fetchall()
cams = dict(con.execute('SELECT camera_id, model FROM cameras').fetchall())
use = {}
for _, c in imgs: use[c] = use.get(c, 0) + 1
ok = len(imgs) > 0 and all(n == 1 for n in use.values()) and all(cams[c] == 1 for c in use)
print(f'{sys.argv[1]}: {len(imgs)} images, {len(cams)} cameras -> per-image PINHOLE: {ok}')
sys.exit(0 if ok else 1)
PYEOF
}

if want prep; then
    mkdir -p "$WORK"; rm -rf "$WORK/prior" "$WORK"/tri*
    [ "$FRESH" = 1 ] && rm -f "$DB"
    if [ ! -f "$DB" ]; then
        if db_is_per_image "$DB_FREE"; then
            log "free-run database already has per-image cameras; copying it"
            cp "$DB_FREE" "$DB"
        else
            log "=== $EXP: free-run database has a shared camera -> new database with one camera per image ==="
            # feature masks: reuse the free run's dilated masks if present, else make them
            CM=$DATA/colmap_work/colmap_masks
            if [ ! -d "$CM" ] || [ "$(ls "$CM" | wc -l)" -lt 362 ]; then
                CM=$WORK/colmap_masks; mkdir -p "$CM"
                MSRC=$(readlink -f "$DATA/exp6_colmap_guided/masks"); [ -d "$MSRC" ] || MSRC=$DATA/masks
                log "making dilated COLMAP feature masks from $MSRC"
                $PY python - "$DATA/images" "$MSRC" "$CM" <<'PYEOF'
import os, sys, cv2, numpy as np
imgs, msrc, out = sys.argv[1:4]
k = np.ones((20, 20), np.uint8); n = 0
for name in sorted(os.listdir(imgs)):
    if not name.lower().endswith('.jpg'): continue
    m = cv2.imread(f'{msrc}/{os.path.splitext(name)[0]}.png', cv2.IMREAD_GRAYSCALE)
    if m is None: m = cv2.imread(f'{msrc}/{name}', cv2.IMREAD_GRAYSCALE)
    assert m is not None, f'no mask for {name} in {msrc}'
    cv2.imwrite(f'{out}/{name}.png', cv2.dilate((m > 127).astype(np.uint8) * 255, k)); n += 1
print(f'{n} COLMAP masks written to {out}')
PYEOF
            fi
            CAM_PARAMS=$(awk '!/^#/{print $5","$6","$7","$8; exit}' "$DATA/sparse_mocap_corrected/0/cameras.txt")
            findopt() { $COL colmap "$1" -h 2>&1 | grep -o -- "--[A-Za-z_]*\.$2" | head -1; }
            EX_MAXSZ=$(findopt feature_extractor max_image_size); EX_PEAK=$(findopt feature_extractor peak_threshold)
            EX_NFEAT=$(findopt feature_extractor max_num_features); EX_GPU=$(findopt feature_extractor use_gpu)
            EX_IDX=$(findopt feature_extractor gpu_index); IR_MASK=$(findopt feature_extractor mask_path)
            M_GPU=$(findopt exhaustive_matcher use_gpu); M_IDX=$(findopt exhaustive_matcher gpu_index)
            M_GUIDED=$(findopt exhaustive_matcher guided_matching); M_RATIO=$(findopt exhaustive_matcher max_ratio)
            log "feature_extractor: PINHOLE per image, params $CAM_PARAMS, masks $CM"
            rm -f "$DB"
            $COL colmap feature_extractor \
                --database_path "$DB" --image_path "$DATA/images" \
                ${IR_MASK:---ImageReader.mask_path} "$CM" \
                --ImageReader.camera_model PINHOLE --ImageReader.single_camera 0 \
                --ImageReader.camera_params "$CAM_PARAMS" \
                ${EX_MAXSZ:+$EX_MAXSZ 3200} ${EX_PEAK:+$EX_PEAK 0.004} ${EX_NFEAT:+$EX_NFEAT 16384} \
                ${EX_GPU:+$EX_GPU 1} ${EX_IDX:+$EX_IDX "$GPU"} \
                || { log "feature extraction FAILED"; exit 1; }
            log "exhaustive_matcher (guided, ratio 0.85)"
            $COL colmap exhaustive_matcher \
                --database_path "$DB" \
                ${M_GUIDED:+$M_GUIDED 1} ${M_RATIO:+$M_RATIO 0.85} \
                ${M_GPU:+$M_GPU 1} ${M_IDX:+$M_IDX "$GPU"} \
                || { log "matching FAILED"; exit 1; }
            db_is_per_image "$DB" || { log "ABORT: new database still not per-image"; exit 1; }
            $PY python - "$DB_FREE" "$DB" <<'PYEOF'
import sqlite3, sys
for tag, p in zip(('free-run db', 'new db'), sys.argv[1:3]):
    c = sqlite3.connect(p)
    kp = c.execute('SELECT SUM(rows) FROM keypoints').fetchone()[0]
    pairs = c.execute('SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0').fetchone()[0]
    print(f'{tag}: {kp} keypoints total, {pairs} verified image pairs')
PYEOF
        fi
    fi
    mkdir -p "$WORK/prior"
    log "=== $EXP: per-image-camera prior from corrected mocap poses ==="
    $PY python - "$DATA" "$WORK" "$DB" <<'PYEOF'
import os, sqlite3, sys
ds, work, db = sys.argv[1:4]
con = sqlite3.connect(db)
imgs = con.execute('SELECT image_id, name, camera_id FROM images').fetchall()
dbcams = {c: (m, w, h) for c, m, w, h in con.execute('SELECT camera_id, model, width, height FROM cameras')}
rig = [l.split() for l in open(f'{ds}/sparse_mocap_corrected/0/cameras.txt') if not l.startswith('#')][0]
model, W, H, params = rig[1], int(rig[2]), int(rig[3]), rig[4:]
assert model == 'PINHOLE' and len(params) == 4, f'expected PINHOLE fx fy cx cy, got {rig}'
print(f'rig calibration: PINHOLE {W}x{H} fx={params[0]} fy={params[1]} cx={params[2]} cy={params[3]}')
use = {}
for _, _, c in imgs: use[c] = use.get(c, 0) + 1
per_image_db = all(n == 1 for n in use.values()) and all(dbcams[c] == (1, W, H) for c in use)
print(f'database: {len(imgs)} images, {len(dbcams)} cameras, per-image PINHOLE cameras: {per_image_db}')
assert per_image_db, 'database must have one PINHOLE camera per image (COLMAP >= 3.12 rig check)'
# COLMAP >= 3.12: a legacy text model gets one implicit rig per camera (rig_id ==
# camera_id) and one frame per image (frame_id == image_id); the database must agree.
try:
    fr = dict(con.execute('SELECT frame_id, rig_id FROM frames').fetchall())
    fd = con.execute('SELECT frame_id, sensor_id, data_id FROM frame_data').fetchall()
    cam_of = {i: c for i, _, c in imgs}
    bad = [(f, s, d) for f, s, d in fd if f != d or fr.get(f) != cam_of.get(d) or s != cam_of.get(d)]
    print(f'rig/frame consistency: {len(fd)} frame entries, {len(bad)} inconsistent')
    assert not bad, f'database rigs/frames do not match implicit per-camera rigs, e.g. {bad[:3]}'
except sqlite3.OperationalError:
    print('database has no rigs/frames tables (COLMAP < 3.12): nothing to check')
poses = {}
for line in open(f'{ds}/sparse_mocap_corrected/0/images.txt'):
    t = line.split()
    if len(t) != 10 or t[0].startswith('#'): continue
    poses[t[9]] = t[1:8]
name2img = {n: (i, c) for i, n, c in imgs}
os.makedirs(f'{work}/prior', exist_ok=True)
open(f'{work}/prior/points3D.txt', 'w').close()
n = 0
with open(f'{work}/prior/cameras.txt', 'w') as fc, open(f'{work}/prior/images.txt', 'w') as fi:
    fc.write('# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]  (one camera per image, rig calibration as init)\n')
    fi.write('# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME  (corrected mocap poses, db-consistent ids)\n')
    for name in sorted(poses):
        if name not in name2img: continue
        iid, cid = name2img[name]
        cam = cid
        fc.write(f'{cam} PINHOLE {W} {H} {" ".join(params)}\n')
        fi.write(f'{iid} {" ".join(poses[name])} {cam} {name}\n\n')
        n += 1
missing = sorted(set(poses) - set(name2img))
print(f'prior model: {n} images (expect 362)' + (f'; not in database: {missing}' if missing else ''))
PYEOF
    [ -s "$WORK/prior/images.txt" ] || { log "ABORT: prior not written"; exit 1; }
fi

# ── 2. triangulate + bundle adjust with focal length FREE ────────────────────
if want colmap; then
    findopt() { $COL colmap "$1" -h 2>&1 | grep -o -- "--[A-Za-z_]*\.$2" | head -1; }
    T_FILT=$(findopt point_triangulator filter_max_reproj_error)
    T_COMP=$(findopt point_triangulator tri_complete_max_reproj_error)
    T_MERG=$(findopt point_triangulator tri_merge_max_reproj_error)
    BA_FL=$(findopt bundle_adjuster refine_focal_length)
    BA_PP=$(findopt bundle_adjuster refine_principal_point)
    BA_EP=$(findopt bundle_adjuster refine_extra_params)
    IN=$WORK/prior
    for round in 200 20; do
        log "=== $EXP: triangulate (max reproj $round px) + bundle adjust (focal FREE, pp fixed) ==="
        rm -rf "$WORK/tri_r$round" "$WORK/tri_ba$round"; mkdir -p "$WORK/tri_r$round" "$WORK/tri_ba$round"
        $COL colmap point_triangulator \
            --database_path "$DB" --image_path "$DATA/images" \
            --input_path "$IN" --output_path "$WORK/tri_r$round" \
            ${T_FILT:+$T_FILT $round} ${T_COMP:+$T_COMP $round} ${T_MERG:+$T_MERG $round} \
            || { log "$EXP triangulation FAILED"; exit 1; }
        $COL colmap bundle_adjuster \
            --input_path "$WORK/tri_r$round" --output_path "$WORK/tri_ba$round" \
            ${BA_FL:---BundleAdjustment.refine_focal_length} 1 \
            ${BA_PP:---BundleAdjustment.refine_principal_point} 0 \
            ${BA_EP:---BundleAdjustment.refine_extra_params} 0 \
            || { log "$EXP bundle adjustment FAILED"; exit 1; }
        IN=$WORK/tri_ba$round
    done
    rm -rf "$WORK/tri"; mkdir -p "$WORK/tri"
    $COL colmap model_converter --input_path "$WORK/tri_ba20" --output_path "$WORK/tri" --output_type TXT
    log "$EXP model: $(grep -c '\.jpg' "$WORK/tri/images.txt") images, $(grep -vc '^#' "$WORK/tri/points3D.txt") points"
    RIGFX=$(awk '!/^#/{print $5; exit}' "$DATA/sparse_mocap_corrected/0/cameras.txt")
    log "focal length after BA (fx px): $(awk '!/^#/{print $5}' "$WORK/tri/cameras.txt" | sort -n | awk -v rig="$RIGFX" '{a[NR]=$1} END{printf "min %.1f  median %.1f  max %.1f  (rig init %s)", a[1], a[int((NR+1)/2)], a[NR], rig}')"

    # experiment folder: images/masks as used by exp6, sparse/0 = this model
    mkdir -p "$DATA/$EXP/sparse/0"
    cp "$WORK/tri"/{cameras,images,points3D}.txt "$DATA/$EXP/sparse/0/"
    for d in images masks; do
        src=$(readlink -f "$DATA/exp6_colmap_guided/$d")
        [ -d "$src" ] || src=$DATA/$d
        rel=$($PY python -c "import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" "$src" "$DATA/$EXP")
        ln -sfn "$rel" "$DATA/$EXP/$d"
    done
    log "experiment folder ready: $DATA/$EXP (images -> $(readlink "$DATA/$EXP/images"))"
fi

# ── 3. train (attempt 2 caps densification on OOM, as in run_full_res.sh) ────
train_once() {
    mkdir -p "$OUT/$NAME"
    $GS env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
        python /workspace/train.py -s "/data/$EXP" -m "/outroot/$NAME" \
        --masks masks --lambda_mask 0.1 --eval --disable_viewer \
        --data_device cpu --device "$GPU" -r "$RES" \
        --random_background --densify_until_iter 25000 \
        --iterations 100000 --save_iterations 30000 60000 90000 100000 "$@" \
        2>&1 | tee "$OUT/$NAME/train.log"
    [ -d "$OUT/$NAME/point_cloud/iteration_100000" ]
}
if want train; then
    [ -f "$DATA/$EXP/sparse/0/images.txt" ] || { log "ABORT: $DATA/$EXP not prepared (run colmap stage)"; exit 1; }
    rm -f "$DATA/$EXP/sparse/0/points3D.ply"
    log "=== TRAIN $NAME (res $RES) attempt 1 ==="
    if train_once; then
        log "$NAME: attempt 1 succeeded"
    else
        log "$NAME: attempt 1 FAILED (OOM?) — attempt 2 with --densify_grad_threshold 0.0004"
        rm -rf "$OUT/$NAME"
        train_once --densify_grad_threshold 0.0004 && log "$NAME: attempt 2 succeeded (NOTE: densify_grad_threshold 0.0004)" \
            || { log "$NAME: attempt 2 FAILED — giving up"; exit 1; }
    fi
fi

# ── 4. render test views + metrics ───────────────────────────────────────────
if want eval; then
    log "=== RENDER $NAME (test views) ==="
    $GS python /workspace/render.py -m "/outroot/$NAME" --device "$GPU" --skip_train || log "render $NAME failed"
    log "=== METRICS $NAME ==="
    $GS env TORCH_HOME=/torch_cache PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
        python /workspace/metrics.py -m "/outroot/$NAME" --device "$GPU" 2>&1 | tee "$OUT/$NAME/metrics_v2.log"
    log "=== SUMMARY ==="
    for n in exp6_colmap_guided_r1 exp4_colmap_perimage "$NAME"; do
        [ -f "$OUT/$n/results.json" ] && { echo "--- $n"; cat "$OUT/$n/results.json"; echo; } || echo "--- $n: no results.json"
    done
fi
log "all done."
