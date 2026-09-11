#!/bin/bash
# =============================================================================
# ring362 — exp10 (mocap + hand-eye as HARD guidance) and exp11 (mocap as SOFT
# prior), both with the FACTORY intrinsics fixed, 30k-iteration 3DGS.
#
#   db     one feature database with ONE PINHOLE camera PER IMAGE whose params are
#          the iPhone factory intrinsics of that photo (data/intrinsics_factory.csv:
#          fx from the lens position, pp = sensor centre). Masked SIFT + exhaustive
#          guided matching, same settings as the free run. Shared by exp10/exp11.
#   exp10  prior = mocap body poses + hand-eye -> lens (data/$PRIOR/0, built on the
#          Mac by build_mocap_colmap_model.py) -> point_triangulator -> bundle_adjuster
#          with intrinsics FIXED, thresholds 200 -> 100 -> 50 -> 20 px -> 3DGS.
#          (exp6/exp8 route; expect it to fail the same way if the prior is still
#          >100 px off — the audit lines tell.)
#   exp11  pose_prior_mapper: incremental SfM from the matches, mocap centres as
#          Gaussian position priors (std PRIOR_STD, robust loss), intrinsics FIXED
#          -> metric insect-frame model without post-hoc alignment -> 3DGS.
#
# REPRODUCIBILITY CONTRACT: nothing existing is deleted or overwritten. Every
# output is under a RUN tag; a stage aborts if its output folder already exists.
# The training copy of each model is SCALED by TRAIN_SCALE (default 100 = cm):
# the stock rasterizer culls everything closer than 0.2 units to the camera and
# this container may not carry the 1 cm near-cull patch (checked and printed).
#
# Inputs (read-only, synced from the Mac gs_upload/):
#   data/images, data/masks                  362 sensor-native JPEGs + masks
#   data/intrinsics_factory.csv              per-image factory intrinsics (image,fx,fy,cx,cy,...)
#   data/$PRIOR/0                            prior model: sparse_mocap_factoryK (default) or
#                                            sparse_mocap_sfmrefit (hand-eye refitted on the SfM)
#   data/colmap_work/colmap_masks            dilated feature masks of the free run (else remade)
#   data/colmap_work2/sfm/<largest>          free perimage model (audit only, optional)
#
#   bash tools/run_exp10_11_factoryK.sh [db|exp10|exp11|all]        (default all)
#   RUN=factoryK  DBRUN=<db tag, default RUN>  PRIOR=sparse_mocap_factoryK  PRIOR_STD=0.015  ITERS=30000
#   TRAIN_SCALE=100  RES=1  GPU=0
#
# Launch on rhea (after syncing the code and rsync-ing gs_upload/ to $ROOT/data/):
#   ROOT=/media/white/nanodrones/roberts.kalvitis/3dgs/popillia/362img_01_09_masked
#   nohup bash /home/robertsk/3dgs-masked/tools/run_exp10_11_factoryK.sh > $ROOT/run_exp10_11_$(date +%Y%m%d_%H%M).log 2>&1 &
# =============================================================================
set -u
STAGE=${1:-all}
ROOT=${ROOT:-/media/white/nanodrones/roberts.kalvitis/3dgs/popillia/362img_01_09_masked}
DATA=$ROOT/data; OUT=$ROOT/output
GPU=${GPU:-0}; RES=${RES:-1}
RUN=${RUN:-factoryK}
PRIOR=${PRIOR:-sparse_mocap_factoryK}
PRIOR_STD=${PRIOR_STD:-0.015}
ITERS=${ITERS:-30000}
TRAIN_SCALE=${TRAIN_SCALE:-100}
CODE=${CODE:-/home/robertsk/3dgs-masked}
TORCH_CACHE=${TORCH_CACHE:-/media/white/nanodrones/roberts.kalvitis/3dgs/torch_cache}
SIF_GS=${SIF_GS:-$HOME/containers/masked-3dgs.sif}
SIF_PY=${SIF_PY:-$HOME/containers/3dgs.sif}; [ -f "$SIF_PY" ] || SIF_PY=$SIF_GS
SIF_COL=${SIF_COL:-$HOME/containers/colmap-cuda.sif}
INTR=$DATA/intrinsics_factory.csv
PRIOR_MODEL=$DATA/$PRIOR/0
DBRUN=${DBRUN:-$RUN}                      # feature database tag (reuse one db for several RUNs)
DBWORK=$DATA/colmap_work_factoryK_$DBRUN
DB=$DBWORK/database.db
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
[ -f "$INTR" ] || { log "ABORT: $INTR missing (rsync gs_upload/intrinsics_factory.csv)"; exit 1; }
[ -f "$PRIOR_MODEL/images.txt" ] || { log "ABORT: $PRIOR_MODEL missing (rsync gs_upload/$PRIOR/)"; exit 1; }
grep -q "Principal point honoured" "$CODE/scene/dataset_readers.py" || { log "ABORT: trainer lacks the principal-point fix"; exit 1; }
AUX=/opt/masked-3dgs-src/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h
if $GS grep -q "p_view.z <= 0.01f" $AUX 2>/dev/null; then PATCH=yes; else PATCH=no; fi
log "RUN=$RUN prior=$PRIOR std=$PRIOR_STD iters=$ITERS train_scale=$TRAIN_SCALE res=$RES | container near-cull patch (1 cm): $PATCH"
[ "$PATCH" = no ] && [ "$TRAIN_SCALE" = 1 ] && { log "ABORT: no near-cull patch in the container and TRAIN_SCALE=1 -> nothing would render"; exit 1; }

# ── helpers (python in 3dgs.sif) ─────────────────────────────────────────────
# prior model with database-consistent image/camera ids, per-image factory K
make_prior() {   # make_prior <db> <prior model dir> <out dir> <intrinsics csv>
    $PY python - "$1" "$2" "$3" "$4" <<'PYEOF'
import csv, os, sqlite3, sys
db, model, out, intr = sys.argv[1:5]
con = sqlite3.connect(db)
imgs = {n: (i, c) for i, n, c in con.execute('SELECT image_id, name, camera_id FROM images')}
K = {r['image'] + '.jpg': (r['fx'], r['fy'], r['cx'], r['cy']) for r in csv.DictReader(open(intr))}
poses = {}
for line in open(f'{model}/images.txt'):
    t = line.split()
    if len(t) == 10 and not t[0].startswith('#') and t[9].lower().endswith('.jpg'):
        poses[t[9]] = t[1:8]
os.makedirs(out, exist_ok=True)
open(f'{out}/points3D.txt', 'w').close()
n = 0
with open(f'{out}/cameras.txt', 'w') as fc, open(f'{out}/images.txt', 'w') as fi:
    for name in sorted(poses):
        if name not in imgs: continue
        iid, cid = imgs[name]; fx, fy, cx, cy = K[name]
        fc.write(f'{cid} PINHOLE 4032 3024 {fx} {fy} {cx} {cy}\n')
        fi.write(f'{iid} {" ".join(poses[name])} {cid} {name}\n\n'); n += 1
print(f'prior: {n} images (expect 362), db-consistent ids, factory K per image')
assert n >= 3
PYEOF
}
# audit a TXT model against the prior (no alignment) and the free perimage model (sim3)
audit() {   # audit <model dir> <prior dir>
    FREE=""; FREE_N=0
    for d in "$DATA"/colmap_work2/sfm/*/; do
        [ -f "$d/images.bin" ] || [ -f "$d/images.txt" ] || continue
        c="$1/free_$(basename "$d")"; mkdir -p "$c"
        $COL colmap model_converter --input_path "$d" --output_path "$c" --output_type TXT >/dev/null 2>&1
        n=$(grep -c '\.jpg' "$c/images.txt" 2>/dev/null || echo 0)
        [ "$n" -gt "$FREE_N" ] && { FREE=$c; FREE_N=$n; }
    done
    $PY python - "$1" "$2" "${FREE:-none}" <<'PYEOF'
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
def rot(Ra, A, B, names): return np.array([np.degrees(np.arccos(np.clip((np.trace((Ra @ A[n][1]).T @ B[n][1]) - 1) / 2, -1, 1))) for n in names])
res, prior = read(f'{res_dir}/images.txt'), read(f'{prior_dir}/images.txt')
common = sorted(set(res) & set(prior))
d = np.array([np.linalg.norm(res[n][0] - prior[n][0]) for n in common]) * 1000
r = rot(np.eye(3), prior, res, common)
print(f'result vs PRIOR (no alignment): {len(common)} cams, moved {pct(d)} mm, turned {pct(r)} deg')
if free_dir != 'none':
    free = read(f'{free_dir}/images.txt')
    for label, M in (('PRIOR', prior), ('RESULT', res)):
        common = sorted(set(M) & set(free))
        A = np.array([free[n][0] for n in common]); B = np.array([M[n][0] for n in common])
        s, R, t = umeyama(A, B)
        d = np.linalg.norm((s * (R @ A.T).T + t) - B, axis=1) * 1000
        print(f'{label} vs free perimage SfM (sim3, scale {s:.5f}): pos {pct(d)} mm, rot {pct(rot(R, free, M, common))} deg')
else:
    print('free perimage model not found under data/colmap_work2/sfm — comparison skipped')
PYEOF
}
# training copy of a TXT model: lengths x TRAIN_SCALE, images/masks links
make_train_folder() {   # make_train_folder <model dir> <exp folder>
    fresh "$2"; mkdir -p "$2/sparse/0"
    $PY python - "$1" "$2/sparse/0" "$TRAIN_SCALE" <<'PYEOF'
import shutil, sys
src, dst, s = sys.argv[1], sys.argv[2], float(sys.argv[3])
shutil.copy2(f'{src}/cameras.txt', f'{dst}/cameras.txt')
with open(f'{src}/images.txt') as fi, open(f'{dst}/images.txt', 'w') as fo:
    for line in fi:
        t = line.split()
        if len(t) >= 10 and not t[0].startswith('#') and t[9].lower().endswith('.jpg'):
            t[5:8] = [f'{float(v) * s:.9f}' for v in t[5:8]]
            fo.write(' '.join(t[:10]) + '\n\n')     # observations dropped (not needed for training)
        elif line.startswith('#'):
            fo.write(line)
n = 0
with open(f'{src}/points3D.txt') as fi, open(f'{dst}/points3D.txt', 'w') as fo:
    for line in fi:
        if line.startswith('#') or not line.strip():
            fo.write(line); continue
        t = line.split(); t[1:4] = [f'{float(v) * s:.6f}' for v in t[1:4]]
        fo.write(' '.join(t[:8]) + '\n'); n += 1
open(f'{dst}/../scale.txt', 'w').write(f'model lengths = metres x {s:g}\n')
print(f'training model: lengths x {s:g}, {n} points -> {dst}')
PYEOF
    for d in images masks; do
        src=$(readlink -f "$DATA/exp6_colmap_guided/$d"); [ -d "$src" ] || src=$DATA/$d
        rel=$($PY python -c "import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" "$src" "$2")
        ln -s "$rel" "$2/$d"
    done
}
train_eval() {   # train_eval <exp folder name> <model output name>
    local exp=$1 name=$2
    fresh "$OUT/$name"; mkdir -p "$OUT/$name"
    log "=== TRAIN $name ($ITERS iterations, res $RES) ==="
    $GS env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
        python /workspace/train.py -s "/data/$exp" -m "/outroot/$name" \
        --masks masks --lambda_mask 0.1 --eval --disable_viewer \
        --data_device cpu --device "$GPU" -r "$RES" --random_background \
        --iterations "$ITERS" --densify_until_iter $((ITERS / 2)) \
        --save_iterations "$ITERS" --test_iterations 7000 "$ITERS" \
        2>&1 | tee "$OUT/$name/train.log"
    [ -d "$OUT/$name/point_cloud/iteration_$ITERS" ] || { log "$name: training FAILED (see train.log)"; return 1; }
    grep -m1 "Principal point honoured" "$OUT/$name/train.log" || log "WARNING: principal-point fix not active for $name"
    log "=== RENDER + METRICS $name ==="
    $GS python /workspace/render.py -m "/outroot/$name" --device "$GPU" --skip_train || log "render $name failed"
    $GS env TORCH_HOME=/torch_cache PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
        python /workspace/metrics.py -m "/outroot/$name" --device "$GPU" 2>&1 | tee "$OUT/$name/metrics.log"
    [ -f "$OUT/$name/results.json" ] && { echo "--- $name"; cat "$OUT/$name/results.json"; echo; }
}

# ── 1. db: per-image factory-K feature database ──────────────────────────────
if want db; then
    fresh "$DBWORK"; mkdir -p "$DBWORK"
    CM=$DATA/colmap_work/colmap_masks
    if [ ! -d "$CM" ] || [ "$(ls "$CM" | wc -l)" -lt 362 ]; then
        CM=$DBWORK/colmap_masks; mkdir -p "$CM"
        log "making dilated COLMAP feature masks in $CM"
        $PY python - "$DATA/images" "$DATA/masks" "$CM" <<'PYEOF'
import os, sys, cv2, numpy as np
imgs, msrc, out = sys.argv[1:4]
k = np.ones((20, 20), np.uint8); n = 0
for name in sorted(os.listdir(imgs)):
    if not name.lower().endswith('.jpg'): continue
    m = cv2.imread(f'{msrc}/{os.path.splitext(name)[0]}.png', cv2.IMREAD_GRAYSCALE)
    assert m is not None, f'no mask for {name}'
    cv2.imwrite(f'{out}/{name}.png', cv2.dilate((m > 127).astype(np.uint8) * 255, k)); n += 1
print(f'{n} COLMAP masks written')
PYEOF
    fi
    KMED=$($PY python -c "
import csv, sys, numpy as np
r = list(csv.DictReader(open(sys.argv[1])))
print(','.join(f'{np.median([float(x[k]) for x in r]):.4f}' for k in ('fx','fy','cx','cy')))" "$INTR")
    EX_MAXSZ=$(findopt feature_extractor max_image_size); EX_PEAK=$(findopt feature_extractor peak_threshold)
    EX_NFEAT=$(findopt feature_extractor max_num_features); EX_GPU=$(findopt feature_extractor use_gpu)
    EX_IDX=$(findopt feature_extractor gpu_index); IR_MASK=$(findopt feature_extractor mask_path)
    M_GPU=$(findopt exhaustive_matcher use_gpu); M_IDX=$(findopt exhaustive_matcher gpu_index)
    M_GUIDED=$(findopt exhaustive_matcher guided_matching); M_RATIO=$(findopt exhaustive_matcher max_ratio)
    log "=== db: masked SIFT, PINHOLE per image (init $KMED), masks $CM ==="
    $COL colmap feature_extractor \
        --database_path "$DB" --image_path "$DATA/images" \
        ${IR_MASK:---ImageReader.mask_path} "$CM" \
        --ImageReader.camera_model PINHOLE --ImageReader.single_camera 0 \
        --ImageReader.camera_params "$KMED" \
        ${EX_MAXSZ:+$EX_MAXSZ 3200} ${EX_PEAK:+$EX_PEAK 0.004} ${EX_NFEAT:+$EX_NFEAT 16384} \
        ${EX_GPU:+$EX_GPU 1} ${EX_IDX:+$EX_IDX "$GPU"} \
        || { log "feature extraction FAILED"; exit 1; }
    log "=== db: per-image factory intrinsics into the cameras table ==="
    $PY python - "$DB" "$INTR" <<'PYEOF'
import csv, sqlite3, sys
import numpy as np
db, intr = sys.argv[1:3]
K = {r['image'] + '.jpg': [float(r[k]) for k in ('fx', 'fy', 'cx', 'cy')] for r in csv.DictReader(open(intr))}
con = sqlite3.connect(db)
imgs = con.execute('SELECT image_id, name, camera_id FROM images').fetchall()
use = {}
for _, _, c in imgs: use[c] = use.get(c, 0) + 1
assert all(v == 1 for v in use.values()), 'database is not one-camera-per-image'
n = 0
for _, name, cid in imgs:
    model, w, h = con.execute('SELECT model, width, height FROM cameras WHERE camera_id=?', (cid,)).fetchone()
    assert model == 1 and (w, h) == (4032, 3024), (name, model, w, h)
    con.execute('UPDATE cameras SET params=?, prior_focal_length=1 WHERE camera_id=?',
                (np.array(K[name], dtype=np.float64).tobytes(), cid)); n += 1
con.commit()
fx = np.array([K[nm][0] for _, nm, _ in imgs])
print(f'{n} cameras set to factory K (fx p10 {np.percentile(fx,10):.0f} p50 {np.median(fx):.0f} p90 {np.percentile(fx,90):.0f}, pp {K[imgs[0][1]][2]:.1f},{K[imgs[0][1]][3]:.1f})')
PYEOF
    [ $? -eq 0 ] || { log "ABORT: setting intrinsics failed"; exit 1; }
    log "=== db: exhaustive guided matching ==="
    $COL colmap exhaustive_matcher --database_path "$DB" \
        ${M_GUIDED:+$M_GUIDED 1} ${M_RATIO:+$M_RATIO 0.85} ${M_GPU:+$M_GPU 1} ${M_IDX:+$M_IDX "$GPU"} \
        || { log "matching FAILED"; exit 1; }
    $PY python - "$DB" <<'PYEOF'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
print(f'database: {c.execute("SELECT SUM(rows) FROM keypoints").fetchone()[0]} keypoints, '
      f'{c.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0").fetchone()[0]} verified pairs')
PYEOF
fi

# ── 2. exp10: hard guidance (prior -> triangulate -> BA, K fixed) ────────────
if want exp10; then
    [ -f "$DB" ] || { log "ABORT: $DB missing (run the db stage with RUN=$RUN)"; exit 1; }
    W10=$DATA/colmap_work_exp10_$RUN; EXP10=exp10_guided_$RUN; NAME10=$EXP10; [ "$RES" = 1 ] && NAME10=${EXP10}_r1
    fresh "$W10"; mkdir -p "$W10"
    printf 'run=%s\nprior=%s\nintrinsics=%s\ndb=%s\niters=%s\ntrain_scale=%s\ndate=%s\n' "$RUN" "$PRIOR_MODEL" "$INTR" "$DB" "$ITERS" "$TRAIN_SCALE" "$(date '+%F %T')" > "$W10/run_params.txt"
    log "=== exp10: prior from $PRIOR_MODEL ==="
    make_prior "$DB" "$PRIOR_MODEL" "$W10/prior" "$INTR" || { log "ABORT: prior failed"; exit 1; }
    T_FILT=$(findopt point_triangulator filter_max_reproj_error); T_COMP=$(findopt point_triangulator tri_complete_max_reproj_error)
    T_MERG=$(findopt point_triangulator tri_merge_max_reproj_error)
    BA_FL=$(findopt bundle_adjuster refine_focal_length); BA_PP=$(findopt bundle_adjuster refine_principal_point)
    BA_EP=$(findopt bundle_adjuster refine_extra_params)
    IN=$W10/prior
    # the prior is ~460 px off (factoryK) / ~260 px (sfmrefit) vs the free SfM, so the
    # first round must accept more than that or nothing survives triangulation
    for round in 500 200 100 50 20; do
        log "=== exp10: triangulate (max reproj $round px) + bundle adjust (K FIXED) ==="
        mkdir -p "$W10/tri_r$round" "$W10/tri_ba$round"
        $COL colmap point_triangulator --database_path "$DB" --image_path "$DATA/images" \
            --input_path "$IN" --output_path "$W10/tri_r$round" \
            ${T_FILT:+$T_FILT $round} ${T_COMP:+$T_COMP $round} ${T_MERG:+$T_MERG $round} \
            || { log "exp10 triangulation FAILED"; exit 1; }
        $COL colmap bundle_adjuster --input_path "$W10/tri_r$round" --output_path "$W10/tri_ba$round" \
            ${BA_FL:---BundleAdjustment.refine_focal_length} 0 ${BA_PP:---BundleAdjustment.refine_principal_point} 0 \
            ${BA_EP:---BundleAdjustment.refine_extra_params} 0 \
            || { log "exp10 bundle adjustment FAILED"; exit 1; }
        IN=$W10/tri_ba$round
    done
    mkdir -p "$W10/tri"
    $COL colmap model_converter --input_path "$IN" --output_path "$W10/tri" --output_type TXT
    log "exp10 model: $(grep -c '\.jpg' "$W10/tri/images.txt") images, $(grep -vc '^#' "$W10/tri/points3D.txt") points"
    audit "$W10/tri" "$W10/prior"
    make_train_folder "$W10/tri" "$DATA/$EXP10"; cp "$W10/run_params.txt" "$DATA/$EXP10/"
    train_eval "$EXP10" "$NAME10" || true
fi

# ── 3. exp11: soft prior (pose_prior_mapper, K fixed) ────────────────────────
if want exp11; then
    [ -f "$DB" ] || { log "ABORT: $DB missing (run the db stage with RUN=$RUN)"; exit 1; }
    $COL colmap pose_prior_mapper -h >/dev/null 2>&1 || { log "ABORT: this COLMAP has no pose_prior_mapper (needs >= 3.11)"; exit 1; }
    W11=$DATA/colmap_work_exp11_$RUN; EXP11=exp11_pose_prior_$RUN; NAME11=$EXP11; [ "$RES" = 1 ] && NAME11=${EXP11}_r1
    fresh "$W11"; mkdir -p "$W11"
    printf 'run=%s\nprior=%s\nprior_std_m=%s\nintrinsics=%s\ndb=%s\niters=%s\ntrain_scale=%s\ndate=%s\n' "$RUN" "$PRIOR_MODEL" "$PRIOR_STD" "$INTR" "$DB" "$ITERS" "$TRAIN_SCALE" "$(date '+%F %T')" > "$W11/run_params.txt"
    cp "$DB" "$W11/database.db"
    make_prior "$DB" "$PRIOR_MODEL" "$W11/prior" "$INTR" || { log "ABORT: prior failed"; exit 1; }
    log "=== exp11: writing pose priors (std $PRIOR_STD m) into the database copy ==="
    $PY python - "$W11/database.db" "$W11/prior" "$PRIOR_STD" <<'PYEOF'
import sqlite3, sys
import numpy as np
db, model, std = sys.argv[1], sys.argv[2], float(sys.argv[3])
con = sqlite3.connect(db)
cols = [r[1] for r in con.execute('PRAGMA table_info(pose_priors)')]
assert cols, 'no pose_priors table (COLMAP < 3.11)'
assert con.execute('SELECT COUNT(*) FROM pose_priors').fetchone()[0] == 0
imgs = {n: (i, c) for i, n, c in con.execute('SELECT image_id, name, camera_id FROM images')}
def q2R(w, x, y, z):
    n = (w*w + x*x + y*y + z*z) ** 0.5; w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
cov = (np.eye(3) * std * std).astype(np.float64); nan3 = np.full(3, np.nan)
new = 'corr_data_id' in cols; n = 0
for line in open(f'{model}/images.txt'):
    t = line.split()
    if len(t) != 10 or t[0].startswith('#') or t[9] not in imgs: continue
    R = q2R(*map(float, t[1:5])); tv = np.array(list(map(float, t[5:8]))); C = (-R.T @ tv).astype(np.float64)
    iid, cid = imgs[t[9]]
    if new:   # SensorType.CAMERA = 0, CoordinateSystem CARTESIAN = 1
        con.execute('INSERT INTO pose_priors (corr_data_id, corr_sensor_id, corr_sensor_type, position, position_covariance, gravity, coordinate_system) VALUES (?,?,?,?,?,?,?)',
                    (iid, cid, 0, C.tobytes(), cov.tobytes(), nan3.tobytes(), 1))
    else:
        con.execute('INSERT INTO pose_priors (image_id, position, coordinate_system, position_covariance) VALUES (?,?,?,?)', (iid, C.tobytes(), 1, cov.tobytes()))
    n += 1
con.commit(); print(f'{n} pose priors written (schema {"COLMAP>=4" if new else "3.11-3.12"})'); assert n >= 3
PYEOF
    [ $? -eq 0 ] || { log "ABORT: priors failed"; exit 1; }
    P_STDX=$(findopt pose_prior_mapper prior_position_std_x); P_STDY=$(findopt pose_prior_mapper prior_position_std_y)
    P_STDZ=$(findopt pose_prior_mapper prior_position_std_z); P_OVR=$(findopt pose_prior_mapper overwrite_priors_covariance)
    P_ROB=$(findopt pose_prior_mapper use_robust_loss_on_prior_position)
    M_FL=$(findopt pose_prior_mapper ba_refine_focal_length); M_PP=$(findopt pose_prior_mapper ba_refine_principal_point)
    M_EP=$(findopt pose_prior_mapper ba_refine_extra_params)
    mkdir -p "$W11/sfm"
    log "=== exp11: pose_prior_mapper (std $PRIOR_STD m, robust loss, K FIXED at factory) ==="
    $COL colmap pose_prior_mapper --database_path "$W11/database.db" --image_path "$DATA/images" --output_path "$W11/sfm" \
        ${P_STDX:+$P_STDX $PRIOR_STD} ${P_STDY:+$P_STDY $PRIOR_STD} ${P_STDZ:+$P_STDZ $PRIOR_STD} \
        ${P_OVR:+$P_OVR 1} ${P_ROB:+$P_ROB 1} ${M_FL:+$M_FL 0} ${M_PP:+$M_PP 0} ${M_EP:+$M_EP 0} \
        || { log "exp11 pose_prior_mapper FAILED"; exit 1; }
    BEST=""; BEST_N=0
    for d in "$W11"/sfm/*/; do
        [ -f "$d/images.bin" ] || [ -f "$d/images.txt" ] || continue
        c="$W11/cand_$(basename "$d")"; mkdir -p "$c"
        $COL colmap model_converter --input_path "$d" --output_path "$c" --output_type TXT >/dev/null 2>&1
        n=$(grep -c '\.jpg' "$c/images.txt" 2>/dev/null || echo 0); log "model $d: $n images"
        [ "$n" -gt "$BEST_N" ] && { BEST=$c; BEST_N=$n; }
    done
    [ -n "$BEST" ] || { log "ABORT: no model"; exit 1; }
    mkdir -p "$W11/tri"; cp "$BEST"/{cameras,images,points3D}.txt "$W11/tri/"
    log "exp11 model: $BEST_N images (expect 362), $(grep -vc '^#' "$W11/tri/points3D.txt") points"
    audit "$W11/tri" "$W11/prior"
    make_train_folder "$W11/tri" "$DATA/$EXP11"; cp "$W11/run_params.txt" "$DATA/$EXP11/"
    train_eval "$EXP11" "$NAME11" || true
fi

log "=== SUMMARY (results.json) ==="
for n in exp4_colmap_perimage exp6_colmap_guided_r1 exp8_colmap_guided_freefocal_r1 exp10_guided_${RUN}_r1 exp10_guided_$RUN exp11_pose_prior_${RUN}_r1 exp11_pose_prior_$RUN; do
    [ -f "$OUT/$n/results.json" ] && { echo "--- $n"; cat "$OUT/$n/results.json"; echo; }
done
log "all done."
