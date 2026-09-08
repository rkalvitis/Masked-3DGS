#!/bin/bash
# =============================================================================
# COLMAP pose refinement for the popillia dataset — SERVER-SIDE (rhea).
#
# Staged so the COLMAP steps can run in the CUDA COLMAP container (GPU SIFT
# + matching) while the python steps run in 3dgs.sif. One-time setup:
#
#   singularity pull ~/containers/colmap-cuda.sif docker://colmap/colmap:latest
#
# Run — code bound at /workspace, dataset at /data:
#
#   export CODE_DIR=/home/robertsk/3dgs-masked
#   export DATA_DIR=/media/white/nanodrones/roberts.kalvitis/3dgs/popillia/48img_20_08_masked/data
#   S="singularity exec --nv --cleanenv --contain --bind $CODE_DIR:/workspace --bind $DATA_DIR:/data"
#
#   $S ~/containers/3dgs.sif        bash /workspace/tools/refine_on_server.sh /data prep
#   $S ~/containers/colmap-cuda.sif bash /workspace/tools/refine_on_server.sh /data sfm
#   $S ~/containers/3dgs.sif        bash /workspace/tools/refine_on_server.sh /data prior
#   $S ~/containers/colmap-cuda.sif bash /workspace/tools/refine_on_server.sh /data tri
#   $S ~/containers/3dgs.sif        bash /workspace/tools/refine_on_server.sh /data align
#
# (python stages — prep/prior/align — run in 3dgs.sif; the colmap-cuda
#  image has no python. COLMAP stages — sfm/tri — run in colmap-cuda.sif.)
#
# prep : COLMAP feature masks (masks_sfm/ = beetle+mount assembly if
#        present, else the training masks) + fixed-intrinsics params file
# sfm  : masked SIFT (GPU) -> exhaustive+guided matching (GPU) -> free
#        incremental mapping. May fail to initialise on this data — fine.
# tri  : fallback/companion that CANNOT fail to initialise: triangulate
#        points with the mocap poses held as prior, then bundle-adjust the
#        poses (intrinsics fixed), loose -> tight. All 48 cameras kept;
#        ones without matches simply keep their mocap pose.
# align: sim3 alignment onto the mocap model -> /data/sparse_refined/0.
#        Prefers a big sfm model; falls back to the tri model.
#
# Option names are discovered from the binary (COLMAP 3.12 renamed groups).
# =============================================================================
set -euo pipefail

DS="${1:?usage: refine_on_server.sh <dataset dir> [prep|sfm|prior|tri|align|all]}"
STAGE="${2:-all}"
WORK="$DS/colmap_work"
USE_GPU="${USE_GPU:-1}"
GPU_INDEX="${GPU_INDEX:-0}"

findopt() {  # findopt <subcommand> <option-suffix>
    colmap "$1" -h 2>&1 | grep -o -- "--[A-Za-z_]*\.$2" | head -1
}

if [ "$STAGE" = "prep" ] || [ "$STAGE" = "all" ]; then
    mkdir -p "$WORK/colmap_masks" "$WORK/sfm"
    MASKS_SRC="$DS/masks"
    [ -d "$DS/masks_sfm" ] && MASKS_SRC="$DS/masks_sfm"
    echo "feature masks from: $MASKS_SRC"

    python - "$DS" "$WORK" "$MASKS_SRC" <<'EOF'
import os, sys
import cv2
import numpy as np
ds, work, msrc = sys.argv[1], sys.argv[2], sys.argv[3]
kernel = np.ones((20, 20), np.uint8)
names = sorted(n for n in os.listdir(f'{ds}/images') if n.endswith('.jpg'))
for name in names:
    m = cv2.imread(f'{msrc}/{name[:-4]}.png', cv2.IMREAD_GRAYSCALE)
    m = cv2.dilate((m > 127).astype(np.uint8) * 255, kernel)
    cv2.imwrite(f'{work}/colmap_masks/{name}.png', m)
print(f'{len(names)} COLMAP masks written')
EOF

    python - "$DS" > "$WORK/cam_params.txt" <<'EOF'
import sys
line = [l.split() for l in open(f'{sys.argv[1]}/sparse/0/cameras.txt')
        if not l.startswith('#')][0]
print(','.join(line[4:8]))
EOF
    echo "prep done: masks + cam_params $(cat "$WORK/cam_params.txt")"
fi

if [ "$STAGE" = "sfm" ] || [ "$STAGE" = "all" ]; then
    CAM_PARAMS=$(cat "$WORK/cam_params.txt")
    EX_MAXSZ=$(findopt feature_extractor max_image_size)
    EX_GPU=$(findopt feature_extractor use_gpu)
    EX_IDX=$(findopt feature_extractor gpu_index)
    EX_PEAK=$(findopt feature_extractor peak_threshold)
    EX_NFEAT=$(findopt feature_extractor max_num_features)
    IR_MASK=$(findopt feature_extractor mask_path)
    M_GPU=$(findopt exhaustive_matcher use_gpu)
    M_IDX=$(findopt exhaustive_matcher gpu_index)
    M_GUIDED=$(findopt exhaustive_matcher guided_matching)
    M_RATIO=$(findopt exhaustive_matcher max_ratio)
    MAP_FL=$(findopt mapper ba_refine_focal_length)
    MAP_PP=$(findopt mapper ba_refine_principal_point)
    MAP_EP=$(findopt mapper ba_refine_extra_params)
    echo "fixed PINHOLE params: $CAM_PARAMS (use_gpu=$USE_GPU idx=$GPU_INDEX)"

    rm -f "$WORK/database.db"
    colmap feature_extractor \
        --database_path "$WORK/database.db" \
        --image_path "$DS/images" \
        ${IR_MASK:---ImageReader.mask_path} "$WORK/colmap_masks" \
        --ImageReader.camera_model PINHOLE \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_params "$CAM_PARAMS" \
        ${EX_MAXSZ:+$EX_MAXSZ 3200} \
        ${EX_PEAK:+$EX_PEAK 0.004} \
        ${EX_NFEAT:+$EX_NFEAT 16384} \
        ${EX_GPU:+$EX_GPU "$USE_GPU"} \
        ${EX_IDX:+$EX_IDX "$GPU_INDEX"}

    colmap exhaustive_matcher \
        --database_path "$WORK/database.db" \
        ${M_GUIDED:+$M_GUIDED 1} \
        ${M_RATIO:+$M_RATIO 0.85} \
        ${M_GPU:+$M_GPU "$USE_GPU"} \
        ${M_IDX:+$M_IDX "$GPU_INDEX"}

    rm -rf "$WORK"/sfm/*
    colmap mapper \
        --database_path "$WORK/database.db" \
        --image_path "$DS/images" \
        --output_path "$WORK/sfm" \
        ${MAP_FL:---Mapper.ba_refine_focal_length} 0 \
        ${MAP_PP:---Mapper.ba_refine_principal_point} 0 \
        ${MAP_EP:---Mapper.ba_refine_extra_params} 0 \
        || echo "mapper failed — use the tri stage"

    if ls -d "$WORK"/sfm/*/ >/dev/null 2>&1; then
        MODEL=$(ls -d "$WORK"/sfm/*/ | head -1)
        colmap model_converter --input_path "$MODEL" \
            --output_path "$MODEL" --output_type TXT
        echo "sfm done: $MODEL"
    fi
fi

if [ "$STAGE" = "prior" ] || [ "$STAGE" = "all" ]; then
    # prior model with database-consistent image ids (python — run this
    # stage in 3dgs.sif, AFTER sfm created the database)
    python - "$DS" "$WORK" <<'EOF'
import os, sqlite3, sys
ds, work = sys.argv[1], sys.argv[2]
ids = dict(sqlite3.connect(f'{work}/database.db')
           .execute('SELECT name, image_id FROM images').fetchall())
os.makedirs(f'{work}/prior', exist_ok=True)
import shutil
shutil.copy2(f'{ds}/sparse/0/cameras.txt', f'{work}/prior/cameras.txt')
open(f'{work}/prior/points3D.txt', 'w').close()
lines = []
# pose lines identified by content: a blank observations line plus an
# alternation toggle used to drop every second image here
for line in open(f'{ds}/sparse/0/images.txt'):
    t = line.split()
    if len(t) != 10 or t[0].startswith('#'):
        continue
    name = t[9]
    if not name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    if name in ids:
        lines.append(' '.join([str(ids[name])] + t[1:9] + [name]))
with open(f'{work}/prior/images.txt', 'w') as f:
    f.write('# prior poses from mocap, db-consistent ids\n')
    for l in lines:
        f.write(l + '\n\n')
print(f'prior model: {len(lines)} images')
EOF
    echo "prior done: $WORK/prior"
fi

if [ "$STAGE" = "tri" ] || [ "$STAGE" = "all" ]; then
    if [ ! -f "$WORK/prior/images.txt" ]; then
        echo "ERROR: no prior model — run the 'prior' stage first (in 3dgs.sif)"
        exit 1
    fi
    T_FILT=$(findopt point_triangulator filter_max_reproj_error)
    T_COMP=$(findopt point_triangulator tri_complete_max_reproj_error)
    T_MERG=$(findopt point_triangulator tri_merge_max_reproj_error)
    BA_FL=$(findopt bundle_adjuster refine_focal_length)
    BA_PP=$(findopt bundle_adjuster refine_principal_point)
    BA_EP=$(findopt bundle_adjuster refine_extra_params)

    IN="$WORK/prior"
    for round in "200" "20"; do
        OUT_T="$WORK/tri_r$round"
        rm -rf "$OUT_T"; mkdir -p "$OUT_T"
        colmap point_triangulator \
            --database_path "$WORK/database.db" \
            --image_path "$DS/images" \
            --input_path "$IN" \
            --output_path "$OUT_T" \
            ${T_FILT:+$T_FILT $round} \
            ${T_COMP:+$T_COMP $round} \
            ${T_MERG:+$T_MERG $round}
        OUT_B="$WORK/tri_ba$round"
        rm -rf "$OUT_B"; mkdir -p "$OUT_B"
        colmap bundle_adjuster \
            --input_path "$OUT_T" \
            --output_path "$OUT_B" \
            ${BA_FL:---BundleAdjustment.refine_focal_length} 0 \
            ${BA_PP:---BundleAdjustment.refine_principal_point} 0 \
            ${BA_EP:---BundleAdjustment.refine_extra_params} 0
        IN="$OUT_B"
    done
    rm -rf "$WORK/tri"; mkdir -p "$WORK/tri"
    colmap model_converter --input_path "$IN" \
        --output_path "$WORK/tri" --output_type TXT
    echo "tri done: $WORK/tri"
fi

if [ "$STAGE" = "align" ] || [ "$STAGE" = "all" ]; then
    MODEL="${3:-}"
    if [ -z "$MODEL" ]; then
        if [ -f "$WORK/tri/images.txt" ]; then
            MODEL="$WORK/tri"
        else
            MODEL=$(ls -d "$WORK"/sfm/*/ | head -1)
        fi
    fi
    echo "aligning model: $MODEL"
    python "${CODE_DIR:-/workspace}/tools/align_sfm_to_mocap.py" \
        "$DS" "$MODEL" --out sparse_refined
    echo "done: $DS/sparse_refined/0"
fi
