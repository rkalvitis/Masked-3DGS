#!/bin/bash
# =============================================================================
# Re-run metrics.py (masked SSIM / PSNR / bbox-crop LPIPS) on already rendered
# experiment output folders, one GPU, sequential.
#
#   bash tools/run_metrics.sh [--gpu N] [--render] <output-dir> [<output-dir> ...]
#
#   --gpu N    CUDA device index (default 0)
#   --render   re-render test views first (needed if test/ours_*/masks is missing)
#
# Each <output-dir> is a trained model folder (contains cfg_args, point_cloud/,
# and after rendering test/ours_<iter>/{gt,renders,masks}). Previous
# results.json / per_view.json are kept as *_oldmetric.json.
#
#   ROOT=/media/white/nanodrones/roberts.kalvitis/3dgs/popillia/362img_01_09_masked
#   nohup bash /home/robertsk/3dgs-masked/tools/run_metrics.sh \
#       $ROOT/output/exp7_mocap_r1 $ROOT/output/exp6_colmap_guided_r1 \
#       $ROOT/output/exp4_colmap_perimage $ROOT/output/exp2_colmapfree_100k \
#       > $ROOT/run_metrics.log 2>&1 &
# =============================================================================
set -u
CODE=${CODE:-/home/robertsk/3dgs-masked}
TORCH_CACHE=${TORCH_CACHE:-/media/white/nanodrones/roberts.kalvitis/3dgs/torch_cache}
SIF_GS=${SIF_GS:-$HOME/containers/masked-3dgs.sif}
GPU=0; RENDER=0
while [ $# -gt 0 ]; do
    case $1 in
        --gpu) GPU=$2; shift 2 ;;
        --render) RENDER=1; shift ;;
        *) break ;;
    esac
done
[ $# -gt 0 ] || { echo "usage: $0 [--gpu N] [--render] <output-dir> ..."; exit 1; }
unset SINGULARITYENV_CUDA_VISIBLE_DEVICES
log() { echo "[$(date '+%F %T')] $*"; }

grep -q masked_ssim "$CODE/metrics.py" || { log "ABORT: $CODE/metrics.py is not the mask-aware version (sync the code)"; exit 1; }
[ -f "$TORCH_CACHE/hub/checkpoints/vgg16-397923af.pth" ] || log "WARNING: VGG weights missing in $TORCH_CACHE — LPIPS will fail"

for exp in "$@"; do
    exp=${exp%/}
    [ -f "$exp/cfg_args" ] || { log "skip $exp: no cfg_args (not a model folder)"; continue; }
    name=$(basename "$exp"); parent=$(dirname "$exp")
    # render.py needs the source data. Training ran with -s /data/<exp>, so
    # cfg_args holds a container path: bind <ROOT>/data (sibling of output/) to
    # /data, and additionally bind the path verbatim if it exists on the host.
    src=$(sed -n "s/.*source_path='\([^']*\)'.*/\1/p" "$exp/cfg_args")
    data=${DATA:-$(dirname "$parent")/data}
    GS="singularity exec --nv --cleanenv --contain \
        --bind $CODE:/workspace --bind $parent:/outroot --bind $TORCH_CACHE:/torch_cache"
    [ -d "$data" ] && GS="$GS --bind $data:/data"
    [ -n "$src" ] && [ -d "$src" ] && [ "${src#/data/}" = "$src" ] && GS="$GS --bind $src:$src"
    GS="$GS $SIF_GS"

    if [ $RENDER -eq 1 ] || ! ls -d "$exp"/test/ours_*/masks >/dev/null 2>&1; then
        log "=== RENDER $name (test views) ==="
        $GS python /workspace/render.py -m "/outroot/$name" --device "$GPU" --skip_train \
            2>&1 | tee "$exp/render_test.log" || { log "render $name failed"; continue; }
    fi

    for f in results.json per_view.json; do
        [ -f "$exp/$f" ] && cp "$exp/$f" "$exp/${f%.json}_oldmetric.json"
    done
    log "=== METRICS $name ==="
    $GS env TORCH_HOME=/torch_cache PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
        python /workspace/metrics.py -m "/outroot/$name" --device "$GPU" \
        2>&1 | tee "$exp/metrics_v2.log" || log "metrics $name failed"
done

log "=== SUMMARY ==="
for exp in "$@"; do
    exp=${exp%/}
    [ -f "$exp/results.json" ] && { echo "--- $(basename $exp)"; cat "$exp/results.json"; echo; } \
        || echo "--- $(basename $exp): no results.json"
done
log "all done."
