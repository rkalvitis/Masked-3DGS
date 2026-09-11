#!/bin/bash
# Transfer the exp10/exp11 inputs + scripts to rhea and launch the run there.
#
#   bash tools/sync_exp10_11_to_rhea.sh [launch]      (default: sync only)
#
# Uses plain ssh/rsync to the `rhea` host of ~/.ssh/config. rhea takes a
# password from this Mac, so either run this yourself (you will be asked a few
# times), or open a shared connection once and everything after it is silent:
#
#   ssh -o ControlMaster=auto -o ControlPath=~/.ssh/cm-%C -o ControlPersist=60m rhea true
#
# (this script passes the same ControlPath, so it reuses that session).
# Data is copied with --ignore-existing: nothing already on the server is touched.
set -eu
HOST=${HOST:-rhea}
ROOT=${ROOT:-/media/white/nanodrones/roberts.kalvitis/3dgs/popillia/362img_01_09_masked}
CODE=${CODE:-/home/robertsk/3dgs-masked}
DS=${DS:-$HOME/Documents/repos/optitrack/datasets/aug31_20deg_9vertical}
TOOLS=$(cd "$(dirname "$0")" && pwd)
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-%C -o ControlPersist=60m"
RS="rsync -av -e \"ssh $SSH_OPTS\""

echo "== data (new files only) -> $HOST:$ROOT/data/"
eval $RS --ignore-existing "$DS/gs_upload/intrinsics_factory.csv" "$HOST:$ROOT/data/"
eval $RS --ignore-existing "$DS/gs_upload/sparse_mocap_factoryK" "$DS/gs_upload/sparse_mocap_sfmrefit" "$HOST:$ROOT/data/"
echo "== scripts -> $HOST:$CODE/tools/"
eval $RS "$TOOLS/run_exp10_11_factoryK.sh" "$TOOLS/run_exp9_pose_prior_mapper.sh" "$HOST:$CODE/tools/"
ssh $SSH_OPTS "$HOST" "ls -d $ROOT/data/sparse_mocap_factoryK/0 $ROOT/data/sparse_mocap_sfmrefit/0 $ROOT/data/intrinsics_factory.csv $CODE/tools/run_exp10_11_factoryK.sh"

if [ "${1:-}" = launch ]; then
    LOG=$ROOT/run_exp10_11_$(date +%Y%m%d_%H%M).log
    echo "== launching on $HOST (GPU ${GPU:-0}, RUN ${RUN:-factoryK}, PRIOR ${PRIOR:-sparse_mocap_factoryK}) -> $LOG"
    ssh $SSH_OPTS "$HOST" "cd $ROOT && GPU=${GPU:-0} RUN=${RUN:-factoryK} PRIOR=${PRIOR:-sparse_mocap_factoryK} ITERS=${ITERS:-30000} TRAIN_SCALE=${TRAIN_SCALE:-100} \
        nohup bash $CODE/tools/run_exp10_11_factoryK.sh ${STAGE:-all} > $LOG 2>&1 < /dev/null & echo started; sleep 3; tail -5 $LOG"
fi
