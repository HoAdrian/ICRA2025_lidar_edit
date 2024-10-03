#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

CONFIG_FILE=./tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext.yaml #custom_cbgs_voxel0075_voxelnext.yaml
CHECKPOINT_FILE=voxelnext_nuscenes_kernel1.pth

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

python -m torch.distributed.launch --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch ${PY_ARGS} --pretrained_model ${CHECKPOINT_FILE} --cfg_file ${CONFIG_FILE}

