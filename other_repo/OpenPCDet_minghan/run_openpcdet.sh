#!/bin/bash

############### FIRST OF ALL, make a symbolic link to the directory containing nuscenes data and the symbolic link should be at /data/nuscenes
VERSION=v1.0-trainval
# export the cloned nuscenes-devkit repo
export PYTHONPATH=/home/shinghei/lidar_generation/nuscenes-devkit/python-sdk:$PYTHONPATH

# cd data
# ln -s /home/shinghei/lidar_generation/our_ws/data/nuscenes/${VERSION}
# mv ${VERSION} nuscenes

############ REMEMBER TO USE THE ORIGINAL NUSCENES INFOS BEFORE TESTING !!!!!!!!!!!!
echo "REMEMBER TO USE THE ORIGINAL NUSCENES INFOS BEFORE TESTING !!!!!!!!!!!!"
############### Ordinary Nuscenes ##############################################
# CONFIG_FILE=./tools/cfgs/nuscenes_models/cbgs_pillar0075_res2d_centerpoint.yaml
# CHECKPOINT_FILE=cbgs_pp_centerpoint_nds6070.pth

CONFIG_FILE=./tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml
#CHECKPOINT_FILE=voxelnext_nuscenes_kernel1.pth

# CONFIG_FILE=tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml
# CHECKPOINT_FILE=pp_multihead_nds5823_updated.pth

# CONFIG_FILE=tools/cfgs/nuscenes_models/cbgs_second_multihead.yaml
# CHECKPOINT_FILE=cbgs_second_multihead_nds6229_updated.pth

# CONFIG_FILE=tools/cfgs/nuscenes_models/transfusion_lidar.yaml
# CHECKPOINT_FILE=cbgs_transfusion_lidar.pth

CHECKPOINT_FILE=/home/shinghei/lidar_generation/OpenPCDet_minghan/output/tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext/default/ckpt/checkpoint_epoch_2.pth

#python tools/demo.py --cfg_file ${CONFIG_FILE} --ckpt ${CHECKPOINT_FILE} --data_path ${PC_PATH}

### for lidar-only setting
#python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version $VERSION

#python tools/test.py --cfg_file ${CONFIG_FILE} --batch_size 1 --ckpt ${CHECKPOINT_FILE}





############### Custom Nuscenes ##############################################



CONFIG_FILE=./tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext.yaml #custom_cbgs_voxel0075_voxelnext.yaml
CHECKPOINT_FILE=voxelnext_nuscenes_kernel1.pth

# CONFIG_FILE=tools/cfgs/nuscenes_models/cbgs_second_multihead.yaml
# CHECKPOINT_FILE=cbgs_second_multihead_nds6229_updated.pth

# CONFIG_FILE=tools/cfgs/nuscenes_models/transfusion_lidar.yaml
# CHECKPOINT_FILE=cbgs_transfusion_lidar.pth


MY_ROOT=/home/shinghei/lidar_generation/OpenPCDet_minghan/data/nuscenes
WHICH_GEN="ours"

# PC_PATH=${MY_ROOT}/${VERSION}/${WHICH_GEN}/${VERSION}/lidar_point_clouds/val_fdddd75ee1d94f14a09991988dab8b3e.bin    #val_9d9bf11fb0e144c8b446d54a8a00184f.bin
#python tools/demo.py --cfg_file ${CONFIG_FILE} --ckpt ${CHECKPOINT_FILE} --data_path ${PC_PATH}

### for lidar-only setting
#python -m pcdet.datasets.nuscenes.custom_nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/custom_nuscenes_dataset.yaml --version $VERSION

python tools/test.py --cfg_file ${CONFIG_FILE} --batch_size 1 --ckpt ${CHECKPOINT_FILE}


#python tools/train.py --pretrained_model ${CHECKPOINT_FILE} --cfg_file ${CONFIG_FILE}














# cp nuscenes_infos_1sweeps_val.pkl ./ours_infos
# cp nuscenes_infos_1sweeps_train.pkl ./ours_infos
# cp nuscenes_dbinfos_1sweeps_withvelo.pkl ./ours_infos

# cp nuscenes_infos_1sweeps_val.pkl ./naive_infos
# cp nuscenes_infos_1sweeps_train.pkl ./naive_infos
# cp nuscenes_dbinfos_1sweeps_withvelo.pkl ./naive_infos

