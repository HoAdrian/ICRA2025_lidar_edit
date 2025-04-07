#!/bin/bash


VERSION=v1.0-trainval
############ export the cloned nuscenes-devkit repo path
export PYTHONPATH=/home/shinghei/lidar_generation/nuscenes-devkit/python-sdk:$PYTHONPATH

############### FIRST OF ALL, make a symbolic link to the directory containing nuscenes data and the symbolic link should be at /data/nuscenes
# cd data
# ln -s /home/shinghei/lidar_generation/our_ws/data/nuscenes/${VERSION}
# mv ${VERSION} nuscenes

############ REMEMBER TO USE THE ORIGINAL NUSCENES INFOS BEFORE TESTING !!!!!!!!!!!!
echo "REMEMBER TO USE THE ORIGINAL NUSCENES INFOS BEFORE TESTING for training experiment!!!!!!!!!!!!"

#########################################################################################
############### Ordinary Nuscenes ##############################################
#####################################################################################

CONFIG_FILE=./tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml

### for lidar-only setting
#python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version $VERSION

# CHECKPOINT_FILE=/home/shinghei/lidar_generation/OpenPCDet_minghan/ckpts_from_cluster/e40/checkpoint_epoch_40.pth
# python tools/test.py --cfg_file ${CONFIG_FILE} --batch_size 1 --ckpt ${CHECKPOINT_FILE}

############# whether to use pretrained model, train from scratch or train from checkpoint
# python tools/train.py --pretrained_model ${CHECKPOINT_FILE} --cfg_file ${CONFIG_FILE}
#python tools/train.py --cfg_file ${CONFIG_FILE}
# python tools/train.py --ckpt ${CHECKPOINT_FILE} --cfg_file ${CONFIG_FILE}

###################################################################################
############### Custom Nuscenes ##############################################
###################################################################################


CONFIG_FILE=./tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext.yaml #custom_cbgs_voxel0075_voxelnext.yaml

CHECKPOINT_FILE=/home/shinghei/lidar_generation/OpenPCDet_minghan/output_train_on_synthetic_only_car_only/tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext/default/ckpt/checkpoint_epoch_15.pth

### for lidar-only setting
# python -m pcdet.datasets.nuscenes.custom_nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/custom_nuscenes_dataset.yaml --version $VERSION


# python tools/train.py --ckpt ${CHECKPOINT_FILE} --cfg_file ${CONFIG_FILE}
# python tools/train.py --pretrained_model ${CHECKPOINT_FILE} --cfg_file ${CONFIG_FILE} 
# python tools/train.py --cfg_file ${CONFIG_FILE}

# CHECKPOINT_FILE=/home/shinghei/lidar_generation/OpenPCDet_minghan/output_start_from_synthetic_pretrain_car_only_epoch_15_then_train_on_discrete_nusc_30_epochs/tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext/default/ckpt/checkpoint_epoch_30.pth
# CHECKPOINT_FILE=/home/shinghei/lidar_generation/OpenPCDet_minghan/ckpts_from_cluster/e20/checkpoint_epoch_20.pth
# CHECKPOINT_FILE=/home/shinghei/lidar_generation/OpenPCDet_minghan/ckpts_from_cluster/e30/checkpoint_epoch_30.pth
# CHECKPOINT_FILE=/home/shinghei/lidar_generation/OpenPCDet_minghan/singlesweep_no_intensity_scratch_epoch20.pth
# CHECKPOINT_FILE=/home/shinghei/lidar_generation/OpenPCDet_minghan/ckpts_from_cluster/e40/checkpoint_epoch_40.pth
# CHECKPOINT_FILE=/home/shinghei/lidar_generation/OpenPCDet_minghan/ckpts_from_cluster/e35/checkpoint_epoch_35.pth
# CHECKPOINT_FILE=/home/shinghei/lidar_generation/OpenPCDet_minghan/output_start_from_synthetic_pretrain_car_only_epoch15_then_train_on_discrete_nusc_40_epochs/tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext/default/ckpt/checkpoint_epoch_40.pth
# CHECKPOINT_FILE=/home/shinghei/lidar_generation/OpenPCDet_minghan/ckpts_from_cluster/e40_ft20/checkpoint_epoch_40.pth
# CHECKPOINT_FILE=/home/shinghei/lidar_generation/OpenPCDet_minghan/ckpts_from_cluster/e30_ft20/checkpoint_epoch_30.pth
# python tools/test.py --cfg_file ${CONFIG_FILE} --batch_size 1 --ckpt ${CHECKPOINT_FILE}

