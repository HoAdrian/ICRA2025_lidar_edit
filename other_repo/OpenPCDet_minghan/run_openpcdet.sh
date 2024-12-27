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
# CONFIG_FILE=./tools/cfgs/nuscenes_models/cbgs_pillar0075_res2d_centerpoint.yaml
# CHECKPOINT_FILE=cbgs_pp_centerpoint_nds6070.pth

CONFIG_FILE=./tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml
#CHECKPOINT_FILE=voxelnext_nuscenes_kernel1.pth
#CHECKPOINT_FILE=singlesweep_checkpoint_epoch_20.pth
# CHECKPOINT_FILE=/home/shinghei/lidar_generation/OpenPCDet_minghan/output/tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext/default/ckpt/checkpoint_epoch_20.pth
CHECKPOINT_FILE=singlesweep_no_intensity_scratch_epoch20.pth
#python tools/demo.py --cfg_file ${CONFIG_FILE} --ckpt ${CHECKPOINT_FILE} --data_path ${PC_PATH}

### for lidar-only setting
#python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version $VERSION


python tools/test.py --cfg_file ${CONFIG_FILE} --batch_size 1 --ckpt ${CHECKPOINT_FILE}

############# whether to use pretrained model, train from scratch or train from checkpoint
#python tools/train.py --pretrained_model ${CHECKPOINT_FILE} --cfg_file ${CONFIG_FILE}
#python tools/train.py --cfg_file ${CONFIG_FILE}
#python tools/train.py --ckpt ${CHECKPOINT_FILE} --cfg_file ${CONFIG_FILE}

###################################################################################
############### Custom Nuscenes ##############################################
###################################################################################


CONFIG_FILE=./tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext.yaml #custom_cbgs_voxel0075_voxelnext.yaml
#CHECKPOINT_FILE=voxelnext_nuscenes_kernel1.pth
#CHECKPOINT_FILE=singlesweep_checkpoint_epoch_20.pth
CHECKPOINT_FILE=singlesweep_no_intensity_scratch_epoch20.pth


# MY_ROOT=/home/shinghei/lidar_generation/OpenPCDet_minghan/data/nuscenes
# WHICH_GEN="ours"
# PC_PATH=${MY_ROOT}/${VERSION}/${WHICH_GEN}/${VERSION}/lidar_point_clouds/val_fdddd75ee1d94f14a09991988dab8b3e.bin    #val_9d9bf11fb0e144c8b446d54a8a00184f.bin
#python tools/demo.py --cfg_file ${CONFIG_FILE} --ckpt ${CHECKPOINT_FILE} --data_path ${PC_PATH}

### for lidar-only setting
#python -m pcdet.datasets.nuscenes.custom_nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/custom_nuscenes_dataset.yaml --version $VERSION

#python tools/test.py --cfg_file ${CONFIG_FILE} --batch_size 1 --ckpt ${CHECKPOINT_FILE}

# CHECKPOINT_FILE=/home/shinghei/lidar_generation/OpenPCDet_minghan/output/tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext/default/ckpt/checkpoint_epoch_67.pth
# python tools/train.py --ckpt ${CHECKPOINT_FILE} --cfg_file ${CONFIG_FILE}

#python tools/train.py --pretrained_model ${CHECKPOINT_FILE} --cfg_file ${CONFIG_FILE}
#python tools/train.py --cfg_file ${CONFIG_FILE}














# cp nuscenes_infos_1sweeps_val.pkl ./ours_infos
# cp nuscenes_infos_1sweeps_train.pkl ./ours_infos
# cp nuscenes_dbinfos_1sweeps_withvelo.pkl ./ours_infos

# cp nuscenes_infos_1sweeps_val.pkl ./naive_infos
# cp nuscenes_infos_1sweeps_train.pkl ./naive_infos
# cp nuscenes_dbinfos_1sweeps_withvelo.pkl ./naive_infos

