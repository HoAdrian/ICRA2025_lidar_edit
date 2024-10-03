#!/bin/bash


#### mmdet3d
conda create -n openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 cudatoolkit-dev=11.3 -c pytorch -c conda-forge
pip install -U openmim
mim install mmengine
#mim install 'mmcv>=2.0.0rc4'
mim install 'mmcv>=2.0.0rc4,<2.2.0'
mim install 'mmdet>=3.0.0'
git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
# "-b dev-1.x" means checkout to the `dev-1.x` branch.
cd mmdetection3d
pip install -v -e .

# pip install spconv

########### this is for OpenPCDet from this point onwards
pip install spconv-cu113
conda install pytorch-scatter -c pyg
pip install kornia==0.5.8


rsync -av --progress --no-perms --inplace --no-group --no-owner /home/shinghei/lidar_generation/nuScenes-lidarseg-all-v1.0.tar.bz2 /mnt/DATA/datasets/extracted/nuscenes
scp -v /home/shinghei/lidar_generation/nuScenes-lidarseg-all-v1.0.tar.bz2 /mnt/DATA/datasets/extracted/nuscenes
tar -xvjf nuScenes-lidarseg-all-v1.0.tar.bz2

tar -xvjf /home/shinghei/lidar_generation/nuScenes-lidarseg-all-v1.0.tar.bz2

