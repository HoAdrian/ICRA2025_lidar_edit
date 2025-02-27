#!/bin/bash

# run conda info | grep -i 'base environment' to check the path to conda

conda_path = "/home/shinghei/miniconda3"
source "$conda_path/etc/profile.d/conda.sh"

# assuming I am using CUDA of version 11.8
conda create -n ultralidar python=3.8 -y

conda activate ultralidar

conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install timm==0.5.4

pip install scikit-learn

pip3 install open3d

pip install nuscenes-devkit

pip install tensorboard
pip install scikit-image

############# installing poinTr (for point cloud completion), the same environment is used for running AnchorFormer
conda create -n pointTr python=3.8 -y
conda activate pointTr
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 cudatoolkit-dev=11.3 -c pytorch -c conda-forge

pip install timm==0.5.4
pip install scikit-learn
pip3 install open3d
pip install nuscenes-devkit
pip install tensorboard
pip install Ninja
#pip install "git+https://github.com/facebookresearch/pytorch3d.git" #optional

## NOW, follow the instruction on the PoinTr repo: https://github.com/yuxumin/PoinTr
## In the AnchorFormer repo https://github.com/chenzhik/AnchorFormer, install dependencies using the provided bash script
## to do point cloud completion (need to modify some paths)
CUDA_VISIBLE_DEVICES=0 python main.py --ckpts anchorformer_dict.pth --config ./cfgs/custom_models/AnchorFormer.yaml --exp_name test_ckpt --test


#### centerPoint
# conda create -n centerpoint3 python=3.8 -y
# conda activate centerpoint3
# conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 cudatoolkit-dev=11.3 -c pytorch -c conda-forge
# pip install spconv
# pip install spconv-cu113
## run other setup scripts provided by the repository
## follow the instructions in the repo and run scripts like this
# python tools/create_data.py nuscenes_data_prep --root_path=data/nuScenesAll --version="v1.0-trainval" --nsweeps=1
# python tools/create_data.py nuscenes_data_prep --root_path=data/nuScenes --version="v1.0-mini" --nsweeps=1
# export PYTHONPATH="${PYTHONPATH}:/home/shinghei/lidar_generation/nuscenes-devkit/python-sdk"
# python ./tools/dist_test.py configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_flip.py --work_dir configs/nusc/voxelnet/ --checkpoint ./epoch_20.pth --speed_test 



