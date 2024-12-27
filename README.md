# File structure
## models: 
	- encoder, decoder, mask git implemented with transformer architecture
	- quantizer
## configs:
	- nuscenes_config.py: configuration containing parameters for voxelization and transformer models' hyperparameters
## datasets:
	- data_utils.py: general utilities
	- data_utils_nuscenes.py: utilities for NuScenes dataset
	- dataset_nuscenes.py: contains datasets for training, inpainting evaluation, foreground object extraction for point cloud completion and foreground object insertion
	- dataset.py: voxelizer and a general dataset class shared by all types of datasets in dataset_nuscenes.py
## train_transformer_models: 
	- codes for training and testing transformer models
## evaluation
	- evaluate performance of our inpainting method using statistical metrics e.g. MMD. We can also save the in-painted backgrounds from different baselines. 
## actor_insertion
	- insert dense foreground vehicles for training (perturbed poses) and evaluation (inserted at original poses) of 3D detection models
## other_repo
	- AnchorFormer: clone the original repo, Some modified scripts are included in this folder. This folder also includes the completed point cloud using AnchorFormer on KITTI dataset. Some scripts are customized to run point cloud completion on our extracted vehicles from NuScenes. Here is the original repo: https://github.com/chenzhik/AnchorFormer and where I downloaded the dataset: https://github.com/yuxumin/PoinTr/blob/master/DATASET.md

	- OpenPCDet: 3D detection model's dataset creation, training and evaluation. We use VoxelNext trained on single-sweep lidar with intensity values (reflectance) set to zero. 

# Installation
- For run the ultralidar.sh script in the installation folder to install dependencies for our repo and AnchorFormer
- For 3D detection experiments, run mmdet3d.sh to install OpenPCDet's dependencies and go to OpenPCDet's repo to install their dependencies
- Follow the instruction of AnchorFormer for point cloud completion

## setting up nuscenes dataset folder structure
- The bash scripts in my repo have a variable called dataset_path, which is where I put the dataset. You can customize the path yourself.
- Inside that dataset_path, follow the intsruction here to set up nuscenes trainval dataset enabled with lidarseg: https://www.nuscenes.org/tutorials/nuscenes_lidarseg_panoptic_tutorial.html


# Running the code
 Run these commands in the bash scripts at the root directory of this repos
- removal.sh contains the commands for foreground object removal. 
- insertion.sh contains commands for foreground object insertion.
- evaluation.sh contains commands for evaluating the performance of our method
