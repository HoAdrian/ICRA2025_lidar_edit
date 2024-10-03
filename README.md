# File structure
## models: 
	- encoder, decoder, mask git for transformers and point models respectively
	- quantizer
## datasets:
	- data_utils: general utilities
	- data_utils_<dataset>: utilities for that particular dataset
	- dataset: voxelizer and the dataset class shared by all types of datasets (e.g. nuscenes)
## train_transformer_models: 
	- codes for training and testing transformer models
## evaluation
	- evaluate performance of our method
## actor_insertion
	- insert foreground vehicles either by nearest neighbor allocentric angle lookup or by point cloud completion
## other_repo
	- AnchorFormer: clone the original repo, Some modified scripts are included in this folder. This folder also includes the completed point cloud using AnchorFormer on KITTI dataset. 
	- The modified scripts are used to output the completed point cloud of the KITTI dataset and to unnormalized the output point cloud (undo the dataset's transformation)
	- Here is the original repo: https://github.com/chenzhik/AnchorFormer and where I downloaded the dataset: https://github.com/yuxumin/PoinTr/blob/master/DATASET.md

# Installation
- run the ultralidar.sh script in the installation folder to install dependencies
- Follow the instruction of AnchorFormer for point cloud completion

## setting up nuscenes dataset folder structure
- The bash scripts in my repo have a variable called dataset_path, which is where I put the dataset. You can customize the path yourself.
- Inside that dataset_path, follow the intsruction here to set up nuscenes trainval dataset enabled with lidarseg: https://www.nuscenes.org/tutorials/nuscenes_lidarseg_panoptic_tutorial.html


# Running the code
 Run these commands in the bash scripts at the root directory of this repos
- removal.sh contains the commands for foreground object removal. 
- insertion.sh contains commands for foreground object insertion.
- evaluation.sh contains commands for evaluating the performance of our method
