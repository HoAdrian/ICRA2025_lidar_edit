#!/bin/bash

################### trainval dataset
version="v1.0-trainval"
dataset_path="./data/nuscenes/v1.0-trainval"
# vqvae_path="./weights/polar/vqvae_trans_weights"
# maskgit_path="./weights/polar/maskgit_trans_weights"
# vqvae_epoch=60
# maskgit_epoch=30

vqvae_path="./weights/vqvae_trans_weights"
maskgit_path="./weights/maskgit_trans_weights"
vqvae_epoch=60
maskgit_epoch=44 #44 #36

################### mini dataset for prototyping
version="v1.0-mini"
dataset_path="./data/nuscenes/v1.0-mini"
# vqvae_path="./weights/vqvae_trans_weights"
# maskgit_path="./weights/maskgit_trans_weights_mini"
# vqvae_epoch=60
# maskgit_epoch=20

#################### commands

################ Before training, set the config (what coordinates (polar or spherical)), double check the path to use/save the model
echo "VERSION: $version"
echo "DATASET_PATH: $dataset_path"
#echo "++++ !!!! REMINDER: remember to change the path to the valid scene idxs in the configs/nuscenes_config.py for different dataset version before training mask git !!!!"

dense=1

#### collect Nuscenes' foreground objects and their properties (e.g. viewing angle, bounding box, allocentric angle)
#python actor_insertion/collect_foreground_objects.py --trainval_data_path=$dataset_path --data_version=$version --pc_path="./foreground_object_pointclouds" --visualize_dense=0 --save_as_pcd=1
#### visualize the completed point clouds
#python actor_insertion/collect_foreground_objects.py --trainval_data_path=$dataset_path --data_version=$version --pc_path="./foreground_object_pointclouds" --visualize_dense=1 --save_as_pcd=1
#### Insert them and visualize!!!
#python actor_insertion/insert_obj.py --trainval_data_path=$dataset_path --data_version=$version --pc_path="./foreground_object_pointclouds" --dense=$dense --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --maskgit_path="$maskgit_path/epoch_$maskgit_epoch"



###### Generate lidar point clouds dictionaries such that dict[lidar_sample_token] = [lidar_path, List of bounding boxes], where lidar_path is the path where the point cloud of shape (N,3) is saved
root="/home/shinghei/lidar_generation/tmp_generated_lidar_2"
#root="/home/shinghei/lidar_generation/OpenPCDet_minghan/data/nuscenes/${version}"

# split="train"
# python actor_insertion/generate_raw_dict.py --trainval_data_path=$dataset_path --data_version=$version --save_lidar_path="$root"/raw/"$version" --split=$split
# split="val"
# python actor_insertion/generate_raw_dict.py --trainval_data_path=$dataset_path --data_version=$version --save_lidar_path="$root"/raw/"$version" --split=$split
# split="train"
# python actor_insertion/generate_no_obj_dict.py --trainval_data_path=$dataset_path --data_version=$version --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --maskgit_path="$maskgit_path/epoch_$maskgit_epoch" --save_lidar_path="$root"/no_obj/"$version" --split=$split
# split="val"
# python actor_insertion/generate_no_obj_dict.py --trainval_data_path=$dataset_path --data_version=$version --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --maskgit_path="$maskgit_path/epoch_$maskgit_epoch" --save_lidar_path="$root"/no_obj/"$version" --split=$split
# split="train"
# python actor_insertion/generate_insert_obj_dict.py --trainval_data_path=$dataset_path --data_version=$version --pc_path="./foreground_object_pointclouds" --dense=$dense --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --maskgit_path="$maskgit_path/epoch_$maskgit_epoch" --save_lidar_path=$root/has_obj/"$version" --split=$split
# split="val"
# python actor_insertion/generate_insert_obj_dict.py --trainval_data_path=$dataset_path --data_version=$version --pc_path="./foreground_object_pointclouds" --dense=$dense --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --maskgit_path="$maskgit_path/epoch_$maskgit_epoch" --save_lidar_path=$root/has_obj/"$version" --split=$split
# split="val"
# python actor_insertion/generate_insert_obj_dict_baselines.py --trainval_data_path=$dataset_path --data_version=$version --pc_path="./foreground_object_pointclouds" --dense=$dense --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --maskgit_path="$maskgit_path/epoch_$maskgit_epoch" --save_lidar_path=$root --split=$split

# echo "HELLO NICE FIGS"
# split="val"
# python actor_insertion/get_nice_figures.py --trainval_data_path=$dataset_path --data_version=$version --pc_path="./foreground_object_pointclouds" --dense=$dense --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --maskgit_path="$maskgit_path/epoch_$maskgit_epoch" --save_lidar_path=$root --split=$split


split="val"
python actor_insertion/generate_insert_obj_dict_baselines.py --trainval_data_path=$dataset_path --data_version=$version --pc_path="./foreground_object_pointclouds" --dense=$dense --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --maskgit_path="$maskgit_path/epoch_$maskgit_epoch" --save_lidar_path=$root --split=$split

























##################################### only buses and trucks
# split="val"
# python actor_insertion/generate_bus_or_truck.py --trainval_data_path=$dataset_path --data_version=$version --pc_path="./foreground_object_pointclouds" --dense=$dense --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --maskgit_path="$maskgit_path/epoch_$maskgit_epoch" --save_lidar_path=$root --split=$split