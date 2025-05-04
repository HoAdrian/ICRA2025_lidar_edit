#!/bin/bash

################### trainval dataset
version="v1.0-trainval"
dataset_path="./data/nuscenes/v1.0-trainval"
# dataset_path="./data/nuscenes"
# vqvae_path="./weights/polar/vqvae_trans_weights"
# maskgit_path="./weights/polar/maskgit_trans_weights"
# vqvae_epoch=60
# maskgit_epoch=30

vqvae_path="../weights/vqvae_trans_weights"
maskgit_path="../weights/maskgit_trans_weights"
vqvae_epoch=60
maskgit_epoch=44 #44 #36

################### mini dataset for prototyping
version="v1.0-mini"
dataset_path="./data_mini"

#################### commands

echo "VERSION: $version"
echo "DATASET_PATH: $dataset_path"


# python demo_app/collect_nuscenes_mini_subset.py --trainval_data_path=$dataset_path --data_version=$version
# python demo_app/push_dataset_HF.py
# python demo_app/push_models_checkpoints_HF.py

python demo_app/demo_app_version_2.py --trainval_data_path=$dataset_path --data_version=$version --full_obj_pc_path="./foreground_object_pointclouds" --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --maskgit_path="$maskgit_path/epoch_$maskgit_epoch"
# python app.py