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
vqvae_epoch=60 #12
maskgit_epoch=44 #36

################### mini dataset for prototyping
version="v1.0-mini"
dataset_path="./data/nuscenes/v1.0-mini"
# vqvae_path="./weights/vqvae_trans_weights"
# maskgit_path="./weights/maskgit_trans_weights_mini"
# vqvae_epoch=60
# maskgit_epoch=20

#################### commands

################ Before training, check dataset path, set the config (what coordinates (polar or spherical)), double check the path to use/save the model
echo "VERSION: $version"
echo "DATASET_PATH: $dataset_path"
# echo "++++ !!!! REMINDER: remember to change the path to the valid scene idxs in the configs/nuscenes_config.py for different dataset version before training mask git !!!!"
echo "++++ !!!! IMPORTANT : Before training, check dataset path, set the config (what coordinates (polar or spherical)), double check the path to use/save the model !!!!"


# python datasets/get_occupancy_stats.py --trainval_data_path=$dataset_path --data_version=$version
# python datasets/profiling.py --trainval_data_path=$dataset_path --data_version=$version
# python datasets/generate_valid_scene_idxs.py --trainval_data_path=$dataset_path --data_version=$version


#python train_transformer_models/train_vqvae_transformers.py --trainval_data_path=$dataset_path --data_version=$version --weight_path="$vqvae_path" --num_epochs=121 --resume_epoch=8
#python train_transformer_models/test_vqvae_transformers.py --trainval_data_path=$dataset_path --data_version=$version --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --figures_path="./figures/vqvae_trans" # epoch 80 for mini


#python train_transformer_models/train_maskgit_transformers.py --trainval_data_path=$dataset_path --data_version=$version --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --weight_path="$maskgit_path" --num_epochs=201 --resume_epoch=40
# python train_transformer_models/test_maskgit_transformers.py --trainval_data_path=$dataset_path --data_version=$version --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --maskgit_path="$maskgit_path/epoch_$maskgit_epoch" --figures_path="./figures/maskgit_trans" --blank_code_path="." --blank_code_name="blank_code" --gen_blank_code=True


# tensorboard --logdir="./figures/mask_git_trans/runs"


intensity_vqvae_path="./weights/intensity_vqvae_weights"
intensity_vqvae_epoch=12 #40 #12
echo "INTENSITY VQVAE PATH: ${intensity_vqvae_path}"
#python train_transformer_models/train_intensity_vqvae.py --trainval_data_path=$dataset_path --data_version=$version --weight_path="$intensity_vqvae_path" --num_epochs=121 --resume_epoch=44
python train_transformer_models/test_intensity_vqvae.py --trainval_data_path=$dataset_path --data_version=$version --vqvae_path="$intensity_vqvae_path/epoch_$intensity_vqvae_epoch" --figures_path="./figures/intensity_vqvae" # epoch 80 for mini
