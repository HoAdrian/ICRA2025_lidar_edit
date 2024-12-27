#!/bin/bash

################### trainval dataset
version="v1.0-trainval"
dataset_path="./data/nuscenes/nuscenes_symbolic"

## polar
# vqvae_path="./weights/polar/vqvae_trans_weights"
# maskgit_path="./weights/polar/maskgit_trans_weights"
# vqvae_epoch=60
# maskgit_epoch=30

## spherical
vqvae_path="./weights/vqvae_trans_weights"
maskgit_path="./weights/maskgit_trans_weights"
vqvae_epoch=60
maskgit_epoch=36

################### mini dataset for prototyping
# version="v1.0-mini"
# dataset_path="./data/nuscenes/v1.0-mini"
# vqvae_path="./weights/vqvae_trans_weights"
# maskgit_path="./weights/maskgit_trans_weights_mini"
# vqvae_epoch=60
# maskgit_epoch=20

#################### commands


echo "VERSION: $version"
echo "DATASET_PATH: $dataset_path"

###### ignore these 
#python evaluation/visualize_discretization.py --trainval_data_path=$dataset_path --data_version=$version
#python train_transformer_models/test_maskgit_transformers.py --trainval_data_path=$dataset_path --data_version=$version --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --maskgit_path="$maskgit_path/epoch_$maskgit_epoch" --figures_path="./figures/maskgit_trans" --blank_code_path="." --blank_code_name="blank_code" --gen_blank_code=True
#python evaluation/chamfer_evaluate.py --trainval_data_path=$dataset_path --data_version=$version --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --maskgit_path="$maskgit_path/epoch_$maskgit_epoch" --figures_path="./figures/maskgit_trans" --blank_code_path="." --blank_code_name="blank_code" --gen_blank_code=True
# python evaluation/histogram_evaluate.py --trainval_data_path=$dataset_path --data_version=$version --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --maskgit_path="$maskgit_path/epoch_$maskgit_epoch" --figures_path="./figures/maskgit_trans" --blank_code_path="." --blank_code_name="blank_code" --gen_blank_code=True
#python evaluation/histogram_evaluate_allocentric.py --trainval_data_path=$dataset_path --data_version=$version --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --maskgit_path="$maskgit_path/epoch_$maskgit_epoch" --figures_path="./figures/maskgit_trans" --blank_code_path="." --blank_code_name="blank_code" --gen_blank_code=True
#python evaluation/range_evaluate_allocentric_compare.py --trainval_data_path=$dataset_path --data_version=$version --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --maskgit_path="$maskgit_path/epoch_$maskgit_epoch" --figures_path="./figures/maskgit_trans" --blank_code_path="." --blank_code_name="blank_code" --gen_blank_code=True




python evaluation/histogram_evaluate_allocentric_compare.py --trainval_data_path=$dataset_path --data_version=$version --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --maskgit_path="$maskgit_path/epoch_$maskgit_epoch" --figures_path="./figures/maskgit_trans" --blank_code_path="." --blank_code_name="blank_code" --gen_blank_code=True
python evaluation/save_pc_for_feat_diff.py --trainval_data_path=$dataset_path --data_version=$version --vqvae_path="$vqvae_path/epoch_$vqvae_epoch" --maskgit_path="$maskgit_path/epoch_$maskgit_epoch" --figures_path="./figures/maskgit_trans" --blank_code_path="." --blank_code_name="blank_code" --gen_blank_code=True
