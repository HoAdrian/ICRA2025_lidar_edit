#!/bin/bash

# run conda info | grep -i 'base environment' to check the path to conda

conda_path = "/home/shinghei/miniconda3"
source "$conda_path/etc/profile.d/conda.sh"

# assuming I am using CUDA of version 11.8
conda create -n "polar" python=3.8
conda activate polar
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install nuscenes-devkit
pip install -r requirements_polar.txt
conda install pytorch-scatter -c pyg
