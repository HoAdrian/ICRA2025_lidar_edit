import copy
import os
import numpy as np
import argparse

os.system(f"pwd")

import sys
sys.path.append("./")
sys.path.append("./datasets")
sys.path.append("./models")
from datasets.data_utils import *
from datasets.dataset import Voxelizer, PolarDataset, collate_fn_BEV
from datasets.dataset_nuscenes import Nuscenes
import torch
import configs.nuscenes_config as config

from models.vqvae_transformers import VQVAETrans
import pickle
        
'''
Get the average ratio of num_occupied_cells over num_cells
'''

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--trainval_data_path', type=str, help="path to training and validation examples")
    parser.add_argument('--data_version', type=str, help="versioin of the dataset e.g. nuscenes")
    args = parser.parse_args()

    device = torch.device(config.device)
    print("--- device: ", device)

    mode = config.mode
    use_z = False
    if mode=="spherical":
        use_z=True

    train_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'train', exhaustive=False, get_stat=True, filter_valid_scene=True, use_z=use_z, mode=mode)
    val_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'val', exhaustive=False, get_stat=True, filter_valid_scene=True, use_z=use_z, mode=mode)

    voxelizer = Voxelizer(grid_size=config.grid_size, max_bound=config.max_bound, min_bound=config.min_bound)
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, is_test=True)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, is_test=True)
       
    print("+++ original num train: ", len(train_pt_dataset))
    print("+++ original num val: ", len(val_pt_dataset))
   
    datasets = [val_dataset, train_dataset]
    for dataset in datasets:
        print("############# num data: ", len(dataset))
        samples = np.arange(len(dataset))
        for k in samples:
           dataset.__getitem__(k)

    print("*************** train stats ***************")
    print(f"num_occupied/num_grid mean: {train_dataset.occupancy_ratio_total/train_dataset.current_iter}")

    print("*************** val stats ***************")
    print(f"num_occupied/num_grid mean: {val_dataset.occupancy_ratio_total/val_dataset.current_iter}")

    print("*************** train stats ***************")
    print(f"minmax_r: {train_pt_dataset.minmax_r}")
    print(f"minmax_z: {train_pt_dataset.minmax_z}")
    print(f"min obj r diff: {train_pt_dataset.obj_min_r_diff}")
    print(f"min obj theta diff: {train_pt_dataset.obj_min_theta_diff}")

    print("*************** val stats ***************")
    print(f"minmax_r: {val_pt_dataset.minmax_r}")
    print(f"minmax_z: {val_pt_dataset.minmax_z}")
    print(f"min obj r diff: {val_pt_dataset.obj_min_r_diff}")
    print(f"min obj theta diff: {val_pt_dataset.obj_min_theta_diff}")

    print("+++ filtered num train: ", np.sum(train_pt_dataset.valid_scene_idxs))
    print("+++ filtered num val: ", np.sum(val_pt_dataset.valid_scene_idxs))


            