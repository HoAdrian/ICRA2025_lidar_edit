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
from datasets.dataset_nuscenes import Nuscenes
import torch
import configs.nuscenes_config as config

from models.vqvae_transformers import VQVAETrans
import pickle
        
'''
Cache the examples that can be used to train transformer for object removal
'''

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--trainval_data_path', type=str, help="path to training and validation examples")
    parser.add_argument('--data_version', type=str, help="versioin of the dataset e.g. nuscenes")
    args = parser.parse_args()

    device = torch.device(config.device)
    print("--- device: ", device)

    train_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'train', exhaustive=True, get_stat=True)
    val_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'val', exhaustive=True, get_stat=True)

    #voxelizer = Voxelizer(grid_size=config.grid_size, max_bound=config.max_bound, min_bound=config.min_bound)

       
    print("+++ original num train: ", len(train_pt_dataset))
    print("+++ original num val: ", len(val_pt_dataset))

    ### only get a subset of valid examples
    train_valid_scene_idxs_path = config.train_valid_scene_idxs_path #os.path.join(".", "train_valid_scene_idxs.pickle")
    val_valid_scene_idxs_path = config.val_valid_scene_idxs_path#os.path.join(".", "val_valid_scene_idxs.pickle")

   
    datasets = [val_pt_dataset, train_pt_dataset]
    for dataset in datasets:
        print("############# num data: ", len(dataset))
        samples = np.arange(len(dataset))
        for k in samples:
           dataset.__getitem__(k)

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
            
            
            
    # cache the valid scene idxs so that we can speed up the tedious rejection sampling of the valid scenes
    assert(train_pt_dataset.valid_scene_idxs is not None)
    assert(val_pt_dataset.valid_scene_idxs is not None)

    print("num train valid: ", np.sum(train_pt_dataset.valid_scene_idxs))
    print("num val valid: ", np.sum(val_pt_dataset.valid_scene_idxs))

    with open(train_valid_scene_idxs_path, 'wb') as handle:
        pickle.dump(train_pt_dataset.valid_scene_idxs, handle, protocol=3)

    with open(val_valid_scene_idxs_path, 'wb') as handle:
        pickle.dump(val_pt_dataset.valid_scene_idxs, handle, protocol=3)

            