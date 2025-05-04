
import copy
import os
import torch
import numpy as np
import argparse

import sys
sys.path.append("./")
sys.path.append("./datasets")
sys.path.append("./models")

from datasets.dataset import Voxelizer, PolarDataset, collate_fn_BEV
from datasets.dataset_nuscenes import NuscenesForeground
from models.vqvae_transformers import voxels2points

import configs.nuscenes_config as config

import open3d
import pickle



##################################################################################################################
######################### Collect nuscenes mini subset to host model ##############################################
##################################################################################################################


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--trainval_data_path', default="./data_mini", type=str, help="path to training and validation examples")
    parser.add_argument('--data_version', default="v1.0-mini", type=str, help="versioin of the dataset e.g. nuscenes")
    args = parser.parse_args()

    config.device = "cpu"
    device = torch.device(config.device)
    print("--- device: ", device)

    ######### use spherical or polar coordinates
    mode = config.mode
    use_z = False
    if mode=="spherical":
        use_z=True
    assert(mode=="spherical")

    ############ initialize voxelizer and NuScenes dataset
    vis = False
    voxelizer = Voxelizer(grid_size=config.grid_size, max_bound=config.max_bound, min_bound=config.min_bound)
    ## important: if you are just inserting the original object point cloud to debug, set filter_obj_point_with_seg to False
    train_pt_dataset = NuscenesForeground(args.trainval_data_path, version = args.data_version, split = 'train', vis=vis, mode=mode, voxelizer=voxelizer, filter_obj_point_with_seg=False, get_raw=True, any_scene=True)
    val_pt_dataset = NuscenesForeground(args.trainval_data_path, version = args.data_version, split = 'val', vis =vis, mode=mode, voxelizer=voxelizer, filter_obj_point_with_seg=False, get_raw=True, any_scene=True)
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=vis, insert=True)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=vis, insert=True)


    split_names = ["train", "val"]
    dataset_list = [train_dataset, val_dataset]

    num_data = len(val_dataset)
    
    for dataset_idx, dataset in enumerate(dataset_list):
        Data = []
        samples = np.arange(num_data) #len(dataset)

        for k in samples:
            print(f"NOTE: =============== currently at {split_names[dataset_idx]} Sample {k} =========================")
            return_from_data = dataset.__getitem__(k)
            if return_from_data is None:
                print(f"sample_idx: {k}")
                assert(1==0)

            data_tuple = collate_fn_BEV([return_from_data])
            has, no, voxel_label, BEV_label = data_tuple
            grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
            grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
            voxels_mask = voxel_label.to(device) #(B, H, W, in_chans)
            BEV_mask = BEV_label.to(device) #(B,H,W)

            ####### what we want
            dataset_obj_boxes_list = dataset.obj_properties[5] #5
            voxels_occupancy_has = voxels_occupancy_has.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
            scene_points_xyz = voxels2points(voxelizer, voxels_occupancy_has, mode=mode)[0]

            ###### for drivable surface visualization
            ground_points = dataset.point_cloud_dataset.ground_points[:,:3]
            other_background_points = dataset.point_cloud_dataset.other_background_points[:,:3]
            

            data = {"scene_points_xyz": scene_points_xyz, "bbox_list":dataset_obj_boxes_list, "ground_points":ground_points, "other_background_points":other_background_points}
            Data.append(data)

        save_path = f"./demo_data/{split_names[dataset_idx]}_mini_data.pickle"
        os.makedirs("./demo_data", exist_ok=True)
        with open(save_path, 'wb') as handle:
            pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)

