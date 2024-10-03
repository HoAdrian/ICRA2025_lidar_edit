import copy
import os
import numpy as np
import argparse

import sys
sys.path.append("./")
sys.path.append("./datasets")
sys.path.append("./models")
from datasets.data_utils import *
from datasets.dataset import Voxelizer, PolarDataset, collate_fn_BEV
from datasets.dataset_nuscenes import Nuscenes
import torch
import configs.nuscenes_config as config

from models.vqvae_transformers import VQVAETrans, MaskGIT, voxels2points
import open3d

import pickle
from evaluation_utils import *



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--trainval_data_path', type=str, help="path to training and validation examples")
    parser.add_argument('--data_version', type=str, help="versioin of the dataset e.g. nuscenes")
    args = parser.parse_args()

    device = torch.device(config.device)
    print("--- device: ", device)

    ######### use spherical or polar coordinates
    mode = config.mode
    mode = "polar"
    use_z = False
    if mode=="spherical":
        use_z=True

    is_test=True # how to generate mask of the object: if is_test==True, mask all possible objects, otherwise pick one object randomly and rotate it to a free interval and generate the occlusion mask

    if mode=="spherical":
        max_bound=[50.635955604688654, 2*np.pi, np.deg2rad(120+40/31/2)] # 120 degree
        min_bound=[0, 0, np.deg2rad(80-40/31/2)] # 80 degree
    else:
        max_bound=[50, 2*np.pi, 3]
        min_bound=[0, 0, -5]
    grid_size = [512, 512, 32] # fine discretization
    #grid_size = [60, 60, 32] # coarse discretization
    print(f"GRID SIZE: {grid_size}")
    voxelizer = Voxelizer(grid_size=grid_size, max_bound=max_bound, min_bound=min_bound)
    train_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'train', filter_valid_scene=True, vis=True, voxelizer=voxelizer, is_test=is_test, use_z=use_z, mode=mode)
    val_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'val', filter_valid_scene=True, vis =True, voxelizer=voxelizer, is_test=is_test, use_z=use_z, mode=mode)
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=True)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=True)

    assert(train_dataset.use_random_mask==False)
    assert(val_dataset.use_random_mask==False)
   
    print("+++ num train: ", len(train_dataset))
    print("+++ num val: ", len(val_dataset))
    
    batch_size = 1
    train_dataset_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                    batch_size = batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = True,
                                                    num_workers=1)
    val_dataset_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                    batch_size = batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = True,
                                                    num_workers=1)

    dataset = val_dataset
    samples = np.arange(len(dataset))

    torch.cuda.empty_cache()
    seed = 100
    torch.manual_seed(seed)
    np.random.seed(seed)

    for k in samples:
        print(f"@@@@@@@@@ at sample {k}/{len(dataset)}")
        #k = 38 #73 #42
        #k = 1
        k = 31 #21 #40 #31 #66
        data_tuple = collate_fn_BEV([dataset.__getitem__(k)])
        has, no, voxel_label, BEV_label = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label.to(device) #(B, H, W, in_chans)
        BEV_mask = BEV_label.to(device) #(B,H,W)

        voxels_occupancy_has = voxels_occupancy_has.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        voxels_occupancy_no = voxels_occupancy_no.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)

        original_points = voxels2points(voxelizer, voxels_occupancy_has, mode=mode)[0]

        #### make the discretization coarser
        coarse_voxelizer = Voxelizer(grid_size=[100,100,32], max_bound=max_bound, min_bound=min_bound)
        _, _, _, coarse_GT_occupancy = coarse_voxelizer.voxelize(cart2polar(original_points, mode=mode), return_point_info=False) #(H,W,in_chans)
        coarse_points = voxels2points(coarse_voxelizer, torch.tensor(coarse_GT_occupancy).permute(2,0,1).unsqueeze(0), mode=mode)[0]

        vox_size = 1
        xlim = [-80, 80]
        ylim = [-80, 80]
        voxelizer.vis_BEV_binary_voxel(voxels_occupancy_has[0].permute(1,2,0), points_xyz=None, intensity=None, vis=False, path=f"./test_figures", name=f"fine_{k}_occupancy", vox_size=vox_size, xlim=xlim, ylim=ylim, only_points=False, mode=mode)
        coarse_voxelizer.vis_BEV_binary_voxel(torch.tensor(coarse_GT_occupancy), points_xyz=None, intensity=None, vis=False, path=f"./test_figures", name=f"coarse_{k}_occupancy?", vox_size=vox_size, xlim=xlim, ylim=ylim, only_points=False, mode=mode)
        
        print("visualizing fine points... ")
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(original_points[:,:3]))
        pcd_colors = np.tile(np.array([[0,0,1]]), (len(original_points), 1))
        pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
        mat = open3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.point_size = 2.0
        open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], show_skybox=False)

        print("visualizing coarse points... ")
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(coarse_points[:,:3]))
        pcd_colors = np.tile(np.array([[0,0,1]]), (len(coarse_points), 1))
        pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
        mat = open3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.point_size = 2.0
        open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], show_skybox=False)