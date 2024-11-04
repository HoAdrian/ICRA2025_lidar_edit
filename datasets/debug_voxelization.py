import copy
import os
import numpy as np
import argparse

import sys
os.system(f"pwd")

sys.path.append("./")
sys.path.append("./datasets")
sys.path.append("./models")


from datasets.data_utils import *
from datasets.dataset import Voxelizer, PolarDataset, collate_fn_BEV_intensity
from datasets.dataset_nuscenes import Nuscenes
import torch
import configs.nuscenes_config as config

from models.vqvae_transformers import VQVAETrans, voxels2points
import open3d

from scipy.spatial import KDTree


'''
TRY cartesian voxelization and see if there is a loss of number of points
'''

def voxels2points_cart(self, voxels):
    '''
    - voxelizer: Voxelizer object from dataset.py
    - voxels: binary, shape (B, in_chans, H, W), assume in_chans corresponds to z, H and W corresponds to r and theta. 

    return: 
    - list of numpy array of point cloud in cartesian coordinate (each may have different number of points)
    '''
    B, _, _, _ = voxels.shape
    point_clouds = []
    for b in range(B):
        voxels_b = voxels[b]
        voxels_b = voxels_b.permute(1,2,0) # (H, W, in_chans)
        non_zero_indices = torch.nonzero(voxels_b).float() #(num_non_zero_voxel, 3)
        ## convert non zero voxels to points
        intervals = torch.tensor(self.intervals).to(voxels.device).unsqueeze(0) #(1,3)
        min_bound = torch.tensor(self.min_bound).to(voxels.device).unsqueeze(0) #(1,3)
        xyz_pol = ((non_zero_indices[:, :]+0.5) * intervals) + min_bound # use voxel center coordinate
        xyz_pol = xyz_pol.cpu().detach().numpy()
        xyz = xyz_pol

        point_clouds.append(xyz)

    return point_clouds #a list of (num_non_zero_voxel_of_the_batch, 3)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--trainval_data_path', type=str, default="data/nuscenes/v1.0-mini", help="path to training and validation examples")
    parser.add_argument('--data_version', type=str, default="v1.0-mini", help="versioin of the dataset e.g. nuscenes")
    parser.add_argument('--figures_path', type=str, default="./train_intensity_models", help="path to save the figures")
    args = parser.parse_args()

    config.device = "cpu"
    device = torch.device(config.device)
    print("--- device: ", device)

    ######### use spherical or polar coordinates
    mode = config.mode
    use_z = False
    if mode=="spherical":
        use_z=True

    train_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'train', filter_valid_scene=False, use_z=use_z, mode=mode)
    val_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'val', filter_valid_scene=False, use_z=use_z, mode=mode)

    voxelizer = Voxelizer(grid_size=config.grid_size, max_bound=config.max_bound, min_bound=config.min_bound)
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, use_intensity_grid=True)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, use_intensity_grid=True)
   
    print("+++ num train: ", len(train_dataset))
    print("+++ num val: ", len(val_dataset))

    grid_size = [512, 512, 32]
    max_bound=[50, 50, 3]
    min_bound=[-50, -50, -5]
    cart_voxelizer = Voxelizer(grid_size=grid_size, max_bound=max_bound, min_bound=min_bound)

    num_vis = 81
    dataset = val_dataset
    samples = np.random.choice(len(dataset), num_vis) #np.arange(len(dataset))#np.random.choice(len(dataset), num_vis)
    l2_errors = []
    for k in samples:
        #k = 31 #56 #66 #31 #44 #56 #31 #56 #31 #66, 31
        print(f"++++++++++|||||| sample index: {k}")
        data_tuple = collate_fn_BEV_intensity([dataset.__getitem__(k)])
        # has, no, voxel_label, BEV_label, intensity_grid = data_tuple
        # grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        # grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        # voxels_mask = voxel_label.to(device) #(1, #r, #theta, #z)
        # BEV_mask = BEV_label.to(device) #(1, #r, #theta)

        points = dataset.points_xyz[:,:4]
        mask = cart_voxelizer.filter_in_bound_points(points)
        points = points[mask][:,:3]
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = voxelizer.voxelize(points, return_point_info=False)

        ###### now we have the original points and the corresponding occupancy grid in cartesian coordinate

        orig_pt_num = len(points)
        voxels_occupancy_has = torch.tensor(voxels_occupancy_has).unsqueeze(0)
        orig_vox_num = torch.sum(voxels_occupancy_has)
        print(f"====== voxelize points: orig vox num: {orig_vox_num} VS num points: {orig_pt_num} vs num unique points: {len(np.unique(points, axis=0))}")
        
        rec_points = voxels2points_cart(voxelizer, voxels_occupancy_has.permute(0,3,1,2))[0]
        print(f"====== reconstruct from voxelization num rec points: {len(rec_points)} vs num unique rec points {len(np.unique(rec_points, axis=0))}")
        
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has_rec = voxelizer.voxelize(rec_points, return_point_info=False)
        new_vox_num = np.sum(voxels_occupancy_has_rec)
        print(f"====re voxelize: orig voxel num: {orig_vox_num} VS new_voxel num: {new_vox_num}")
        assert(orig_vox_num==new_vox_num)