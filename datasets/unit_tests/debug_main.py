import copy
import os
import numpy as np

import sys
sys.path.append("../")
sys.path.append("../../models")
from data_utils import *
from dataset import Voxelizer, PolarDataset, collate_fn_BEV
from dataset_nuscenes import Nuscenes
import torch
import open3d
from vqvae_transformers import voxels2points



'''
example usage of dataset
'''

if __name__=="__main__":

    mode = "spherical"
    #mode = "polar"
    use_z = False
    if mode=="spherical":
        use_z=True

    if mode=="polar":
        print("config choosing polar mode")
        max_bound=[50, 2*np.pi, 3]
        min_bound=[0, 0, -5]
    elif mode=="spherical":
        print("config choosing spherical mode")
        max_bound=[50.635955604688654, 2*np.pi, np.deg2rad(120+40/31/2)] # 140 degree
        min_bound=[0, 0, np.deg2rad(80-40/31/2)] # 79 degree
        # max_bound=[50.635955604688654, 2*np.pi, np.deg2rad(180)] 
        # min_bound=[0, 0, 0] 
        # max_bound=[50.635955604688654, 2*np.pi, np.deg2rad(125)] # 140 degree
        # min_bound=[0, 0, np.deg2rad(75)] # 79 degree
    else:
        raise Exception("INVALID MODE")
    
    grid_size = [512, 512, 32]
    voxelizer = Voxelizer(grid_size=grid_size, max_bound=max_bound, min_bound=min_bound)
    train_pt_dataset = Nuscenes('../../data/nuscenes/v1.0-mini', version = 'v1.0-mini', split = 'val', filter_valid_scene=True, vis=True, voxelizer=voxelizer, use_z=use_z, mode=mode)
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, rotate_aug=False, flip_aug=False, use_voxel_random_mask=False, vis=True, is_test=True) 
   
    print("num train: ", len(train_dataset))

    for k in [1]:
        k = 66 #31 #66, 31
        data_tuple = collate_fn_BEV([train_dataset.__getitem__(k)])
        # remember to put tensors to the correct device
        has, no, voxel_label, BEV_label = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label
        BEV_mask = BEV_label

        print(f"grid_ind: {type(grid_ind_has)}")
        print(f"voxels_occupancy_has: {voxels_occupancy_has.shape}")
        print(f"voxel_label: {voxel_label.shape}")
        print(f"BEV_label: {BEV_mask.shape}")

        xlim, ylim = [-80, 80], [-80, 80]
        voxelizer.vis_BEV_binary_voxel(voxels_occupancy_has[0], points_xyz=None, intensity=None, vis=False, path="./test_figures", name="test_grid_occupancy", vox_size=1, xlim=xlim, ylim=ylim, only_points=False, mode=mode)
        voxelizer.vis_BEV_binary_voxel(voxels_mask[0], points_xyz=None, intensity=None, vis=False, path="./test_figures", name="test_grid_mask", vox_size=1, xlim=xlim, ylim=ylim, only_points=False, mode=mode)

        print("1voxel_pos: ", voxelizer.compute_voxel_position(grid_idx=[1,0,0]))

        print("2voxel_pos: ", voxelizer.compute_voxel_position(grid_idx=[1,0,30]))


        voxels = voxels_occupancy_has
        
        # # visualize point cloud
        #point_cloud = voxels2points(voxelizer, voxels.permute(0,3,1,2), mode=mode)[0]

        original_points = train_dataset.points_xyz #point_cloud
        print(original_points.shape)

        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(np.array(original_points[:,:3]))
        # pcd_colors = np.tile(np.array([[0,0,1]]), (len(original_points), 1))
        # pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
        # mat = open3d.visualization.rendering.MaterialRecord()
        # mat.shader = 'defaultUnlit'
        # mat.point_size = 2.0
        # open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], show_skybox=False)

        for i in range(32):
            print(f"++++ visualizings different rays {i}")
            voxelizer.vis_BEV_binary_voxel(voxels[0][:,:,i:i+1], points_xyz=None, intensity=None, vis=False, path="./test_figures", name=f"ray {i}", vox_size=1, xlim=xlim, ylim=ylim, only_points=False, mode=mode)


        break