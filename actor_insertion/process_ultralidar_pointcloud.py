import open3d
import copy
import os
import numpy as np
import argparse

os.system(f"pwd")

import sys
sys.path.append("./")
sys.path.append("./datasets")
sys.path.append("./models")
sys.path.append("./actor_insertion")
from datasets.data_utils import *
from datasets.dataset import Voxelizer
import torch
import configs.nuscenes_config as config

def process_ply_files(directory, voxelizer, num_pointclouds, save_lidar_path, save_lidar_path_voxelized):
    # List all .ply files in the specified directory
    ply_files = [f for f in os.listdir(directory) if f.endswith('.ply')]
    if not ply_files:
        print("No .ply files found in the directory.")
        return

    count = 0
    for idx, ply_file in enumerate(ply_files):
        if count==num_pointclouds:
            break
        file_path = os.path.join(directory, ply_file)
        print(f"##### processing : point cloud {file_path} [{idx+1}/{len(ply_files)}]")

        # Load the point cloud from the .ply file
        pcd = open3d.io.read_point_cloud(file_path)
        if pcd.is_empty():
            print(f"Warning: {ply_file} is empty or could not be read.")
            continue
        
        points_xyz = np.asarray(pcd.points)
        print(f"point cloud shape:{points_xyz.shape}")

        points_within_bound_mask = voxelizer.filter_in_bound_points(cart2polar(points_xyz, mode=config.mode))
        points_xyz_voxelized = points_xyz[points_within_bound_mask]
        points_polar = cart2polar(points_xyz_voxelized[:,:3], mode=config.mode)
        _, _, _, voxels_occupancy_has = voxelizer.voxelize(points_polar, return_point_info=False)
        voxels_occupancy_has = np.transpose(voxels_occupancy_has, (2,0,1))[np.newaxis,...] #(1,z,r,theta) or (1,in_chans,H,W)
        points_xyz_voxelized = voxelizer.voxels2points(torch.tensor(voxels_occupancy_has), mode=config.mode)[0]

        print(f"point cloud shape voxelized:{points_xyz_voxelized.shape}")

        # print("VISUALIZING voxelized Ultralidar POINT CLOUD HERE.......") 
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(points_xyz_voxelized)
        # pcd_colors = np.tile(np.array([[0,0,1]]), (len(points_xyz_voxelized), 1))
        # pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
        # mat = open3d.visualization.rendering.MaterialRecord()
        # mat.shader = 'defaultUnlit'
        # mat.point_size = 2.0
        # open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], show_skybox=False)

        # print("VISUALIZING original Ultralidar POINT CLOUD HERE.......") 
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(points_xyz)
        # pcd_colors = np.tile(np.array([[0,0,1]]), (len(points_xyz), 1))
        # pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
        # mat = open3d.visualization.rendering.MaterialRecord()
        # mat.shader = 'defaultUnlit'
        # mat.point_size = 2.0
        # open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], show_skybox=False)

        extras = np.zeros((len(points_xyz), 2))
        points_xyz = np.concatenate((points_xyz, extras), axis=1)
        assert(points_xyz.shape[-1]==5)

        extras = np.zeros((len(points_xyz_voxelized), 2))
        points_xyz_voxelized = np.concatenate((points_xyz_voxelized, extras), axis=1)
        assert(points_xyz_voxelized.shape[-1]==5)

        #### lidar original
        lidar_sample_token = ply_file
        pc_name = f'{lidar_sample_token}.bin'
        os.makedirs(os.path.join(save_lidar_path, "lidar_point_clouds"), exist_ok=True)
        lidar_full_path = os.path.join(save_lidar_path, "lidar_point_clouds", pc_name)
        points_xyz.astype(np.float32).tofile(lidar_full_path)

        #### lidar voxelized
        lidar_sample_token = ply_file
        pc_name = f'{lidar_sample_token}.bin'
        os.makedirs(os.path.join(save_lidar_path_voxelized, "lidar_point_clouds"), exist_ok=True)
        lidar_full_path = os.path.join(save_lidar_path_voxelized, "lidar_point_clouds", pc_name)
        points_xyz_voxelized.astype(np.float32).tofile(lidar_full_path)

        count+=1


if __name__ == '__main__':

    directory = "/home/shinghei/UltraLiDAR_nusc_waymo/generated_10000sample_ultralidar"
    version="v1.0-mini" #"v1.0-trainval"
    save_lidar_path = f"/home/shinghei/lidar_generation/OpenPCDet_minghan/data/nuscenes/{version}/ultralidar"
    save_lidar_path_voxelized = f"/home/shinghei/lidar_generation/OpenPCDet_minghan/data/nuscenes/{version}/ultralidar_voxelized"

    num_pointclouds=None
    voxelizer = Voxelizer(grid_size=config.grid_size, max_bound=config.max_bound, min_bound=config.min_bound)
    process_ply_files(directory, voxelizer, num_pointclouds, save_lidar_path, save_lidar_path_voxelized)
