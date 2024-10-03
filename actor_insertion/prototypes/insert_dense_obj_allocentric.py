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
# from datasets.data_utils_nuscenes import rotation_method
from datasets.dataset_nuscenes import Nuscenes, NuscenesForeground, vehicle_names
import torch
import configs.nuscenes_config as config

from models.vqvae_transformers import VQVAETrans, voxels2points
import open3d
import pickle


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--trainval_data_path', type=str, help="path to training and validation examples")
    parser.add_argument('--data_version', type=str, help="versioin of the dataset e.g. nuscenes")
    parser.add_argument('--pc_path', type=str, help="path to save the point clouds")
    args = parser.parse_args()

    config.device = "cpu"
    device = torch.device(config.device)
    print("--- device: ", device)

    ######### use spherical or polar coordinates
    mode = config.mode
    use_z = False
    if mode=="spherical":
        use_z=True

    voxelizer = Voxelizer(grid_size=config.grid_size, max_bound=config.max_bound, min_bound=config.min_bound)
    train_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'train', filter_valid_scene=True, vis=True, voxelizer=voxelizer, is_test=True, use_z=use_z, mode=mode)
    val_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'val', filter_valid_scene=True, vis =True, voxelizer=voxelizer, is_test=True, use_z=use_z, mode=mode)
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=True)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=True)

    ### TODO: get nearest ground height by segmentation
    dataset = train_dataset
    samples = np.arange(len(dataset))
    name = "car"
    insert_xyz_pos = np.array([1,-10, -1]) # the z coordinate is a dummy, 10,10,1, [-5, 30,-1]
    alpha = np.deg2rad(45) # radian 120
    for k in samples:
        data_tuple = collate_fn_BEV([dataset.__getitem__(k)])
        has, no, voxel_label, BEV_label = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label.to(device) #(B, H, W, in_chans)
        BEV_mask = BEV_label.to(device) #(B,H,W)


        ### get the point cloud we saved with the nearest allocentric angle
        pc_path = os.path.join(args.pc_path, name)
        with open(os.path.join(args.pc_path, "allocentric.pickle"), 'rb') as handle:
            allocentric_dict = pickle.load(handle)
        allocentric_angles = np.array(allocentric_dict[name][0])
        viewing_angles = np.array(allocentric_dict[name][2])
        pc_filenames = np.array(allocentric_dict[name][1])

        ### get the pc of the same vehicle name that has the nearest allocentric angle
        nearest_idx = np.argmin(np.abs(allocentric_angles - alpha))
        nearest_viewing_angle = viewing_angles[nearest_idx]
        pc_filename = pc_filenames[nearest_idx]
        pc_full_path = os.path.join(pc_path, pc_filename)
        vehicle_pc = np.asarray(open3d.io.read_point_cloud(pc_full_path).points) #np.load(pc_full_path)
        #vehicle_pc = np.load(pc_full_path)

        current_viewing_angle = compute_viewing_angle(insert_xyz_pos[:2])
        print(f"---- nearest alpha in degree: {np.rad2deg(allocentric_angles[nearest_idx])}")
        print(f"---- nearest viewing in degree: {np.rad2deg(nearest_viewing_angle)}")
        print(f"+++++ current viewing angle in degree {np.rad2deg(current_viewing_angle)}")

        ##### We have assumed that the vehicle is centered at its centroid defined by its bounding box
        # vehicle_pc = vehicle_pc - np.mean(vehicle_pc, axis=0)

        #### Since alpha is preserved with varying gamma only (rotating the object), the vehicle_pc can be a rotated version of the one we want. 
        # ### We have to align the vehicle_pc orientation
        rotation_angle = -(current_viewing_angle - nearest_viewing_angle) # negative sign because gamma increases clockwise
        vehicle_pc = cart2polar(vehicle_pc, mode=mode)
        theta = vehicle_pc[:,1]
        theta = theta + rotation_angle
        theta[theta<0] += 2*np.pi
        theta = theta%(2*np.pi)
        vehicle_pc[:,1] = theta
        vehicle_pc = polar2cart(vehicle_pc, mode=mode)

        #### shift vehicle to the desired position
        vehicle_pc[:,:2] += insert_xyz_pos[:2]
        
        #### get the non-empty grid nearest to the vehicle in BEV, and set its lowest z-value as the lowest z-value of the vehicle
        nearest_polar_voxels_pos = voxelizer.get_nearest_occupied_BEV_voxel(voxels_occupancy_has[0].cpu().detach().numpy(), cart2polar(insert_xyz_pos[np.newaxis,:], mode=mode), mode=mode) #(M,3)
        nearest_cart_voxels_pos = polar2cart(nearest_polar_voxels_pos, mode=mode)
        nearest_min_z = np.min(nearest_cart_voxels_pos[:,2])
        vehicle_min_z = np.min(vehicle_pc[:,2])
        print("vehicle_min_z: ", vehicle_min_z)
        print("nearest min z: ", nearest_min_z)
        height_diff = nearest_min_z - vehicle_min_z
        print("height diff: ", height_diff)
        vehicle_pc[:,2] += height_diff
        insert_xyz_pos[2]=height_diff

        #### project to spherical grid, apply occlusion and convert back to point cloud
        polar_vehicle = cart2polar(vehicle_pc, mode=mode)     
        new_occupancy = voxelizer.voxelize_and_occlude(voxels_occupancy_has[0].cpu().detach().numpy(), polar_vehicle)
        print("done occlusion ...")
        new_scene_points_xyz = voxels2points(voxelizer, voxels=torch.tensor(new_occupancy).permute(2,0,1).unsqueeze(0), mode=mode)[0]
        print("done getting points...")

        # visualize inserted vehicle only
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(vehicle_pc))
        pcd_colors = np.tile(np.array([[0,0,1]]), (len(vehicle_pc), 1))
        pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
        car_vis_pos = open3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        car_vis_pos.paint_uniform_color([1,0,0])  
        car_vis_pos.translate(tuple(insert_xyz_pos))
        open3d.visualization.draw_geometries([pcd, car_vis_pos]) 

        

        # visualize the scene with an object added
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(new_scene_points_xyz))
        pcd_colors = np.tile(np.array([[0,0,1]]), (len(new_scene_points_xyz), 1))
        pcd.colors = open3d.utility.Vector3dVector(pcd_colors)

        car_vis_pos = open3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        car_vis_pos.paint_uniform_color([1,0,0])  
        car_vis_pos.translate(tuple(insert_xyz_pos))


        mat = open3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.point_size = 2.0
        open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}, {'name': 'car', 'geometry': car_vis_pos, 'material': mat}], show_skybox=False)

       





