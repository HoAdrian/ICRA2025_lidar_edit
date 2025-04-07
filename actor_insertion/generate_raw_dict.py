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
from datasets.dataset import Voxelizer, PolarDataset, collate_fn_BEV
# from datasets.data_utils_nuscenes import rotation_method
from datasets.dataset_nuscenes import Nuscenes, NuscenesForeground, vehicle_names, plot_obj_regions
import torch
import configs.nuscenes_config as config

from models.vqvae_transformers import VQVAETrans, MaskGIT, voxels2points
import open3d
import pickle

from actor_insertion.insertion_utils import insert_vehicle_pc
from nuscenes.utils.geometry_utils import points_in_box

'''
Generate and save a dictionary that maps from lidar sample token to the original lidar point cloud (or where it is saved) and its list of bounding boxes
'''

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--trainval_data_path', type=str, help="path to training and validation examples")
    parser.add_argument('--data_version', type=str, help="versioin of the dataset e.g. nuscenes")
    parser.add_argument('--save_lidar_path', type=str, help="path to save the dictionary and lidar point clouds, must be a full path")
    parser.add_argument('--split', type=str, help="train/val")
    args = parser.parse_args()

    #config.device = "cpu"
    device = torch.device(config.device)
    print("--- device: ", device)

    ######### use spherical or polar coordinates
    mode = config.mode
    use_z = False
    if mode=="spherical":
        use_z=True
    assert(mode=="spherical")

    vis = False
    voxelizer = Voxelizer(grid_size=config.grid_size, max_bound=config.max_bound, min_bound=config.min_bound)
    train_pt_dataset = NuscenesForeground(args.trainval_data_path, version = args.data_version, split = 'train', vis=vis, mode=mode, ignore_collect=False, any_scene=True, get_raw=True)
    val_pt_dataset = NuscenesForeground(args.trainval_data_path, version = args.data_version, split = 'val', vis=vis, mode=mode, ignore_collect=False, any_scene=True, get_raw=True)
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=vis)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=vis)

    if args.split=='train':
        dataset = train_dataset
    elif args.split=='val':
        dataset = val_dataset
    else:
        raise Exception("invalid split argument")
    samples = np.arange(len(dataset))

    torch.cuda.empty_cache()
    seed = 100
    torch.manual_seed(seed)
    np.random.seed(seed)

    token2sample_dict = {}
    args.save_lidar_path = os.path.join(args.save_lidar_path, "ours", args.data_version)

    token2sample_dict_full_path = os.path.join(args.save_lidar_path, "token2sample.pickle")
    if os.path.exists(token2sample_dict_full_path):
        with open(token2sample_dict_full_path, 'rb') as handle:
            token2sample_dict = pickle.load(handle)
        print("@@@ token2sample dict loaded, continue populating it...")

    #samples = [0,1]
    for sample_id, k in enumerate(samples):
        print(f"######## raw for testing: {args.data_version}_{args.split} sample ", sample_id+1, "/", len(samples))
        #k = 4
        data_tuple = collate_fn_BEV([dataset.__getitem__(k)])
        has, no, voxel_label, BEV_label = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label.to(device) #(B, H, W, in_chans)
        BEV_mask = BEV_label.to(device) #(B,H,W)

        #points_xyz_original = dataset.points_xyz
        voxels_occupancy_has = voxels_occupancy_has.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        points_xyz = voxelizer.voxels2points(voxels_occupancy_has)[0]
        extras = np.zeros((len(points_xyz), 2))
        points_xyz = np.concatenate((points_xyz, extras), axis=1)
        assert(points_xyz.shape[-1]==5)

        # print(f"###### len(points_voxelized): {len(points_xyz)}")
        # print(f"###### len(points_original): {len(points_xyz_original)}")
        # assert(1==0)

        lidar_sample_token = dataset.lidar_sample_token

        # original_box_idxs = [i for i, box in enumerate(dataset.obj_properties[5]) if (box.name in vehicle_names and vehicle_names[box.name] in {"bus", "car", "truck"})]
        # original_vehicle_boxes = [dataset.obj_properties[5][i] for i in original_box_idxs]
        # bounding_boxes = original_vehicle_boxes
        # names = [vehicle_names[box.name] for box in original_vehicle_boxes]
        bounding_boxes = dataset.obj_properties[5]
        #(obj_point_cloud_list, obj_name_list, points_3D, obj_allocentric_list, obj_centers_list, obj_boxes_list, obj_gamma_list, kitti_boxes_list, lidar_sample_token, boxes, obj_ann_token_list, sample_annotation_tokens, sample_records, obj_ann_info_list)
        new_obj_ann_token_list = dataset.obj_properties[10]
        sample_records = dataset.obj_properties[12]
        new_ann_info_list = dataset.obj_properties[13]


        # print("VISUALIZING original POINT CLOUD HERE.......") 
        # pcd = open3d.geometry.PointCloud()
        # print("dataset original points: ", dataset.points_xyz.shape)
        # print("dataset original points: ", type(dataset.points_xyz))
        # print("dataset original points: ", dataset.points_xyz.dtype)
        # pcd.points = open3d.utility.Vector3dVector(np.array(dataset.points_xyz[:,:3]))
        # pcd_colors = np.tile(np.array([[0,0,1]]), (len(dataset.points_xyz), 1))
        # pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
        
        # ground_pcd = open3d.geometry.PointCloud()
        # grd_points = dataset.point_cloud_dataset.ground_points[:,:3][:,:3] #points_list[0] #
        # ground_pcd.points = open3d.utility.Vector3dVector(grd_points)
        # ground_pcd_colors = np.tile(np.array([[0,1,0]]), (len(grd_points), 1))
        # ground_pcd.colors = open3d.utility.Vector3dVector(ground_pcd_colors)

        # lines = [[0, 1], [1, 2], [2, 3], [0, 3],
        # [4, 5], [5, 6], [6, 7], [4, 7],
        # [0, 4], [1, 5], [2, 6], [3, 7]]
        # visboxes = []
        # for box in bounding_boxes:
        #     line_set = open3d.geometry.LineSet()
        #     line_set.points = open3d.utility.Vector3dVector(box.corners().T)
        #     line_set.lines = open3d.utility.Vector2iVector(lines)
        #     if box.name in vehicle_names:
        #         if vehicle_names[box.name]=="car":
        #             colors = [[1, 0, 0] for _ in range(len(lines))]
        #         elif vehicle_names[box.name]=="truck":
        #             colors = [[0, 1, 0] for _ in range(len(lines))]
        #         elif vehicle_names[box.name]=="bus":
        #             colors = [[0, 0, 1] for _ in range(len(lines))]
        #         else:
        #             colors = [[0, 0, 0] for _ in range(len(lines))]
        #     else:
        #         colors = [[0, 1, 1] for _ in range(len(lines))]
        #     line_set.colors = open3d.utility.Vector3dVector(colors)
        #     visboxes.append(line_set)
        # open3d.visualization.draw_geometries([pcd, ground_pcd]+visboxes)

    

        pc_name = f'{args.split}_{lidar_sample_token}.bin'
        os.makedirs(os.path.join(args.save_lidar_path, "lidar_point_clouds"), exist_ok=True)
        lidar_full_path = os.path.join(args.save_lidar_path, "lidar_point_clouds", pc_name)
        #assert(not os.path.exists(lidar_full_path))
        assert(points_xyz is not None)
        #points_xyz.tofile(lidar_full_path)
        points_xyz.astype(np.float32).tofile(lidar_full_path)

        #token2sample_dict[lidar_sample_token] = (lidar_full_path, bounding_boxes)
        token2sample_dict[lidar_sample_token] = (lidar_full_path, bounding_boxes, new_obj_ann_token_list, sample_records, new_ann_info_list)

    
    with open(token2sample_dict_full_path, 'wb') as handle:
        pickle.dump(token2sample_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)




