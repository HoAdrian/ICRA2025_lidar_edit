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
from datasets.dataset import Voxelizer, PolarDataset, collate_fn_range_intensity
from datasets.dataset_nuscenes import Nuscenes
import torch
import configs.nuscenes_config as config

from models.unet import UNet
import open3d


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--trainval_data_path', type=str, help="path to training and validation examples")
    parser.add_argument('--data_version', type=str, help="versioin of the dataset e.g. nuscenes")
    parser.add_argument('--model_path', type=str, help="path to the trained vqvae's weights of a specific epoch")
    parser.add_argument('--figures_path', type=str, help="path to save the figures")
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
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, use_range_proj=True)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, use_range_proj=True)
   
    print("+++ num train: ", len(train_dataset))
    print("+++ num val: ", len(val_dataset))
    
    vqvae_config = config.vqvae_trans_config

    window_size=vqvae_config["window_size"]
    patch_size=vqvae_config["patch_size"]
    patch_embed_dim = vqvae_config["patch_embed_dim"]
    num_heads = vqvae_config["num_heads"]
    depth = vqvae_config["depth"]
    codebook_dim = vqvae_config["codebook_dim"]
    num_code = vqvae_config["num_code"]
    beta = vqvae_config["beta"]
    dead_limit = vqvae_config["dead_limit"]


    model = UNet(in_channels=1, out_channels=1).to(device)

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    num_vis = 100
    dataset = val_dataset
    samples = np.random.choice(len(dataset), num_vis) #np.arange(len(dataset))#np.random.choice(len(dataset), num_vis)
    l2_errors = []
    for k in samples:
        k = 66 #31 #44 #56 #31 #56 #31 #66, 31
        print(f"++++++++++|||||| sample index: {k}")
        data_tuple = collate_fn_range_intensity([dataset.__getitem__(k)])
        has, no, voxel_label, BEV_label, range_intensity_data = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label
        BEV_mask = BEV_label.to(device)
        range_image, intensity_image = range_intensity_data
        range_image = range_image.to(device)
        intensity_image = intensity_image.to(device)

        range_image = range_image.unsqueeze(1).float()
        intensity_image = intensity_image.unsqueeze(1).float()
        pred_intensity_image = model(range_image)
        loss = torch.mean((pred_intensity_image - intensity_image)**2)
        pred_intensity_image = torch.clip(pred_intensity_image, min=0.0, max=255.0)

        original_points = voxelizer.range2pc(range_image.squeeze().detach().cpu().numpy(), intensity_image.squeeze().detach().cpu().numpy())
        pred_points = voxelizer.range2pc(range_image.squeeze().detach().cpu().numpy(), pred_intensity_image.squeeze().detach().cpu().numpy())

        l2_error = torch.sqrt(loss)
        l2_errors.append(l2_error)
        print(f"l2_error: {l2_error}")
        print(f"max GT intensity: {np.max(original_points[:,3])}, min GT intensity:{np.min(original_points[:,3])}, mean GT intensity: {np.mean(original_points[:,3])}")
        print(f"max pred intensity: {np.max(pred_points[:,3])}, min pred intensity:{np.min(pred_points[:,3])}, mean pred intensity: {np.mean(pred_points[:,3])}")



        ### visualize original point cloud
        print(f"++++ visualizing ORIGINAL POINT CLOUD")
        print(original_points.shape)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(original_points[:,:3]))
        pcd_colors = np.tile(np.array([[0,1,0]]), (len(original_points), 1))*original_points[:,3:4]/255.0#np.max(original_points[:,3:4])
        pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
        mat = open3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.point_size = 3.0
        open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], show_skybox=False)
        #####################################
        
        print(f"++++ visualizing predicted POINT CLOUD")
        print(pred_points.shape)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(pred_points[:,:3]))
        pcd_colors = np.tile(np.array([[0,1,0]]), (len(pred_points), 1))*pred_points[:,3:4]/255.0#np.max(original_points[:,3:4])
        pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
        mat = open3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.point_size = 3.0
        open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], show_skybox=False)

        
    l2_erros = np.array(l2_errors)
    avg_l2_error = np.mean(l2_errors)
    max_error = np.max(l2_errors)
    min_error = np.min(l2_errors)
    std = np.std(l2_errors)

    # print(f"l2 errors: {l2_errors}")
    print(f"avg_l2_erros: ", avg_l2_error)
    print(f"max_l2_erros: ", max_error)
    print(f"min_l2_erros: ", min_error)
    print(f"std: ", std)