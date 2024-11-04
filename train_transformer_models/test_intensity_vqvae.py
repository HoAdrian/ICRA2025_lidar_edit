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
from datasets.dataset import Voxelizer, PolarDataset, collate_fn_BEV_intensity
from datasets.dataset_nuscenes import Nuscenes
import torch
import configs.nuscenes_config as config

from models.vqvae_transformers import VQVAETrans, voxels2points
import open3d


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--trainval_data_path', type=str, help="path to training and validation examples")
    parser.add_argument('--data_version', type=str, help="versioin of the dataset e.g. nuscenes")
    parser.add_argument('--vqvae_path', type=str, help="path to the trained vqvae's weights of a specific epoch")
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
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, use_intensity_grid=True)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, use_intensity_grid=True)
   
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


    vqvae = VQVAETrans(
        img_size=voxelizer.grid_size[0:2],
        in_chans=voxelizer.grid_size[2],
        patch_size=patch_size,
        window_size=window_size,
        patch_embed_dim=patch_embed_dim,
        num_heads=num_heads,
        depth=depth,
        codebook_dim=codebook_dim,
        num_code=num_code,
        beta=beta,
        device=device,
        dead_limit=dead_limit
    ).to(device)

    checkpoint = torch.load(args.vqvae_path)
    vqvae.load_state_dict(checkpoint["model_state_dict"])

    num_vis = 100
    dataset = val_dataset
    samples = np.random.choice(len(dataset), num_vis) #np.arange(len(dataset))#np.random.choice(len(dataset), num_vis)
    l2_errors = []
    for k in samples:
        k = 66 #31 #44 #56 #31 #56 #31 #66, 31
        print(f"++++++++++|||||| sample index: {k}")
        data_tuple = collate_fn_BEV_intensity([dataset.__getitem__(k)])
        has, no, voxel_label, BEV_label, intensity_grid = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label.to(device) #(1, #r, #theta, #z)
        BEV_mask = BEV_label.to(device) #(1, #r, #theta)


        voxels_occupancy_has = voxels_occupancy_has.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        voxels_occupancy_no = voxels_occupancy_no.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        intensity_grid = intensity_grid.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        # loss_has, rec_x_has, rec_x_logit_has, perplexity_has, min_encodings_has, min_encoding_indices_has = vqvae.compute_intensity_loss(voxels_occupancy_has, intensity_grid)
        loss_has, rec_x_logit_has = vqvae.compute_intensity_loss(voxels_occupancy_has, intensity_grid)

        rec_x_logit_has = torch.clip(rec_x_logit_has, min=0.0, max=255.0)
        masking = torch.logical_and(voxels_occupancy_has!=0, intensity_grid!=0)
        avg_L2_error = torch.sqrt(torch.sum(((intensity_grid[masking] - rec_x_logit_has[masking]))**2)/torch.sum(masking))
        print(f"?????? pred: {rec_x_logit_has[masking][:100]}")
        print(f"?????? gt: {intensity_grid[masking][:100]}")
        print("########## L2 error: ", avg_L2_error)
        print("_______ avg pred vs gt intensity: ", torch.mean(rec_x_logit_has[masking]), torch.mean(intensity_grid[masking]))
        print("_______ maxmin pred intensity: ", torch.max(rec_x_logit_has[masking]), torch.min(rec_x_logit_has[masking]))
        l2_errors.append(avg_L2_error.item())

        ### visualize original point cloud
        print(f"++++ visualizing ORIGINAL POINT CLOUD")
        original_points = dataset.points_xyz
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
        
        

        points_xyz_has = voxels2points(voxelizer, voxels_occupancy_has, mode=mode)
        print(f"#### points_xyz_original: ", len(points_xyz_has[0]))

        voxels_occupancy_list = [voxels_occupancy_has[0].permute(1,2,0)]
        points_list = [points_xyz_has]
        names_list = ["voxelized original points with predicted intensity"]
        print(rec_x_logit_has.shape)
        rec_x_logit_has = rec_x_logit_has[0].permute(1,2,0).detach().cpu().numpy()
        intensity_grid = intensity_grid[0].permute(1,2,0).detach().cpu().numpy()


        for j, points in enumerate(points_list):
            print(f"++++ visualizing {names_list[j]}")
            points = points[0]

            # get the grid index of each point
            voxel_occupancy = voxels_occupancy_list[j]
            non_zero_indices = torch.nonzero(voxel_occupancy.detach().cpu(), as_tuple=True)
            voxel_mask_ = voxels_mask[0]
            point_intensity = np.zeros((len(points),))

            if True:
                non_zero_indices = torch.nonzero(torch.tensor(voxel_occupancy).detach().cpu(), as_tuple=True)
                point_intensity = rec_x_logit_has[non_zero_indices[0].numpy(), non_zero_indices[1].numpy(), non_zero_indices[2].numpy()] # get the intensity of the occupied voxels
                #point_intensity = intensity_grid[non_zero_indices[0].numpy(), non_zero_indices[1].numpy(), non_zero_indices[2].numpy()] # get the intensity of the occupied voxels
                print("**************** any points in mask region? ", (np.sum(point_intensity)))

            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(np.array(points))
            pcd_colors = np.tile(np.array([[0,1,0]]), (len(points), 1))*point_intensity[:,np.newaxis]/255.0
            pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
            #open3d.visualization.draw_geometries([pcd]) 

            mat = open3d.visualization.rendering.MaterialRecord()
            mat.shader = 'defaultUnlit'
            mat.point_size = 3.0

            open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], show_skybox=False)
                
            #voxelizer.vis_BEV_binary_voxel(BEV_mask[0].unsqueeze(-1), points_xyz=points, intensity=point_intensity, vis=False, path=f"{args.figures_path}", name=f"{k}_{names_list[j]}_points_masked", vox_size=vox_size, xlim=xlim, ylim=ylim, only_points=True)

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