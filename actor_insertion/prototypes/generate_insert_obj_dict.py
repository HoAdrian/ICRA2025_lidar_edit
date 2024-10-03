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

from actor_insertion.insertion_utils import insert_vehicle_pc, sample_valid_insert_pos, insertion_vehicles_driver
from nuscenes.utils.geometry_utils import points_in_box

'''
Remove all existing foreground objects first and insert new foreground objects at random locations on driveable surface. 
Save a dictionary that maps from lidar sample tokens to its generated point cloud, bounding boxes and annotation tokens. 
'''

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--trainval_data_path', type=str, help="path to training and validation examples")
    parser.add_argument('--data_version', type=str, help="versioin of the dataset e.g. nuscenes")
    parser.add_argument('--pc_path', type=str, help="path to save the point clouds")
    parser.add_argument('--dense', type=int, help="whether to use dense point cloud from point cloud completion")
    parser.add_argument('--maskgit_path', type=str, help="path to the trained maskGIT's weights of a specific epoch")
    parser.add_argument('--vqvae_path', type=str, help="path to the trained vqvae's weights of a specific epoch")
    parser.add_argument('--save_lidar_path', type=str, help="path to save the dictionary and lidar point clouds, must be a full path")
    parser.add_argument('--split', type=str, help="train/val")
    parser.add_argument('--gen_method', type=str, help="options: ours, reconstruction, naive")
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
    train_pt_dataset = NuscenesForeground(args.trainval_data_path, version = args.data_version, split = 'train', vis=vis, mode=mode)
    val_pt_dataset = NuscenesForeground(args.trainval_data_path, version = args.data_version, split = 'val', vis =vis, mode=mode)
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=vis)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=vis)


    ############ load trained models ###########
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
    vqvae.load_state_dict(torch.load(args.vqvae_path)["model_state_dict"])

    maskgit_config = config.maskgit_trans_config
    mask_git = MaskGIT(vqvae=vqvae, voxelizer=voxelizer, hidden_dim=maskgit_config["hidden_dim"], depth=maskgit_config["depth"], num_heads=maskgit_config["num_heads"]).to(device)
    mask_git.load_state_dict(torch.load(args.maskgit_path)["model_state_dict"])
    
    #### get blank codes for blank code suppressing 
    gen_blank_code = True
    if gen_blank_code:
        print("generating blank code")
        mask_git.get_blank_code(path=".", name="blank_code", iter=100)

    print(f"--- num blank code: {len(mask_git.blank_code)}")
    print(f"--- blank code: {mask_git.blank_code}")

    
    if args.split=='train':
        dataset = train_dataset
    elif args.split=='val':
        dataset = val_dataset
    else:
        raise Exception("invalid split argument")
    samples = np.arange(len(dataset))
    num_objs_insert = 5
    available_objs = ["car", "truck"] #["car"]


    torch.cuda.empty_cache()
    seed = 100
    torch.manual_seed(seed)
    np.random.seed(seed)

    token2sample_dict = {}
    token2sample_dict_full_path = os.path.join(args.save_lidar_path, "token2sample.pickle")
    if os.path.exists(token2sample_dict_full_path):
        with open(token2sample_dict_full_path, 'rb') as handle:
            token2sample_dict = pickle.load(handle)
        print("@@@ token2sample dict loaded, continue populating it...")

    for k in samples:
        if k==2:
            break
        #k = 4
        print(f"NOTE: =============== currently at Sample {k} =========================")
        data_tuple = collate_fn_BEV([dataset.__getitem__(k)])
        has, no, voxel_label, BEV_label = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label.to(device) #(B, H, W, in_chans)
        BEV_mask = BEV_label.to(device) #(B,H,W)

        #### remove existing foreground objects first
        voxels_occupancy_has = voxels_occupancy_has.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        voxels_occupancy_no = voxels_occupancy_no.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        gen_binary_voxels = mask_git.iterative_generation_driver(voxels_mask, voxels_occupancy_no, BEV_mask, generation_iter=20, denoise_iter=0, mode=mode)
        
        voxels_occupancy_list = [voxels_occupancy_has[0].permute(1,2,0), gen_binary_voxels[0].permute(1,2,0)]
        points_list = [voxels2points(voxelizer, voxels_occupancy_has, mode=mode)[0], voxels2points(voxelizer, gen_binary_voxels, mode=mode)[0]]
        names_list = ["has (masked)", "generated (masked)"]
        if True:
            visualize_generated_pointclouds(voxelizer, voxels_occupancy_list, points_list, names_list, voxels_mask, image_path="./actor_insertion/vis_no_background", image_name=f"sample_{k}")

        # the new occupancy with foreground points removed
        voxels_occupancy_has = gen_binary_voxels.permute(0,2,3,1)

        #### Insert objects
        names = [available_objs[np.random.choice(len(available_objs), p=np.array([0.8, 0.2]))] for _ in range(num_objs_insert)]
        new_scene_points_xyz, new_bboxes, token2sample_dict = insertion_vehicles_driver(voxels_occupancy_has, names, dataset, voxelizer, token2sample_dict, args, os.path.join(args.save_lidar_path, "ours"), mode=mode)

        plot_obj_regions([], [], new_scene_points_xyz, 40, new_bboxes, xlim=[-40,40], ylim=[-40,40], title=f"generated{k}", path="./actor_insertion/vis_insert_actual", name=f"insert_sample_{k}", vis=False)


        

    





