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
from models.unet import UNet
import open3d
import pickle

from actor_insertion.insertion_utils import insertion_vehicles_driver, insertion_vehicles_driver_perturbed, copy_and_paste_method, save_reconstruct_data, count_vehicle_name_in_box_list
from nuscenes.utils.geometry_utils import points_in_box
import logging
import timeit
'''
Remove all existing foreground objects first and insert new foreground objects at random locations on driveable surface. 
Save a dictionary that maps from lidar sample tokens to its generated point cloud, bounding boxes and annotation tokens. 
'''

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--trainval_data_path', type=str, help="path to training and validation examples")
    parser.add_argument('--data_version', type=str, help="versioin of the dataset e.g. nuscenes")
    parser.add_argument('--maskgit_path', type=str, help="path to the trained maskGIT's weights of a specific epoch")
    parser.add_argument('--vqvae_path', type=str, help="path to the trained vqvae's weights of a specific epoch")
    parser.add_argument('--save_lidar_path', type=str, help="path to the root to save the dictionary and lidar point clouds, must be a full path")
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
    ## important: if you are just inserting the original object point cloud to debug, set filter_obj_point_with_seg to False
    train_pt_dataset = NuscenesForeground(args.trainval_data_path, version = args.data_version, split = 'train', vis=vis, mode=mode, voxelizer=voxelizer, filter_obj_point_with_seg=False, get_raw=True)
    val_pt_dataset = NuscenesForeground(args.trainval_data_path, version = args.data_version, split = 'val', vis =vis, mode=mode, voxelizer=voxelizer, filter_obj_point_with_seg=False, get_raw=True, any_scene=True, only_cars=False)
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=vis, insert=True)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=vis, insert=True)


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



    ############### TODO intensity unet

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

    assert(args.split=="val")
    
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
    
    # where to save the generated data
    method_names = ["ours"] #, "reconstruct", "ours", "naive"
    lidar_path_list = [os.path.join(args.save_lidar_path) for method_name in method_names]
    token2sample_dict_list = [{} for _ in method_names]
    
    for i in range(len(token2sample_dict_list)):
        token2sample_dict_full_path = os.path.join(lidar_path_list[i], "token2sample.pickle")
        if os.path.exists(token2sample_dict_full_path):
            with open(token2sample_dict_full_path, 'rb') as handle:
                token2sample_dict = pickle.load(handle)
                token2sample_dict_list[i] = token2sample_dict
            print("@@@ token2sample dict loaded, continue populating it...")
            print("........_____num data: ", len(token2sample_dict))
    
    assert(len(lidar_path_list)==len(token2sample_dict_list)==len(method_names))

    num_methods = len(method_names)

    assert(len(token2sample_dict_list)==1)
    
    
    logging.basicConfig(filename="insertion_message.log", 
                        format='%(asctime)s: %(levelname)s: %(message)s', 
                        level=logging.INFO) 
    
    start_time = timeit.default_timer()
    for k in samples:
        print(f"NOTE: =============== currently at Sample {k} =========================")
        return_from_data = dataset.__getitem__(k)
        if return_from_data is None and args.split=='val':
            print(f"sample_idx: {k} excluded")
            continue

        print(f"=====++++ current token2sample length: {len(token2sample_dict_list[0])}")

        count_orig_vehicle = count_vehicle_name_in_box_list(dataset.nonempty_boxes, vehicle_names_dict=vehicle_names)
        count_car, count_bus, count_truck = count_orig_vehicle["car"], count_orig_vehicle["bus"], count_orig_vehicle["truck"]
        logging.info("counting ...??? ", count_orig_vehicle)
        # if count_bus==0 or count_truck==0:
        # if count_truck==0:
        #     print("SKIP NO BUS and TRUCK SAMPLES... ")
        #     continue


        data_tuple = collate_fn_BEV([return_from_data])
        has, no, voxel_label, BEV_label = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label.to(device) #(B, H, W, in_chans)
        BEV_mask = BEV_label.to(device) #(B,H,W)

        

        #### remove existing foreground objects first
        voxels_occupancy_has = voxels_occupancy_has.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        voxels_occupancy_no = voxels_occupancy_no.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        
        gen_binary_voxels_list = []
        for method_idx in range(num_methods):
            if method_names[method_idx]=="ours":
                gen_binary_voxels_ours = mask_git.iterative_generation_driver(voxels_mask, voxels_occupancy_no, BEV_mask, generation_iter=20, denoise_iter=0, mode=mode)
                gen_binary_voxels_list.append(gen_binary_voxels_ours)
            elif method_names[method_idx]=="naive":
                gen_binary_voxels_naive = copy_and_paste_method(voxels_occupancy_has, dataset, voxelizer)
                gen_binary_voxels_list.append(gen_binary_voxels_naive)
            elif method_names[method_idx]=="reconstruct":
                _, vqvae_rec_x_has, _, _, _, _ = vqvae.compute_loss(voxels_occupancy_has)
                vqvae_rec_x_has[voxels_mask.permute(0,3,1,2)==0] = voxels_occupancy_has[voxels_mask.permute(0,3,1,2)==0]
                gen_binary_voxels_list.append(vqvae_rec_x_has)
            else:
                raise Exception("Invalid method idx")
        
        # voxels_occupancy_no_copy = torch.clone(voxels_occupancy_no)
        # voxels_occupancy_no_copy[voxels_mask.permute(0,3,1,2)==1] = 0
        # voxels_occupancy_list = [voxels_occupancy_has[0].permute(1,2,0), voxels_occupancy_no_copy[0].permute(1,2,0)] + [gen_voxels[0].permute(1,2,0) for gen_voxels in gen_binary_voxels_list]
        # points_list = [voxels2points(voxelizer, voxels_occupancy_has, mode=mode)[0], voxels2points(voxelizer, voxels_occupancy_no_copy, mode=mode)[0]] + [voxels2points(voxelizer, gen_voxels, mode=mode)[0] for gen_voxels in gen_binary_voxels_list]
        # names_list = ["original", "object removed"] + [f"{name}" for name in method_names]
        # if True:
        #     #visualize_generated_pointclouds(voxelizer, voxels_occupancy_list, points_list, names_list, voxels_mask, image_path="./actor_insertion/vis_no_background_baselines", image_name=f"{k}_sample")
        #     visualize_generated_pointclouds(voxelizer, voxels_occupancy_list, points_list, names_list, voxels_mask, image_path=None, image_name=None)

        
        ###### insert vehicles to the new occupancy grids with foreground points removed by different methods
        insert_vehicle_names = None

        for method_idx, gen_grid in enumerate(gen_binary_voxels_list):
            logging.info(f"============ sample {k}: inserting to {method_names[method_idx]} occupancy grid")

            inpainted_points  = voxels2points(voxelizer, gen_grid, mode=mode)[0]

            extras = np.zeros((len(inpainted_points), 2))
            new_points_xyz = np.concatenate((inpainted_points, extras), axis=1)

            lidar_sample_token = dataset.lidar_sample_token
            pc_name = f'{args.split}_{lidar_sample_token}.bin'
            os.makedirs(os.path.join(lidar_path_list[0], "lidar_point_clouds"), exist_ok=True)
            lidar_full_path = os.path.join(lidar_path_list[0], "lidar_point_clouds", pc_name)
            new_points_xyz.astype(np.float32).tofile(lidar_full_path)
           
            sample_records = dataset.obj_properties[12]

            ############## Save the data needed to build the new database
            token2sample_dict_list[0][lidar_sample_token] = (lidar_full_path, [],[], sample_records, [])
            
    end_time = timeit.default_timer()
    print("HEYYYO: time used for generating insertion dataset: ", end_time-start_time, "seconds")

    save_lidar_path = lidar_path_list[0]
    print(f"========= SAVING DATA TO {save_lidar_path}")
    token2sample_dict_full_path = os.path.join(save_lidar_path, "token2sample.pickle")
    with open(token2sample_dict_full_path, 'wb') as handle:
        pickle.dump(token2sample_dict_list[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    

    





