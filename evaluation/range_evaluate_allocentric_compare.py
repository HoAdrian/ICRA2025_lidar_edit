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
from datasets.dataset_nuscenes import NuscenesEval
import torch
import configs.nuscenes_config as config

from models.vqvae_transformers import VQVAETrans, MaskGIT, voxels2points
import open3d

import pickle
from evaluation_utils import *

from skimage.metrics import structural_similarity as ssim

import timeit

'''
evaluate performance of our foreground object removal pipeline by creating artificial bounding boxes at a fixed radius from the camera and at a fxied allocentric angle to create occlusion
using ssim
'''


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--trainval_data_path', type=str, help="path to training and validation examples")
    parser.add_argument('--data_version', type=str, help="versioin of the dataset e.g. nuscenes")
    parser.add_argument('--vqvae_path', type=str, help="path to the trained vqvae's weights of a specific epoch")
    parser.add_argument('--maskgit_path', type=str, help="path to the trained maskGIT's weights of a specific epoch")
    parser.add_argument('--figures_path', type=str, help="path to save the figures")
    parser.add_argument('--blank_code_path', default=".", type=str, help="root path to save or load blank code")
    parser.add_argument('--blank_code_name', default="blank_code", type=str, help="filename to save or load blank code")
    parser.add_argument('--gen_blank_code', default="True", type=str, help="True if want to generate blank code again")
    args = parser.parse_args()

    device = torch.device(config.device)
    print("--- device: ", device)

    gen_blank_code = (args.gen_blank_code=="True")

    ######### use spherical or polar coordinates
    mode = config.mode
    use_z = False
    if mode=="spherical":
        use_z=True

    ######## evaluation ######
    blank_code_check_iter = 100 #20
    if mode=="polar":
        generation_iter = 20 #80 #40
    else:
        generation_iter = 20
    methods = ["iterative", "one_step", "copy", "uniform"]
    method = methods[0]
    vis=False
    allocentric_angle = 0 #np.pi/2 #0 #np.pi/4
    ##################

    voxelizer = Voxelizer(grid_size=config.grid_size, max_bound=config.max_bound, min_bound=config.min_bound)
    train_pt_dataset = NuscenesEval(args.trainval_data_path, version = args.data_version, split = 'train', vis=False, voxelizer=voxelizer, use_z=use_z, mode=mode, allocentric_angle=allocentric_angle)
    val_pt_dataset = NuscenesEval(args.trainval_data_path, version = args.data_version, split = 'val',vis =False, voxelizer=voxelizer, use_z=use_z, mode=mode, allocentric_angle=allocentric_angle)
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=False)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=False)

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
    if gen_blank_code:
        print("generating blank code")
        mask_git.get_blank_code(path=args.blank_code_path, name=args.blank_code_name, iter=blank_code_check_iter)
    else:
        print("loading previously generated blank code")
        with open(os.path.join(args.blank_code_path, f"{args.blank_code_name}.pickle"), 'rb') as handle:
            blank_code = pickle.load(handle)
            mask_git.blank_code = blank_code

    print(f"--- num blank code: {len(mask_git.blank_code)}")
    print(f"--- blank code: {mask_git.blank_code}")

    dataset = val_dataset
    samples = np.arange(len(dataset))
    #samples = [38]


    torch.cuda.empty_cache()
    seed = 100
    torch.manual_seed(seed)
    np.random.seed(seed)

    range_list_dict = {"our method":[], "vqvae":[], "GT":[], "cut":[]}
    
    start_time = timeit.default_timer()

    for k in samples:
        print(f"@@@@@@@@@ at sample {k}/{len(dataset)}")
        #k = 38 #73 #42
        #k = 1
        #k = 31 #21 #40 #31 #66
        data_tuple = collate_fn_BEV([dataset.__getitem__(k)])
        has, no, voxel_label, BEV_label = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label.to(device) #(B, H, W, in_chans)
        print("YYYYYYYYYYYYOOOOOOOOOOOOOOOOOOO", torch.sum(voxels_mask))

        if torch.sum(voxels_mask)==0:
            print(f"########### ignore sample {k}")
            continue
        

        BEV_mask = BEV_label.to(device) #(B,H,W)

        voxels_occupancy_has = voxels_occupancy_has.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        voxels_occupancy_no = voxels_occupancy_no.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)

        #########################################################################################################
        ################# choose iterative generation or one step prediction ####################################
        #########################################################################################################
        _, vqvae_rec_x_has, _, _, _, _ = vqvae.compute_loss(voxels_occupancy_has)

        if method==methods[0]:
            print(f"+++++ ITERATIVE GENERATION")
            if mode=="polar":
                gen_binary_voxels = mask_git.iterative_generation_driver(voxels_mask, voxels_occupancy_no, BEV_mask, generation_iter, denoise_iter=0, mode=mode)
            else:
                gen_binary_voxels = mask_git.iterative_generation_driver(voxels_mask, voxels_occupancy_no, BEV_mask, generation_iter, denoise_iter=0, mode=mode)
        elif method==methods[1]:
            print(f"ONE-STEP PREDICTION")
            loss, acc, cache = mask_git.one_step_predict(voxels_occupancy_no, voxels_occupancy_has, BEV_mask)
            upsampled_patch_mask = cache["upsampled_patch_mask"].unsqueeze(1).expand(-1,voxels_occupancy_has.shape[1],-1,-1) #(B,in_chans,H,W)
            pred_code_indices_logit = cache["pred_logit"] #(B, total, code_dim)
            _, gen_binary_voxels = mask_git.one_step_decode(pred_code_indices_logit) #(B, in_chans, H, W)
            gen_binary_voxels[voxels_mask.permute(0,3,1,2)==0] = voxels_occupancy_has[voxels_mask.permute(0,3,1,2)==0]

        elif method==methods[2]:
            gen_binary_voxels = torch.clone(voxels_occupancy_has.permute(0,2,3,1)[0].detach().cpu())
            for mask in dataset.obj_voxels_mask_list:
                gen_binary_voxels = voxelizer.copy_and_paste_neighborhood(gen_binary_voxels, voxels_mask=torch.tensor(mask)) #(H,W,in_chans)
            gen_binary_voxels = gen_binary_voxels.unsqueeze(0).permute(0,3,1,2)

        elif method==methods[3]:
            v, m = voxels_occupancy_has.permute(0,2,3,1)[0].detach().cpu(), voxels_mask[0].detach().cpu()
            H,W,C = v.shape
            num_occ = torch.sum(v[m==1])
            prob_occ = num_occ/H/W/C
            print(prob_occ)

            gen_binary_voxels = torch.clone(voxels_occupancy_has)
            gen_binary_voxels[voxels_mask.permute(0,3,1,2)==1] = torch.bernoulli(torch.ones_like(gen_binary_voxels[voxels_mask.permute(0,3,1,2)==1])*prob_occ)

        cut_voxels = torch.clone(voxels_occupancy_has.permute(0,2,3,1)[0].detach().cpu())
        for mask in dataset.obj_voxels_mask_list:
            cut_voxels = voxelizer.copy_and_paste_neighborhood(cut_voxels, voxels_mask=torch.tensor(mask)) #(H,W,in_chans)
        cut_voxels = cut_voxels.unsqueeze(0).permute(0,3,1,2)

        ### evaluate chamfer distance of the masked region
        masked_gen_occupancy = torch.clone(gen_binary_voxels).detach().cpu()
        masked_GT_occupancy = torch.clone(voxels_occupancy_has).detach().cpu()
        masked_vqvae_occupancy = torch.clone(vqvae_rec_x_has).detach().cpu()
        masked_cut_occupancy = torch.clone(cut_voxels).detach().cpu()

        # only keep the masked voxels
        masked_gen_occupancy[voxels_mask.permute(0,3,1,2)!=1] = 0
        masked_GT_occupancy[voxels_mask.permute(0,3,1,2)!=1] = 0
        masked_vqvae_occupancy[voxels_mask.permute(0,3,1,2)!=1] = 0
        masked_cut_occupancy[voxels_mask.permute(0,3,1,2)!=1] = 0

        points_masked_GT = voxels2points(voxelizer, masked_GT_occupancy, mode=mode)
        points_masked_gen = voxels2points(voxelizer, masked_gen_occupancy, mode=mode)
        points_masked_vqvae = voxels2points(voxelizer, masked_vqvae_occupancy, mode=mode)
        points_masked_cut = voxels2points(voxelizer, masked_cut_occupancy, mode=mode)

        print("num_points_GT", len(points_masked_GT[0]))
        print("num_points_gen", len(points_masked_gen[0]))
        print("num_points_vqvae", len(points_masked_vqvae[0]))
        print("num_points_cut", len(points_masked_cut[0]))
        
        fov_up = 10
        fov_down = -30
        img_width = config.grid_size[1]
        img_height = config.grid_size[2]

        range_img_GT = point_cloud_to_range_image(points_masked_GT[0], fov_up, fov_down, img_width, img_height, max_range=100.0)
        range_img_gen = point_cloud_to_range_image(points_masked_gen[0], fov_up, fov_down, img_width, img_height,max_range=100.0)
        range_img_vqvae = point_cloud_to_range_image(points_masked_vqvae[0], fov_up, fov_down, img_width, img_height, max_range=100.0)
        range_img_cut = point_cloud_to_range_image(points_masked_cut[0], fov_up, fov_down, img_width, img_height, max_range=100.0)
        
        # plot_path = "./range_images"
        # plot_range_img(range_img_GT, path=plot_path, name=f"{k}_GT", vis=vis)
        # plot_range_img(range_img_gen, path=plot_path, name=f"{k}_gen", vis=vis)
        # plot_range_img(range_img_vqvae, path=plot_path, name=f"{k}_vqvae", vis=vis)
        # plot_range_img(range_img_cut, path=plot_path, name=f"{k}_cut_naive", vis=vis)
        
        range_list_dict["our method"].append(range_img_GT)
        range_list_dict["vqvae"].append(range_img_gen)
        range_list_dict["GT"].append(range_img_vqvae)
        range_list_dict["cut"].append(range_img_cut)
        
        # the larger ssim is, the more aligned the two images are
        max_data = max([range_img_gen.max(), range_img_GT.max()])
        min_data = min([range_img_gen.min(), range_img_GT.min()])
        ssim_gen , full_ssim_gen = ssim(range_img_GT, range_img_gen, data_range=max_data-min_data, full=True)
        
        max_data = max([range_img_cut.max(), range_img_GT.max()])
        min_data = min([range_img_cut.min(), range_img_GT.min()])
        ssim_cut , full_ssim_cut = ssim(range_img_GT, range_img_cut, data_range=max_data-min_data, full=True)
        
        # plot_range_img(full_ssim_gen, path=plot_path, name=f"{k}_ssim_gen", vis=vis)
        # plot_range_img(full_ssim_cut, path=plot_path, name=f"{k}_ssim_cut_naive", vis=vis)

            
        print(f"SSSSSSSIM   ssim gen: {ssim_gen}                  ==================scene {k}")
        print(f"SSSSSSSIM   ssim cut: {ssim_cut}                  ==================scene {k}")

        #vis = ssim_gen<ssim_cut

        if vis:
            #vis = False
            ## for visualization
            masked_gen_occupancy = torch.clone(gen_binary_voxels).detach().cpu()
            masked_GT_occupancy = torch.clone(voxels_occupancy_has).detach().cpu()
            masked_vqvae_occupancy = torch.clone(vqvae_rec_x_has).detach().cpu()
            masked_cut_occupancy = torch.clone(cut_voxels).detach().cpu()

            points_masked_GT = voxels2points(voxelizer, masked_GT_occupancy, mode=mode)
            points_masked_gen = voxels2points(voxelizer, masked_gen_occupancy, mode=mode)
            points_masked_vqvae = voxels2points(voxelizer, masked_vqvae_occupancy, mode=mode)
            points_masked_cut = voxels2points(voxelizer, masked_cut_occupancy, mode=mode)

        
            voxels_occupancy_list = [masked_GT_occupancy[0].permute(1,2,0), masked_gen_occupancy[0].permute(1,2,0), masked_vqvae_occupancy[0].permute(1,2,0), masked_cut_occupancy[0].permute(1,2,0)]
            points_list = [points_masked_GT, points_masked_gen, points_masked_vqvae, points_masked_cut]
            names_list = ["has (masked)", "generated (masked)", "vqvae reconstruction (masked)", "naive cut and paste"]

            for j, points in enumerate(points_list):
                points = points[0]
                print(f"++++ visualizing {names_list[j]}")

                # get the grid index of each point
                voxel_occupancy = voxels_occupancy_list[j]
                non_zero_indices = torch.nonzero(voxel_occupancy.detach().cpu(), as_tuple=True)
                voxel_mask_ = voxels_mask[0].detach().cpu()

                point_intensity = np.zeros((len(points),))
                assert(len(points)==len(voxel_mask_[non_zero_indices[0], non_zero_indices[1], non_zero_indices[2]]))
                if True:
                    # color the masked regions if there are points there
                    point_intensity_mask = (voxel_mask_[non_zero_indices[0], non_zero_indices[1], non_zero_indices[2]] == 1).detach().cpu().numpy()
                    point_intensity[point_intensity_mask] = 1
                    print("**************** any points in mask region? ", (np.sum(point_intensity)))

                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(np.array(points))
                pcd_colors = np.tile(np.array([[0,0,1]]), (len(points), 1))
                pcd_colors[point_intensity==1, 0] = 1
                pcd_colors[point_intensity==1, 2] = 0
                pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
            
                mat = open3d.visualization.rendering.MaterialRecord()
                mat.shader = 'defaultUnlit'
                mat.point_size = 3.0

                open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], show_skybox=False)

    print("range image categories: ", list(range_list_dict.keys()))

    
    ssim_gen_list, full_ssim_gen_list = compute_ssim(range_list_dict["GT"], range_list_dict["our method"])
    print("---- gen ssim: ")
    print(f"+++ mean ssim: {np.mean(ssim_gen_list)}")
    print(f"max ssim: {np.max(ssim_gen_list)}")
    print(f"min ssim: {np.min(ssim_gen_list)}")
    print(f"std ssim: {np.sqrt(np.var(ssim_gen_list))}")
    
    ssim_vqvae_list, full_ssim_vqvae_list = compute_ssim(range_list_dict["GT"], range_list_dict["vqvae"])
    print("---- vqvae ssim: ")
    print(f"+++ mean ssim: {np.mean(ssim_vqvae_list)}")
    print(f"max ssim: {np.max(ssim_vqvae_list)}")
    print(f"min ssim: {np.min(ssim_vqvae_list)}")
    print(f"std ssim: {np.sqrt(np.var(ssim_vqvae_list))}")

    ssim_cut_list, full_ssim_cut_list = compute_ssim(range_list_dict["GT"], range_list_dict["cut"])
    print("---- cut ssim: ")
    print(f"+++ mean ssim: {np.mean(ssim_cut_list)}")
    print(f"max ssim: {np.max(ssim_cut_list)}")
    print(f"min ssim: {np.min(ssim_cut_list)}")
    print(f"std ssim: {np.sqrt(np.var(ssim_cut_list))}")
    
    mid_time = timeit.default_timer()

    print("TOTAL TIME NEEDED: ------ ", mid_time - start_time)


    