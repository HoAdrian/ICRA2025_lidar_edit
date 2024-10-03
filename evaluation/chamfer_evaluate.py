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
from datasets.dataset_nuscenes import Nuscenes
import torch
import configs.nuscenes_config as config

from models.vqvae_transformers import VQVAETrans, MaskGIT, voxels2points
import open3d

import pickle
from evaluation_utils import *
#from pytorch3d.loss import chamfer_distance as chamfer



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

    ######## evaluation
    blank_code_check_iter = 100 #20
    generation_iter = 20 #80 #40
    iterative = True
    vis=True
    is_test=False# how to generate mask of the object: if is_test==True, mask all possible objects, otherwise pick one object randomly and rotate it to a free interval and generate the occlusion mask

    voxelizer = Voxelizer(grid_size=config.grid_size, max_bound=config.max_bound, min_bound=config.min_bound)
    train_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'train', filter_valid_scene=True, vis=False, voxelizer=voxelizer, is_test=is_test, use_z=use_z, mode=mode)
    val_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'val', filter_valid_scene=True, vis =False, voxelizer=voxelizer, is_test=is_test, use_z=use_z, mode=mode)
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=True)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=True)

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

    torch.cuda.empty_cache()
    seed = 100
    torch.manual_seed(seed)
    np.random.seed(seed)

    baselines = ["our method", "vqvae"]
    chamfer_dists_total = np.zeros((len(baselines),))
    sample_count = 0
    bad_samples = []
    diffs = []
    chamfer_dists_list = []



    for k in samples:
        print(f"@@@@@@@@@ at sample {k}/{len(dataset)}")
        #k = 38 #73 #42
        #k = 1
        #k = 31 #21 #40 #31 #66
        k = 42
        data_tuple = collate_fn_BEV([dataset.__getitem__(k)])
        has, no, voxel_label, BEV_label = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label.to(device) #(B, H, W, in_chans)
        BEV_mask = BEV_label.to(device) #(B,H,W)

        voxels_occupancy_has = voxels_occupancy_has.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        voxels_occupancy_no = voxels_occupancy_no.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)

        #########################################################################################################
        ################# choose iterative generation or one step prediction ####################################
        #########################################################################################################
        

        if iterative:
            print(f"+++++ ITERATIVE GENERATION")
            #### compute iterative generation performance
            xyzs_list, rec_lidar_logit, gen_binary_voxels = mask_git.conditional_generation(voxels_occupancy_no, BEV_mask, T=generation_iter) #logit and voxel: (B,in_chans,H,W)
            _, vqvae_rec_x_has, _, _, _, _ = vqvae.compute_loss(voxels_occupancy_has)
            gen_binary_voxels[voxels_mask.permute(0,3,1,2)==0] = voxels_occupancy_has[voxels_mask.permute(0,3,1,2)==0]

            #### denoising ######
            for i in range(8):
                print(f"$$$$$ denoising iteration {i} $$$$$$")
                ## randomly drop some of the mask, and generate again
                mask_ratio = 0.5
                num_masked = len(BEV_mask[BEV_mask==1])
                drop_mask = np.ones((num_masked, ))
                drop_mask[:(int)((1-mask_ratio)*num_masked)] = 0
                np.random.shuffle(drop_mask)
                rand_BEV_mask = torch.clone(BEV_mask)
                rand_BEV_mask[rand_BEV_mask==1] = torch.tensor(drop_mask).to(device).long()

                loss, acc, cache = mask_git.one_step_predict(gen_binary_voxels, voxels_occupancy_has, rand_BEV_mask)
                pred_code_indices_logit = cache["pred_logit"] #(B, total, code_dim)
                _, gen_binary_voxels = mask_git.one_step_decode(pred_code_indices_logit) #(B, in_chans, H, W)

                gen_binary_voxels[voxels_mask.permute(0,3,1,2)==0] = voxels_occupancy_has[voxels_mask.permute(0,3,1,2)==0]
        else:
            print(f"ONE-STEP PREDICTION")
            loss, acc, cache = mask_git.one_step_predict(voxels_occupancy_no, voxels_occupancy_has, BEV_mask)
            upsampled_patch_mask = cache["upsampled_patch_mask"].unsqueeze(1).expand(-1,voxels_occupancy_has.shape[1],-1,-1) #(B,in_chans,H,W)
            pred_code_indices_logit = cache["pred_logit"] #(B, total, code_dim)
            _, gen_binary_voxels = mask_git.one_step_decode(pred_code_indices_logit) #(B, in_chans, H, W)
            gen_binary_voxels[voxels_mask.permute(0,3,1,2)==0] = voxels_occupancy_has[voxels_mask.permute(0,3,1,2)==0]

            _, vqvae_rec_x_has, _, _, _, _ = vqvae.compute_loss(voxels_occupancy_has)

        ### evaluate chamfer distance of the masked region
        masked_gen_occupancy = torch.clone(gen_binary_voxels).detach().cpu()
        masked_GT_occupancy = torch.clone(voxels_occupancy_has).detach().cpu()
        masked_vqvae_occupancy = torch.clone(vqvae_rec_x_has).detach().cpu()

        # only keep the masked voxels
        masked_gen_occupancy[voxels_mask.permute(0,3,1,2)!=1] = 0
        masked_GT_occupancy[voxels_mask.permute(0,3,1,2)!=1] = 0
        masked_vqvae_occupancy[voxels_mask.permute(0,3,1,2)!=1] = 0

        points_masked_GT = voxels2points(voxelizer, masked_GT_occupancy, mode=mode) #[point cloud]
        points_masked_gen = voxels2points(voxelizer, masked_gen_occupancy, mode=mode)
        points_masked_vqvae = voxels2points(voxelizer, masked_vqvae_occupancy, mode=mode)

        #### make the discretization coarser
        # coarse_voxelizer = Voxelizer(grid_size=[60,60,32], max_bound=config.max_bound, min_bound=config.min_bound)
        # _, _, _, coarse_masked_GT_occupancy = coarse_voxelizer.voxelize(cart2polar(points_masked_GT[0], mode=mode), return_point_info=False) #(H,W,in_chans)
        # _, _, _, coarse_masked_gen_occupancy = coarse_voxelizer.voxelize(cart2polar(points_masked_gen[0], mode=mode), return_point_info=False)
        # _, _, _, coarse_masked_vqvae_occupancy = coarse_voxelizer.voxelize(cart2polar(points_masked_vqvae[0], mode=mode), return_point_info=False)
        # points_masked_GT = voxels2points(coarse_voxelizer, torch.tensor(coarse_masked_GT_occupancy).permute(2,0,1).unsqueeze(0), mode=mode) # a list containing one point cloud
        # points_masked_gen = voxels2points(coarse_voxelizer, torch.tensor(coarse_masked_gen_occupancy).permute(2,0,1).unsqueeze(0), mode=mode)
        # points_masked_vqvae = voxels2points(coarse_voxelizer, torch.tensor(coarse_masked_vqvae_occupancy).permute(2,0,1).unsqueeze(0), mode=mode)

        # if vis:
        #     coarse_points_list = [points_masked_GT, points_masked_gen, points_masked_vqvae]
        #     names_list = ["coarse has (masked)", "coarse generated (masked)", "coarse vqvae reconstruction (masked)"]

        #     for j, pointcloud in enumerate(coarse_points_list):
        #         print(f"++++ visualizing {names_list[j]}")

        #         pcd = open3d.geometry.PointCloud()
        #         pcd.points = open3d.utility.Vector3dVector(np.array(pointcloud[0][:,:3]))
        #         pcd_colors = np.tile(np.array([[0,0,1]]), (len(pointcloud[0]), 1))
        #         pcd.colors = open3d.utility.Vector3dVector(pcd_colors)

        #         mat = open3d.visualization.rendering.MaterialRecord()
        #         mat.shader = 'defaultUnlit'
        #         mat.point_size = 2.0

        #         open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], show_skybox=False)


        print("num_points_GT", len(points_masked_GT[0]))
        print("num_points_gen", len(points_masked_gen[0]))
        print("num_points_vqvae", len(points_masked_vqvae[0]))

        if len(points_masked_gen[0])==0 or len(points_masked_GT[0])==0 or len(points_masked_vqvae[0])==0:
            print(f"WARNING: skip sample {k} due to empty point cloud (either generated or reconstructed)")
            continue

        # chamfer_dist_gen = chamfer(torch.tensor(points_masked_GT_occupancy[0]).unsqueeze(0).float(), torch.tensor(points_masked_gen_occupancy[0]).unsqueeze(0).float())
        # chamfer_dist_vqvae = chamfer(torch.tensor(points_masked_GT_occupancy[0]).unsqueeze(0).float(),  torch.tensor(points_masked_vqvae_occupancy[0]).unsqueeze(0).float())
        #### use BEV ####
        # points_BEV_GT = np.unique(points_masked_GT[0][:,:2], axis=0)
        # points_BEV_gen = np.unique(points_masked_gen[0][:,:2], axis=0)
        # points_BEV_vqvae = np.unique(points_masked_vqvae[0][:,:2], axis=0)
        ############
        chamfer_dist_gen = chamfer_distance_pytorch(torch.tensor(points_masked_GT[0]).float(), torch.tensor(points_masked_gen[0]).float())
        chamfer_dist_vqvae = chamfer_distance_pytorch(torch.tensor(points_masked_GT[0]).float(),  torch.tensor(points_masked_vqvae[0]).float())
        # chamfer_dist_gen = chamfer_distance_pytorch(torch.tensor(points_BEV_GT).float(), torch.tensor(points_BEV_gen).float())
        # chamfer_dist_vqvae = chamfer_distance_pytorch(torch.tensor(points_BEV_GT).float(),  torch.tensor(points_BEV_vqvae).float())
        print(f"++ chamfer gen vs GT: {chamfer_dist_gen}")
        print(f"++ chamfer vqvae vs GT: {chamfer_dist_vqvae}")
        
        if chamfer_dist_gen > chamfer_dist_vqvae:
            bad_samples.append(k)
            diffs.append(chamfer_dist_vqvae - chamfer_dist_gen)

        chamfer_dists_total[0] += chamfer_dist_gen
        chamfer_dists_total[1] += chamfer_dist_vqvae
        sample_count+=1
        chamfer_dists_list.append(chamfer_dist_gen)

        if vis:
            ## for visualization
            masked_gen_occupancy = torch.clone(gen_binary_voxels).detach().cpu()
            masked_GT_occupancy = torch.clone(voxels_occupancy_has).detach().cpu()
            masked_vqvae_occupancy = torch.clone(vqvae_rec_x_has).detach().cpu()

            points_masked_GT = voxels2points(voxelizer, masked_GT_occupancy, mode=mode)
            points_masked_gen = voxels2points(voxelizer, masked_gen_occupancy, mode=mode)
            points_masked_vqvae = voxels2points(voxelizer, masked_vqvae_occupancy, mode=mode)
        
            voxels_occupancy_list = [masked_GT_occupancy[0].permute(1,2,0), masked_gen_occupancy[0].permute(1,2,0), masked_vqvae_occupancy[0].permute(1,2,0)]
            points_list = [points_masked_GT, points_masked_gen, points_masked_vqvae]
            names_list = ["has (masked)", "generated (masked)", "vqvae reconstruction (masked)"]

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

    chamfer_dists_mean = chamfer_dists_total/sample_count
    chamfer_dists_list = np.array(chamfer_dists_list)
    diffs = np.array(diffs)
    bad_samples = np.array(bad_samples)
    sort_args = np.argsort(diffs)
    diffs = diffs[sort_args]
    bad_samples = bad_samples[sort_args]

    print("baselines: ", baselines)
    print(f"chamfer_dists_mean: ", chamfer_dists_mean)
    print(f"chamfer_dists_std: ", np.std(chamfer_dists_list))
    print(f"max chamfer: {np.max(chamfer_dists_list)}")
    print(f"min chamfer: {np.min(chamfer_dists_list)}")
    print(f"sample count: {sample_count}/{len(dataset)}")
    print(f"bad sample idxs: {bad_samples}")
    print(f"chamfer gen minus vqvae : {diffs}")
    