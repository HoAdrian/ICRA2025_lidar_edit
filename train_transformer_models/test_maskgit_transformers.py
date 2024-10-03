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

    voxelizer = Voxelizer(grid_size=config.grid_size, max_bound=config.max_bound, min_bound=config.min_bound)
    train_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'train', filter_valid_scene=True, vis=True, voxelizer=voxelizer, is_test=True, use_z=use_z, mode=mode)
    val_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'val', filter_valid_scene=True, vis =True, voxelizer=voxelizer, is_test=True, use_z=use_z, mode=mode)
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

    blank_code_check_iter = 100 #20
    generation_iter = 20 #80 #40 ## polar: 20 - 40, sphere: 80
    
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

    num_vis = 100
    dataset = val_dataset
    samples = np.random.choice(len(dataset), num_vis)
    torch.cuda.empty_cache()
    seed = 100
    torch.manual_seed(seed)
    np.random.seed(seed)
    for k in samples:
        k = 38 #62
        #k = 44 #80 #44 #66 #21 #31 #21 #40 #31 #66
        print(f"############# SAMPLE k: {k}")
        data_tuple = collate_fn_BEV([dataset.__getitem__(k)])
        has, no, voxel_label, BEV_label = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label.to(device) #(B, H, W, in_chans)
        BEV_mask = BEV_label.to(device) #(B,H,W)

        voxels_occupancy_has = voxels_occupancy_has.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        voxels_occupancy_no = voxels_occupancy_no.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)

        _, vqvae_rec_x_has, _, _, _, _ = vqvae.compute_loss(voxels_occupancy_has)

        ###### uncomment this if you want unconditional generation and you also need to set the mask ratio in the denoising iterations
        # voxels_mask[...] = 1
        # BEV_mask[...] = 1

        ### compute one step prediction performance
        # loss, acc, cache = mask_git.one_step_predict(voxels_occupancy_no, voxels_occupancy_has, BEV_mask)
        # upsampled_patch_mask = cache["upsampled_patch_mask"].unsqueeze(1).expand(-1,voxels_occupancy_has.shape[1],-1,-1) #(B,in_chans,H,W)
        # pred_code_indices_logit = cache["pred_logit"] #(B, total, code_dim)
        # _, rec_occupancy = mask_git.one_step_decode(pred_code_indices_logit) #(B, in_chans, H, W)

        # masked_rec_occupancy = (rec_occupancy)[upsampled_patch_mask==1].reshape(-1).detach().cpu().numpy()
        # masked_GT_occupancy = (voxels_occupancy_has)[upsampled_patch_mask==1].reshape(-1).detach().cpu().numpy()

        # _, _, TPs, FPs, FNs, TNs =confusion_matrix_wrapper(masked_GT_occupancy, masked_rec_occupancy, labels=np.arange(2))

        # accuracy, precision, recall, f1_score, specificity, TPR, FPR = compute_perf_metrics(TPs, FPs, FNs, TNs)  
        
        # print("++++ PERFORMANCE for one step prediction: ")
        # print(f"precision: {precision[1]}")
        # print(f"recall: {recall[1]}")
        # print(f"f1 score: {f1_score[1]}")
        # print(f"sepcificity: {specificity[1]}")
        # print(f"TPR: {TPR[1]}")
        # print(f"FPR: {FPR[1]}")
  
        #### compute iterative generation performance
        # xyzs_list, rec_lidar_logit, gen_binary_voxels = mask_git.conditional_generation(voxels_occupancy_no, BEV_mask, T=generation_iter) #logit and voxel: (B,in_chans,H,W)
        # gen_binary_voxels[voxels_mask.permute(0,3,1,2)==0] = voxels_occupancy_has[voxels_mask.permute(0,3,1,2)==0]

        #### denoising ######
        # for i in range(8):
        #     print(f"$$$$$ denoising iteration {i} $$$$$$")
        #     ## randomly drop some of the mask, and generate again
        #     ## mask ratio use 0.02 if it is unconditional generation (everything is masked), otherwise use mask ratio = 0.5
        #     mask_ratio = 0.02 # 0.02 #0.1 #0.5 #0.02
        #     num_masked = len(BEV_mask[BEV_mask==1])
        #     drop_mask = np.ones((num_masked, ))
        #     drop_mask[:(int)((1-mask_ratio)*num_masked)] = 0
        #     np.random.shuffle(drop_mask)
        #     rand_BEV_mask = torch.clone(BEV_mask)
        #     rand_BEV_mask[rand_BEV_mask==1] = torch.tensor(drop_mask).to(device).long()
        #     # print("---rand bev mask: ", torch.sum(rand_BEV_mask))
        #     # print("---bev mask: ", torch.sum(BEV_mask))

        #     loss, acc, cache = mask_git.one_step_predict(gen_binary_voxels, voxels_occupancy_has, rand_BEV_mask)
        #     pred_code_indices_logit = cache["pred_logit"] #(B, total, code_dim)
        #     _, gen_binary_voxels = mask_git.one_step_decode(pred_code_indices_logit) #(B, in_chans, H, W)

        #     gen_binary_voxels[voxels_mask.permute(0,3,1,2)==0] = voxels_occupancy_has[voxels_mask.permute(0,3,1,2)==0]

        gen_binary_voxels = mask_git.iterative_generation_driver(voxels_mask, voxels_occupancy_no, BEV_mask, generation_iter, denoise_iter=0, mode=mode)
            
        xyzs_list = voxels2points(voxelizer, gen_binary_voxels, mode=mode)

            
        

        vox_size = 1
        xlim = [-80, 80]
        ylim = [-80, 80]
        #voxelizer.vis_BEV_binary_voxel(voxels_occupancy_has[0].permute(1,2,0), points_xyz=None, intensity=None, vis=False, path=f"{args.figures_path}", name=f"{k}_has_occupancy", vox_size=vox_size, xlim=xlim, ylim=ylim, only_points=False)
        #voxelizer.vis_BEV_binary_voxel(voxels_occupancy_no[0].permute(1,2,0), points_xyz=None, intensity=None, vis=False, path=f"{args.figures_path}", name=f"{k}_no_occupancy", vox_size=vox_size, xlim=xlim, ylim=ylim, only_points=False)
        #voxelizer.vis_BEV_binary_voxel(gen_binary_voxels[0].permute(1,2,0), points_xyz=None, intensity=None, vis=False, path=f"{args.figures_path}", name=f"{k}_generated_occupancy", vox_size=vox_size, xlim=xlim, ylim=ylim, only_points=False)
        #voxelizer.vis_BEV_binary_voxel(BEV_mask[0].unsqueeze(-1), points_xyz=None, intensity=None, vis=False, path=f"{args.figures_path}", name=f"{k}_BEV_label", vox_size=vox_size, xlim=xlim, ylim=ylim, only_points=False, mode=mode)
        #voxelizer.vis_BEV_binary_voxel(vqvae_rec_x_has[0].permute(1,2,0), points_xyz=None, intensity=None, vis=False, path=f"{args.figures_path}", name=f"{k}_vqvae_rec_has_occupancy", vox_size=vox_size, xlim=xlim, ylim=ylim, only_points=False)
        
        ### visualize original point cloud
        print(f"++++ visualizing ORIGINAL POINT CLOUD")
        original_points = dataset.points_xyz
        print(original_points.shape)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(original_points[:,:3]))
        point_intensity = dataset.points_in_box_mask
        pcd_colors = np.tile(np.array([[0,0,1]]), (len(original_points), 1))
        pcd_colors[point_intensity==1, 0] = 1
        pcd_colors[point_intensity==1, 2] = 0
        #pcd_colors = np.tile(np.array([[0,0,1]]), (len(original_points), 1))*original_points[:,4:5]/np.max(original_points[:,4:5])
        pcd.colors = open3d.utility.Vector3dVector(pcd_colors)

        mat = open3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.point_size = 2.0

        open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], show_skybox=False)
        ########################################

        points_xyz_has = voxels2points(voxelizer, voxels_occupancy_has, mode=mode)
        points_xyz_no = voxels2points(voxelizer, voxels_occupancy_no, mode=mode)
        points_xyz_vqvae_rec = voxels2points(voxelizer, vqvae_rec_x_has, mode=mode)

        voxels_occupancy_list = [voxels_occupancy_has[0].permute(1,2,0), voxels_occupancy_no[0].permute(1,2,0), gen_binary_voxels[0].permute(1,2,0), vqvae_rec_x_has[0].permute(1,2,0)]
        points_list = [points_xyz_has, points_xyz_no, xyzs_list, points_xyz_vqvae_rec]
        names_list = ["has", "no", "generated", "vqvae reconstruction"]

        for j, points in enumerate(points_list):
            points = points[0]
            print(f"++++ visualizing {names_list[j]}")

            # get the grid index of each point
            voxel_occupancy = voxels_occupancy_list[j]
            non_zero_indices = torch.nonzero(voxel_occupancy.detach().cpu(), as_tuple=True)
            voxel_mask_ = voxels_mask[0].detach().cpu()

            point_intensity = np.zeros((len(points),))
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
            #open3d.visualization.draw_geometries([pcd]) 

            mat = open3d.visualization.rendering.MaterialRecord()
            mat.shader = 'defaultUnlit'
            mat.point_size = 2.0

            open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], show_skybox=False)

            #voxelizer.vis_BEV_binary_voxel(BEV_mask[0].unsqueeze(-1), points_xyz=points, intensity=point_intensity, vis=False, path=f"{args.figures_path}", name=f"{k}_{names_list[j]}_points_masked", vox_size=vox_size, xlim=xlim, ylim=ylim, only_points=True)