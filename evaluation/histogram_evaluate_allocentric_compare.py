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

import timeit

import logging 




'''
evaluate performance of our foreground object removal pipeline by creating artificial bounding boxes at a fixed radius from the camera and at a fxied allocentric angle to create occlusion
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
    polar_histogram = True
    use_cached_histograms = False
    allocentric_angle = 0 #np.pi/2 #0 #np.pi/4
    ##################

    voxelizer = Voxelizer(grid_size=config.grid_size, max_bound=config.max_bound, min_bound=config.min_bound)
    train_pt_dataset = NuscenesEval(args.trainval_data_path, version = args.data_version, split = 'train', vis=vis, voxelizer=voxelizer, use_z=use_z, mode=mode, allocentric_angle=allocentric_angle)
    val_pt_dataset = NuscenesEval(args.trainval_data_path, version = args.data_version, split = 'val',vis =vis, voxelizer=voxelizer, use_z=use_z, mode=mode, allocentric_angle=allocentric_angle)
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=vis)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=vis)

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

    dataset = train_dataset
    samples = np.arange(len(dataset))
    #samples = [38]


    torch.cuda.empty_cache()
    seed = 100
    torch.manual_seed(seed)
    np.random.seed(seed)

    histograms_list_dict = {"our method":[], "vqvae":[], "GT":[], "cut":[]}


    start_time = timeit.default_timer()
    print("start time: ", start_time)
    
    num_skips = 0
    
    # configuring the logger to display log message 
    # along with log level and time 
    logging.basicConfig(filename="evaluate_message.log", 
                        format='%(asctime)s: %(levelname)s: %(message)s', 
                        level=logging.INFO) 

    # setting logger to critical message 
    logging.critical("starting!!!")
    if use_cached_histograms:
        assert(1==0)
        print(f"USING CACHED HISTOGRAMS ! ! !")
        with open("./histograms_list_dict.pickle", 'rb') as handle:
            histograms_list_dict = pickle.load(handle)
        print("length GT: ", len(histograms_list_dict["GT"]))
        print("length our: ", len(histograms_list_dict["our method"]))
        print("length vqvae: ", len(histograms_list_dict["vqvae"]))
        print("length cut: ", len(histograms_list_dict["cut"]))
        assert(len(histograms_list_dict["GT"])==len(histograms_list_dict["our method"])==len(histograms_list_dict["vqvae"])==len(histograms_list_dict["cut"]))
        #assert(1==0)
        samples = samples[len(histograms_list_dict["GT"]):]
    
    torch.cuda.empty_cache()
    for k in samples:
        print(f"@@@@@@@@@ at sample {k}/{len(dataset)}")
        #k = 38 #73 #42
        #k = 1
        #k = 31 #21 #40 #31 #66
        #k = 40
        k = 26 #80
        logging.info(f"at sample {k}/{len(dataset)}")
        tmp = dataset.__getitem__(k)
        if tmp is None:
            num_skips+=1
            continue
        data_tuple = collate_fn_BEV([tmp])
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
            #### compute iterative generation performance
            # xyzs_list, rec_lidar_logit, gen_binary_voxels = mask_git.conditional_generation(voxels_occupancy_no, BEV_mask, T=generation_iter) #logit and voxel: (B,in_chans,H,W)
            # gen_binary_voxels[voxels_mask.permute(0,3,1,2)==0] = voxels_occupancy_has[voxels_mask.permute(0,3,1,2)==0]

            #### denoising ######
            # for i in range(8):
            #     print(f"$$$$$ denoising iteration {i} $$$$$$")
            #     ## randomly drop some of the mask, and generate again
            #     if mode=="polar":
            #         mask_ratio = 0.5
            #     else:
            #         mask_ratio = 0.02
            #     num_masked = len(BEV_mask[BEV_mask==1])
            #     drop_mask = np.ones((num_masked, ))
            #     drop_mask[:(int)((1-mask_ratio)*num_masked)] = 0
            #     np.random.shuffle(drop_mask)
            #     rand_BEV_mask = torch.clone(BEV_mask)
            #     rand_BEV_mask[rand_BEV_mask==1] = torch.tensor(drop_mask).to(device).long()

            #     loss, acc, cache = mask_git.one_step_predict(gen_binary_voxels, voxels_occupancy_has, rand_BEV_mask)
            #     pred_code_indices_logit = cache["pred_logit"] #(B, total, code_dim)
            #     _, gen_binary_voxels = mask_git.one_step_decode(pred_code_indices_logit) #(B, in_chans, H, W)

            #     gen_binary_voxels[voxels_mask.permute(0,3,1,2)==0] = voxels_occupancy_has[voxels_mask.permute(0,3,1,2)==0]

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

        # points_masked_GT = voxels2points(voxelizer, masked_GT_occupancy, mode=mode)
        # points_masked_gen = voxels2points(voxelizer, masked_gen_occupancy, mode=mode)
        # points_masked_vqvae = voxels2points(voxelizer, masked_vqvae_occupancy, mode=mode)
        # points_masked_cut = voxels2points(voxelizer, masked_cut_occupancy, mode=mode)

        # print("num_points_GT", len(points_masked_GT[0]))
        # print("num_points_gen", len(points_masked_gen[0]))
        # print("num_points_vqvae", len(points_masked_vqvae[0]))
        # print("num_points_cut", len(points_masked_cut[0]))

        

        if polar_histogram:
            # BEV
            # min_bound = [0, 0]#config.min_bound
            # max_bound = [50.635955604688654, 2*np.pi]#config.max_bound
            #max_bound[0] = 25
            # grid_size = [100,100]#config.grid_size
            #grid_size = [300,300]#config.grid_size
            # histogram1 = point_cloud_to_histogram_2D(min_bound, max_bound, grid_size, point_cloud=polar2cart(points_masked_gen[0], mode=mode)).reshape(-1)
            # histogram2 = point_cloud_to_histogram_2D(min_bound, max_bound, grid_size, point_cloud=polar2cart(points_masked_vqvae[0], mode=mode)).reshape(-1)
            # histogram3 = point_cloud_to_histogram_2D(min_bound, max_bound, grid_size, point_cloud=polar2cart(points_masked_GT[0], mode=mode)).reshape(-1)
            # histogram_cut = point_cloud_to_histogram_2D(min_bound, max_bound, grid_size, point_cloud=polar2cart(points_masked_cut[0], mode=mode)).reshape(-1)
            
            ### only take the masked ones
            # masked_gen_occupancy = masked_gen_occupancy[(voxels_mask.permute(0,3,1,2)==1).cpu()]
            # masked_GT_occupancy = masked_GT_occupancy[(voxels_mask.permute(0,3,1,2)==1).cpu()]
            # masked_vqvae_occupancy = masked_vqvae_occupancy[(voxels_mask.permute(0,3,1,2)==1).cpu()]
            # masked_cut_occupancy = masked_cut_occupancy[(voxels_mask.permute(0,3,1,2)==1).cpu()]

            assert(masked_gen_occupancy.shape[1]==config.grid_size[2])
            assert(masked_GT_occupancy.shape[1]==config.grid_size[2])
            assert(masked_cut_occupancy.shape[1]==config.grid_size[2])
            
            ### get BEV occupancy
            masked_gen_occupancy = (torch.sum(masked_gen_occupancy, dim=1)>=1).float()
            masked_GT_occupancy = (torch.sum(masked_GT_occupancy, dim=1)>=1).float()
            masked_vqvae_occupancy = (torch.sum(masked_vqvae_occupancy, dim=1)>=1).float()
            masked_cut_occupancy = (torch.sum(masked_cut_occupancy, dim=1)>=1).float()
            
            
            print("num histogram bin binary: ", masked_cut_occupancy.shape)
            
            histogram1 = masked_gen_occupancy.cpu().numpy().reshape(-1)
            histogram2 = masked_GT_occupancy.cpu().numpy().reshape(-1)
            histogram3 = masked_vqvae_occupancy.cpu().numpy().reshape(-1)
            histogram_cut = masked_cut_occupancy.cpu().numpy().reshape(-1)
            
            
            if np.sum(histogram1)==0 or np.sum(histogram2)==0 or np.sum(histogram3)==0 or np.sum(histogram_cut)==0:
                print(f"== SKIP THIS HISTOGRAM sample {k}")
                num_skips+=1
                continue

            mask = (histogram3!=0)&(histogram1!=0)
            print("number of coarse non zero grid: ", np.sum(mask))
            histograms_list_dict["our method"].append(histogram1)
            histograms_list_dict["vqvae"].append(histogram2)
            histograms_list_dict["GT"].append(histogram3)
            histograms_list_dict["cut"].append(histogram_cut)

            jsds_gen = compute_jsd([histogram3], [histogram1])
            
            print(f"JJJJJJJJJJSD   jsd gen: {jsds_gen}                  ==================scene {k}")

            jsds_cut = compute_jsd([histogram3], [histogram_cut])
            
            print(f"JJJJJJJJJJSD   jsd cut: {jsds_cut}                  ==================scene {k}")

            assert(len(jsds_gen)==1)
            vis = jsds_cut[0]<jsds_gen[0]
            vis=True
           
        else:
            assert(1==0)
            # BEV in cartesian coordinates
            min_bound = [-50,-50]
            max_bound = [50,50]
            grid_size = [60,60]
            histogram = point_cloud_to_histogram_2D(min_bound, max_bound, grid_size, point_cloud=points_masked_gen[0]).reshape(-1)
            
            histograms_list_dict["our method"].append(histogram)
            histogram = point_cloud_to_histogram_2D(min_bound, max_bound, grid_size, point_cloud=points_masked_vqvae[0]).reshape(-1)
            histograms_list_dict["vqvae"].append(histogram)
            histogram = point_cloud_to_histogram_2D(min_bound, max_bound, grid_size, point_cloud=points_masked_GT[0]).reshape(-1)
            histograms_list_dict["GT"].append(histogram)
            
        # if k%1000==0 or k==samples[-1]:
        #     print("saving histogram checkpoint ........")
        #     with open("./histograms_list_dict.pickle", 'wb') as handle:
        #         pickle.dump(histograms_list_dict, handle, protocol=3)
            

        if vis:
            vis = False
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
    
    #if not use_cached_histograms:
    with open("./histograms_list_dict.pickle", 'wb') as handle:
        pickle.dump(histograms_list_dict, handle, protocol=3)
    # else:
    #     with open("./histograms_list_dict.pickle", 'rb') as handle:
    #         histograms_list_dict = pickle.load(handle)
            
    logging.info(f"NUM_SKIPS: {num_skips}")
    

    print("histograms categories: ", list(histograms_list_dict.keys()))

    
    logging.info(f"#### MAXIMUM MEAN DISCREPANCY (MMD)")
    sigma, median = compute_mmd_sigma(histograms_list_dict["GT"], histograms_list_dict["GT"], is_hist=True)
    sigma = np.sqrt(sigma)
    logging.info(f"sigma GT: {sigma}")
    logging.info(f"median GT: {median}")

    
    mmd_gen = compute_mmd(histograms_list_dict["GT"], histograms_list_dict["our method"], kernel=gaussian_kernel, is_hist=True, sigma=(median))
    sigma_gen, median_gen = compute_mmd_sigma(histograms_list_dict["GT"], histograms_list_dict["our method"], is_hist=True)
    sigma_gen = np.sqrt(sigma_gen)
    logging.info(f"++++mmd gen: {mmd_gen}")
    logging.info(f"sigma gen: {sigma_gen}")
    logging.info(f"median gen: {median_gen}")

    
    mmd_vqvae = compute_mmd(histograms_list_dict["GT"], histograms_list_dict["vqvae"], kernel=gaussian_kernel, is_hist=True, sigma=(median))
    sigma_vqvae, median_vqvae = compute_mmd_sigma(histograms_list_dict["GT"], histograms_list_dict["vqvae"], is_hist=True)
    sigma_vqvae = np.sqrt(sigma_vqvae)
    logging.info(f"++++mmd vqvae: {mmd_vqvae}")
    logging.info(f"sigma vqvae: {sigma_vqvae}")
    logging.info(f"median vqvae: {median_vqvae}")
    
    mmd_cut = compute_mmd(histograms_list_dict["GT"], histograms_list_dict["cut"], kernel=gaussian_kernel, is_hist=True, sigma=(median))
    sigma_cut, median_cut = compute_mmd_sigma(histograms_list_dict["GT"], histograms_list_dict["cut"], is_hist=True)
    sigma_cut = np.sqrt(sigma_cut)
    logging.info(f"++++mmd cut: {mmd_cut}")
    logging.info(f"sigma cut: {sigma_cut}")
    logging.info(f"median cut: {median_cut}")

    logging.info(f"##### JSD BETWEEN HISTOGRAMS")

    jsds_gen = compute_jsd(histograms_list_dict["GT"], histograms_list_dict["our method"])
    logging.info("---- gen jsd: ")
    logging.info(f"+++ mean jsd: {np.mean(jsds_gen)}")
    logging.info(f"max jsd: {np.max(jsds_gen)}")
    logging.info(f"min jsd: {np.min(jsds_gen)}")
    logging.info(f"std jsd: {np.sqrt(np.var(jsds_gen))}")

    jsds_vqvae = compute_jsd(histograms_list_dict["GT"], histograms_list_dict["vqvae"])
    logging.info("---- vqvae jsd: ")
    logging.info(f"+++ mean jsd: {np.mean(jsds_vqvae)}")
    logging.info(f"max jsd: {np.max(jsds_vqvae)}")
    logging.info(f"min jsd: {np.min(jsds_vqvae)}")
    logging.info(f"std jsd: {np.sqrt(np.var(jsds_vqvae))}")

    jsds_cut = compute_jsd(histograms_list_dict["GT"], histograms_list_dict["cut"])
    logging.info("---- naive cut and paste jsd: ")
    logging.info(f"+++ mean jsd: {np.mean(jsds_cut)}")
    logging.info(f"max jsd: {np.max(jsds_cut)}")
    logging.info(f"min jsd: {np.min(jsds_cut)}")
    logging.info(f"std jsd: {np.sqrt(np.var(jsds_cut))}")
    
    mid_time = timeit.default_timer()

    logging.info(f"TOTAL TIME NEEDED: ------ {mid_time - start_time} seconds")

    logging.info("###############################################################")


    # print(f"##### JSD BETWEEN HISTOGRAMS datasets")

    # jsds_gen = compute_jsd_between_sets(histograms_list_dict["GT"], histograms_list_dict["our method"])
    # print("---- gen jsd: ")
    # print(f"+++ dataset distance jsd: {jsds_gen}")
    

    # jsds_vqvae = compute_jsd_between_sets(histograms_list_dict["GT"], histograms_list_dict["vqvae"])
    # print("---- vqvae jsd: ")
    # print(f"+++ dataset distance jsd: {(jsds_vqvae)}")
    
    # print(f"##### EARTH MOVER DISTANCE BETWEEN HISTOGRAMS")

    # emds_gen = compute_emd(histograms_list_dict["GT"], histograms_list_dict["our method"], distance_scaling=1.0)
    # print("---- gen emd: ")
    # print(f"+++ mean emd: {np.mean(emds_gen)}")
    # print(f"max emd: {np.max(emds_gen)}")
    # print(f"min emd: {np.min(emds_gen)}")
    # print(f"std emd: {np.sqrt(np.var(emds_gen))}")

    # emds_vqvae = compute_emd(histograms_list_dict["GT"], histograms_list_dict["vqvae"], distance_scaling=1.0)
    # print("---- vqvae emd: ")
    # print(f"+++ mean emd: {np.mean(emds_vqvae)}")
    # print(f"max emd: {np.max(emds_vqvae)}")
    # print(f"min emd: {np.min(emds_vqvae)}")
    # print(f"std emd: {np.sqrt(np.var(emds_vqvae))}")