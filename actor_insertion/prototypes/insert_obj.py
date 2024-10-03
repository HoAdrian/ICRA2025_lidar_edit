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

from actor_insertion.insertion_utils import insert_vehicle_pc, sample_valid_insert_pos
from nuscenes.utils.geometry_utils import points_in_box

'''
Remove all existing foreground objects first and insert new foreground objects at random locations on driveable surface
'''

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--trainval_data_path', type=str, help="path to training and validation examples")
    parser.add_argument('--data_version', type=str, help="versioin of the dataset e.g. nuscenes")
    parser.add_argument('--pc_path', type=str, help="path to load the point clouds")
    parser.add_argument('--dense', type=int, help="whether to use dense point cloud from point cloud completion")
    parser.add_argument('--maskgit_path', type=str, help="path to the trained maskGIT's weights of a specific epoch")
    parser.add_argument('--vqvae_path', type=str, help="path to the trained vqvae's weights of a specific epoch")
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

    voxelizer = Voxelizer(grid_size=config.grid_size, max_bound=config.max_bound, min_bound=config.min_bound)
    train_pt_dataset = NuscenesForeground(args.trainval_data_path, version = args.data_version, split = 'train', vis=True, mode=mode)
    val_pt_dataset = NuscenesForeground(args.trainval_data_path, version = args.data_version, split = 'val', vis =True, mode=mode)
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=True)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, vis=True)


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

    dataset = val_dataset
    samples = np.arange(len(dataset))
    num_objs_insert = 5
    available_objs = ["car", "bus", "truck"]

    
    torch.cuda.empty_cache()
    seed = 100
    torch.manual_seed(seed)
    np.random.seed(seed)


    for k in samples:
        #k = 9
        data_tuple = collate_fn_BEV([dataset.__getitem__(k)])
        has, no, voxel_label, BEV_label = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label.to(device) #(B, H, W, in_chans)
        BEV_mask = BEV_label.to(device) #(B,H,W)

        # remove existing foreground objects first
        voxels_occupancy_has = voxels_occupancy_has.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        voxels_occupancy_no = voxels_occupancy_no.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        gen_binary_voxels = mask_git.iterative_generation_driver(voxels_mask, voxels_occupancy_no, BEV_mask, generation_iter=20, denoise_iter=0, mode=mode)
        
        voxels_occupancy_list = [voxels_occupancy_has[0].permute(1,2,0), gen_binary_voxels[0].permute(1,2,0)]
        points_list = [voxels2points(voxelizer, voxels_occupancy_has, mode=mode)[0], voxels2points(voxelizer, gen_binary_voxels, mode=mode)[0]]
        names_list = ["has (masked)", "generated (masked)"]
        visualize_generated_pointclouds(voxelizer, voxels_occupancy_list, points_list, names_list, voxels_mask, image_path="./actor_insertion/vis_no_background", image_name=f"sample_{k}")

        # the new occupancy with foreground points removed
        voxels_occupancy_has = gen_binary_voxels.permute(0,2,3,1)
        new_bboxes = []

        names = [available_objs[np.random.choice(len(available_objs), p=np.array([0.7, 0.15, 0.15]))] for _ in range(num_objs_insert)]
        # iterate over each object we want to insert
        for i, name in enumerate(names):
            ##### set a valid pos to insert point cloud

            with open(os.path.join(args.pc_path, "allocentric.pickle"), 'rb') as handle:
                allocentric_dict = pickle.load(handle)

            obj_properties = allocentric_dict[name] # the properties of objects belonging to the specified name
            allocentric_angles = np.array(obj_properties[0])
            pc_filenames = obj_properties[1]
            viewing_angles = np.array(obj_properties[2])
            boxes = obj_properties[3]
            center3Ds = np.array(obj_properties[4])
            N = len(allocentric_angles)
            assert(len(allocentric_angles)==len(pc_filenames)==len(viewing_angles)==len(boxes)==len(center3Ds))

            ####### choose an object
            #np.argmin(np.abs(viewing_angles - desired_viewing_angle))
            # center3Ds_polar = cart2polar(center3Ds, mode="polar")
            # desired_polar = cart2polar(insert_xyz_pos[np.newaxis,:], mode="polar")
            # top_k = 10 #len(center3Ds_polar)
            # chosen_idxs = np.argpartition(np.abs(desired_polar[0,0] - center3Ds_polar[:,0]), top_k)[:top_k]
            # chosen_idx = chosen_idxs[0]
            chosen_idx = np.random.randint(low=0, high=N)
           
            pc_filename = pc_filenames[chosen_idx]
            bbox = boxes[chosen_idx]
            center3D = center3Ds[chosen_idx]
            pc_path = os.path.join(args.pc_path, name)

            use_dense = args.dense==1
            if use_dense:
                pc_full_path = os.path.join(args.pc_path, "dense_nusc", name, pc_filename)
            else:
                pc_full_path = os.path.join(pc_path, pc_filename)
                print("using sparse vehicle point cloud")
                #raise Exception("I prefer completed point cloud LOL")
            vehicle_pc = np.asarray(open3d.io.read_point_cloud(pc_full_path).points) #np.load(pc_full_path)

            ###### choose where to insert vehicle
            insert_xyz_pos = sample_valid_insert_pos(name, viewing_angles[chosen_idx], dataset, bbox, new_bboxes)

            desired_viewing_angle = compute_viewing_angle(insert_xyz_pos[:2])
            current_viewing_angle = viewing_angles[chosen_idx]
            rotation_align = -(desired_viewing_angle - current_viewing_angle) # negative sign because gamma increases clockwise

            new_scene_points_xyz, new_occupancy, new_bbox, insert_xyz_pos, vehicle_pc = insert_vehicle_pc(vehicle_pc, bbox, insert_xyz_pos, rotation_align, voxels_occupancy_has, voxelizer, dataset, mode=mode, use_ground_seg=True, center=center3D, kitti2nusc=False, use_dense=use_dense)
            voxels_occupancy_has = torch.tensor(new_occupancy).unsqueeze(0)
            new_bboxes.append(new_bbox)

            # if k==9:
            #     box = new_bbox
            #     plt.figure(figsize=(8, 6))
            #     plt.gca().set_aspect('equal')
            #     corners = box.corners() #(3,8)
            #     corner_1 = corners[:,0][:2]
            #     corner_2 = corners[:,1][:2]
            #     corner_5 = corners[:,4][:2]
            #     corner_6 = corners[:,5][:2]
            #     rect = patches.Polygon([corner_1, corner_2, corner_6,corner_5], linewidth=1, edgecolor='r', facecolor='none')
            #     # Get the current axes and plot the polygon patch
            #     plt.gca().add_patch(rect)
            #     plt.scatter(vehicle_pc[:,0], vehicle_pc[:,1], s=10)
            #     plt.show()

            

            
        # visualize inserted vehicle only
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(np.array(vehicle_pc))
        # pcd_colors = np.tile(np.array([[0,0,1]]), (len(vehicle_pc), 1))
        # pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
        # car_vis_pos = open3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        # car_vis_pos.paint_uniform_color([1,0,0])  
        # car_vis_pos.translate(tuple(insert_xyz_pos))
        # open3d.visualization.draw_geometries([pcd, car_vis_pos])

        ### remove new bboxes that contain no points after applying occlusion
        new_bboxes_copy = [copy.deepcopy(box) for box in new_bboxes]
        new_bboxes = []
        for i, box in enumerate(new_bboxes_copy):
            mask = points_in_box(box, new_scene_points_xyz.T, wlh_factor = 1.0)
            if np.sum(mask)!=0:# and np.sum(mask)>50:
                new_bboxes.append(box)
                #new_bboxes.remove(new_bboxes[i])
                #del new_bboxes[i]


        plot_obj_regions([], [], new_scene_points_xyz, 40, new_bboxes, xlim=[-40,40], ylim=[-40,40], title=f"raw{i}", path="./actor_insertion/vis_insert", name=f"insert_sample_{k}", vis=False)


        # # visualize the scene with an object added
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(np.array(new_scene_points_xyz))
        # pcd_colors = np.tile(np.array([[0,0,1]]), (len(new_scene_points_xyz), 1))
        # pcd.colors = open3d.utility.Vector3dVector(pcd_colors)

        # car_vis_pos = open3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        # car_vis_pos.paint_uniform_color([1,0,0])  
        # car_vis_pos.translate(tuple(insert_xyz_pos))


        # mat = open3d.visualization.rendering.MaterialRecord()
        # mat.shader = 'defaultUnlit'
        # mat.point_size = 2.0
        # open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}, {'name': 'car', 'geometry': car_vis_pos, 'material': mat}], show_skybox=False)


        

    





