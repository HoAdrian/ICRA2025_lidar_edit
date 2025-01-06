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
from datasets.dataset import Voxelizer, PolarDataset, collate_fn_BEV
from datasets.dataset_nuscenes import Nuscenes, NuscenesForeground, vehicle_names
import torch
import configs.nuscenes_config as config

from models.vqvae_transformers import VQVAETrans, voxels2points
import open3d
import pickle

from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import points_in_box
import time

import math

'''
Collect foreground object point clouds and their properties such as bounding boxes. REMEMBER to remove all previously collected foreground objects first. otherwise, there would be 
inconsistencies between the dictionaries and the point clouds we save. 
'''

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--trainval_data_path', type=str, help="path to training and validation examples")
    parser.add_argument('--data_version', type=str, help="versioin of the dataset e.g. nuscenes")
    parser.add_argument('--pc_path', type=str, help="path to save the point clouds")
    parser.add_argument('--visualize_dense', type=int, help="1 means visualize completed point clouds, 0 means collect foreground objects")
    parser.add_argument('--save_as_pcd', type=int, help="1 means save point cloud as .pcd file, 0 means save pointcloud as .npy file")
    args = parser.parse_args()

    print("WARNING: REMEMBER TO REMOVE ALL PREVIOUSLY COLLECTED FOREGROUND OBJECTS FIRST...... ")
    time.sleep(2)

    config.device = "cpu"
    device = torch.device(config.device)
    print("--- device: ", device)

    ######### use spherical or polar coordinates
    mode = config.mode
    use_z = False
    if mode=="spherical":
        use_z=True

    visualize_dense = args.visualize_dense==1
    save_as_pcd = args.save_as_pcd==1
    count_bus_val = 0

    np.random.seed(24) #2021
    #trainval
    #20: good single truck, bus, car
    #25
    #53: good bus, bad truck
    #24: all good

    os.makedirs(args.pc_path, exist_ok=True)
    
    if not visualize_dense:
        multisweep = False
        train_dataset = NuscenesForeground(args.trainval_data_path, version = args.data_version, split = 'train', mode=mode, vis=False, multisweep=multisweep)
        val_dataset = NuscenesForeground(args.trainval_data_path, version = args.data_version, split = 'val', mode=mode, vis=False, multisweep=multisweep)

        #### maps from vehicle name to a 2d array of two rows. 
        # First row contains the allocentric angles and 
        # the second rows contains the filename of the corresponding point cloud, 
        # third row contains the viewing angles
        # forth row contains the bounding boxes
        # fifth row contains the 3D positions of the boxes' centers
        # sixth row contains kitti's bounding boxes tuple (cam, lidar)
        allocentric_dict = {name: [[], [], [], [], [], [], [], [], [],[],[],[], []] for name in vehicle_names.values()}
        sample_dict = {}

        max_num_bus = 1e9#100
        max_num_truck = 1e9#100
        max_num_car = 1e9#100
        num_car, num_truck, num_bus = 0,0,0
        datasets = [val_dataset]#[val_dataset, train_dataset]

        # use val_dataset first if it is mini
        for i, dataset in enumerate(datasets):
            if i==1:
                is_train=True
                ### for only using val dataset
                break
            else:
                is_train=False
            
            ############ for handpicking vehicles
            samples = np.arange(len(dataset))
            np.random.shuffle(samples)
            # samples[0] = 40
            #samples = [196, 296]
            for k in samples:
                if os.path.exists(os.path.join(args.pc_path, "bus")):
                    num_bus = len(os.listdir(os.path.join(args.pc_path, "bus")))
                if os.path.exists(os.path.join(args.pc_path, "car")):
                    num_car = len(os.listdir(os.path.join(args.pc_path, "car")))
                if os.path.exists(os.path.join(args.pc_path, "truck")):
                    num_truck = len(os.listdir(os.path.join(args.pc_path, "truck")))

                print(f" . num bus: {num_bus} | num car: {num_car} | num truck: {num_truck}")
                if num_bus>=max_num_bus and num_car>=max_num_car and num_truck>=max_num_truck: #40, 500, 180
                    print(num_bus)
                    print(num_car)
                    print(num_truck)
                    print("ENOUGH is ENOUGH, enough car, bus and trucks")
                    break
                #k = 31 #56 #31 #66, 31
                #k = 44
                item = dataset.__getitem__(k)
                if item is None:
                    continue
                _, _, _ ,_ , _, obj_properties = item
                obj_point_cloud_list, obj_name_list, scene_points, obj_allocentric_list, obj_centers_list, obj_boxes_list, \
                    obj_gamma_list, kitti_boxes_list, lidar_sample_token, all_boxes, obj_ann_token_list, sample_annotation_tokens, sample_records, obj_ann_info_list, curr_sample_table_token, not_empty = obj_properties

                print(f"#### NUM OBJECTS: {len(obj_point_cloud_list)}")
                num_obj = len(obj_point_cloud_list)

                for i in range(num_obj):
                    points = obj_point_cloud_list[i]
                    name = obj_name_list[i]
                    alpha = obj_allocentric_list[i]
                    gamma = obj_gamma_list[i]
                    center3D = obj_centers_list[i]
                    box = obj_boxes_list[i]

                    if name=="bus" and num_bus>=max_num_bus:
                        continue
                    elif name=="car" and num_car>=max_num_car:
                        continue
                    elif name=="truck" and num_truck>=max_num_truck:
                        continue
                    
                    # print(name)
                    # print(len(points))
                    # time.sleep(3)

                    #### for very good completion and handpicking
                    # is_good_completion = ((len(points) >= 1000) or (name=="car" and len(points)>=600)) #500, 200
                    is_good_completion = ((len(points) >= 200) or (name=="car" and len(points)>=200)) #500, 200

                    front, side, vert = dataset.box_length_stats["front"], dataset.box_length_stats["side"], dataset.box_length_stats["vertical"]
                    print(f"---front={front[-1]}, side={side[-1]}, vertical={vert[-1]}")
                    if math.isnan(front[-1]) or math.isnan(side[-1]) or math.isnan(vert[-1]):
                        print(front)
                        print(side)
                        print(vert)
                        time.sleep(100)

                    # if (True or is_good_completion) and len(points)>0:
                    if is_good_completion:

                        print(f"sample {k}: object {i}, category={name}, num_points: {len(points)} ***len sample dict: {len(sample_dict)}")
                        ## filter out ground points by heuristics
                        # min_z = np.min(points[:,2])
                        # mean_z = center3D[2]
                        # mask = points[:,2]> (min_z + (mean_z - min_z)/9)
                        # points = points[mask]

                        ## normalize points
                        points_normalized = points - center3D

                        save_path = os.path.join(args.pc_path, name)
                        os.makedirs(save_path, exist_ok=True)
                        num_existing = len(os.listdir(save_path))

                        sample_num = num_existing
                        print(f"---- num existing in {name}: {sample_num}")
                        
                        #### save in npy format
                        if not save_as_pcd:
                            pc_name = f'sample_{sample_num}.npy'
                            pc_full_path = os.path.join(save_path, pc_name)
                            # np.save(pc_full_path, points_normalized)
                            assert(not os.path.exists(pc_full_path))
                            np.save(pc_full_path, points)
                        else:
                            ### save in pcd format
                            pc_name = f'sample_{sample_num}.pcd'
                            pc_full_path = os.path.join(save_path, pc_name)
                            assert(not os.path.exists(pc_full_path))

                            # if name=="truck" and sample_num==448:
                            #     print("Problem: ", points.shape)
                            #     assert(1==0)

                            pcd = open3d.geometry.PointCloud()
                            print("$$$$$$$$$$$$$$$$ poitns shape: ", points.shape)
                            pcd.points = open3d.utility.Vector3dVector(np.array(points))
                            open3d.io.write_point_cloud(pc_full_path, pcd)

                        allocentric_dict[name][0].append(alpha)
                        allocentric_dict[name][1].append(pc_name)
                        allocentric_dict[name][2].append(gamma)
                        allocentric_dict[name][3].append(box)
                        allocentric_dict[name][4].append(center3D)
                        allocentric_dict[name][5].append(kitti_boxes_list[i])
                        allocentric_dict[name][6].append(lidar_sample_token)
                        allocentric_dict[name][7].append(obj_ann_token_list[i])
                        allocentric_dict[name][8].append(is_train)
                        allocentric_dict[name][9].append(sample_records)
                        allocentric_dict[name][10].append(obj_ann_info_list[i])
                        allocentric_dict[name][11].append(is_good_completion)
                        allocentric_dict[name][12].append(np.prod(box.wlh))


                        assert(len(allocentric_dict[name][0])==len(allocentric_dict[name][1])==len(allocentric_dict[name][2])==len(allocentric_dict[name][3])==len(allocentric_dict[name][4])==len(allocentric_dict[name][5])==len(allocentric_dict[name][6]))

                        key = (name, pc_name)
                        if key not in sample_dict:
                            # First row contains the allocentric angles and 
                            # the second rows contains the category name of the corresponding point clouds, 
                            # third row contains the viewing angles
                            # forth row contains the bounding boxes
                            # fifth row contains the 3D position of the boxes's center
                            sample_dict[key] = [[],[],[],[],[],[], [],[], [],[], [],[],[]]
                        else:
                            print(key)
                            assert(1==0)

                        
                        sample_dict[key][0].append(alpha)
                        sample_dict[key][1].append(name)
                        sample_dict[key][2].append(gamma)
                        sample_dict[key][3].append(box)
                        sample_dict[key][4].append(center3D)
                        sample_dict[key][5].append(kitti_boxes_list[i])
                        sample_dict[key][6].append(lidar_sample_token)
                        sample_dict[key][7].append(obj_ann_token_list[i])
                        sample_dict[key][8].append(is_train)
                        sample_dict[key][9].append(sample_records)
                        sample_dict[key][10].append(obj_ann_info_list[i])
                        sample_dict[key][11].append(is_good_completion)
                        sample_dict[key][12].append(np.prod(box.wlh))
                        
                        if name=="bus" and is_train==False:
                            print("YESSSS, IT's BUS")
                            count_bus_val+=1
                            
                        print("????? BUS VAL", count_bus_val)


                        kitti_cam_box, kitti_lidar_box = kitti_boxes_list[i]


                        # # # # visualize bounding box and center
                        mask = points_in_box(box, points.T, wlh_factor = 1.0)
                        assert(np.all(mask))
                        # if np.sum(mask)<len(points) or k==263:
                        #     print(f"box center: {box.center}")
                        #     print(f"box orientation: {box.orientation}")
                        #     print(f"mean point: {np.mean(points, axis=0)}")
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

                        #     plt.scatter(center3D[0], center3D[1], s=20)
                        #     plt.scatter(points[:,0], points[:,1], s=10)
                        #     plt.show()


                        # ### visualize kitti
                        # kitti_to_nu_lidar_inv = Quaternion(axis=(0, 0, 1), angle=np.pi / 2).inverse
                        # kitti_to_nu_lidar_mat_inv = (np.array(kitti_to_nu_lidar_inv.rotation_matrix))
                        # kitti_points = np.matmul(kitti_to_nu_lidar_mat_inv, points.T).T

                        # plt.figure(figsize=(8, 6))
                        # plt.gca().set_aspect('equal')
                        # corners = kitti_lidar_box.corners() #(3,8)
                        # corner_1 = corners[:,0][:2]
                        # corner_2 = corners[:,3][:2]
                        # corner_5 = corners[:,5][:2]
                        # corner_6 = corners[:,6][:2]
                        # rect = patches.Polygon([corner_1, corner_2, corner_6,corner_5], linewidth=1, edgecolor='r', facecolor='none')
                        # # Get the current axes and plot the polygon patch
                        # plt.gca().add_patch(rect)
                        # plt.scatter(kitti_points[:,0], kitti_points[:,1], s=10)
                        # plt.show()


                        # # visualized unnormalized points
                        ##if name=="car":
                        # print("visualize unnormalized")
                        # pcd = open3d.geometry.PointCloud()
                        # pcd.points = open3d.utility.Vector3dVector(np.array(points))
                        # pcd_colors = np.tile(np.array([[0,0,1]]), (len(points), 1))
                        # pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
                        # car_vis_pos = open3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                        # car_vis_pos.paint_uniform_color([1,0,0])  
                        # car_vis_pos.translate(tuple(center3D))
                        # open3d.visualization.draw_geometries([pcd, car_vis_pos]) 

                        # print("visualize normalized")
                        # print(f"num points: {len(points_normalized)}")
                        # # visualized normalized points
                        # pcd = open3d.geometry.PointCloud()
                        # pcd.points = open3d.utility.Vector3dVector(np.array(points_normalized))
                        # pcd_colors = np.tile(np.array([[0,0,1]]), (len(points_normalized), 1))
                        # pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
                        # car_vis_pos = open3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                        # car_vis_pos.paint_uniform_color([1,0,0])  
                        # car_vis_pos.translate(tuple([0,0,0]))
                        # open3d.visualization.draw_geometries([pcd, car_vis_pos]) 

            print("count_bus in val: ", count_bus_val)
            #assert(count_bus_val!=0)

            allocentric_full_path = os.path.join(args.pc_path, "allocentric.pickle")
            with open(allocentric_full_path, 'wb') as handle:
                pickle.dump(allocentric_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            sample_full_path = os.path.join(args.pc_path, "sample_dict.pickle")
            with open(sample_full_path, 'wb') as handle:
                pickle.dump(sample_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            assert(('bus', 'sample_0.pcd') in sample_dict)
        
            print(f"statistics of bounding boxes")
            front, side, vert = dataset.box_length_stats["front"], dataset.box_length_stats["side"], dataset.box_length_stats["vertical"]
            print(f"---front={front}, side={side}, vertical={vert}")
            front = np.array(front)
            side = np.array(side)
            vert = np.array(vert)
            front_mean = np.mean(front)
            front_std = np.sqrt(np.var(front))
            side_mean = np.mean(side)
            side_std = np.sqrt(np.var(side))
            vert_mean = np.mean(vert)
            vert_std = np.sqrt(np.var(vert))
            print(f"### MEAN: front={front_mean}, side={side_mean}, vert={vert_mean}")
            print(f"### STD: front={front_std}, side={side_std}, vert={vert_std}")

            z_coords = dataset.box_length_stats["z_coord"]
            z_coords = np.array(z_coords)
            print(f"### MEAN: z:{np.mean(z_coords)}")
            print(f"### STD: z:{np.sqrt(np.var(z_coords))}")

            # print("HELLLLLO .......")
            # with open(os.path.join(args.pc_path , "allocentric.pickle"), 'rb') as handle:
            #     allocentric_load = pickle.load(handle)

            # for name in vehicle_names.values():
            #     print(name)
            #     assert(len(allocentric_load[name][3])==len(allocentric_load[name][1]))
            #     # for i in range(len(allocentric_load[name][3])):
            #     #     print(allocentric_dict[name][1][i])
            #     #     assert(np.all(points_in_box(allocentric_dict[name][3][i], np.asarray(open3d.io.read_point_cloud(allocentric_dict[name][1][i]).points).T, wlh_factor = 1.0)))
            #     #  for P in range(4):
            #     #     print("P: ", P)
            #     #     assert(len(allocentric_dict[name][P])==len(allocentric_load[name][P]))
            #     #     for i in range(len(allocentric_load[name][P])):
            #     #         assert(allocentric_dict[name][P][i]==allocentric_load[name][P][i])


    else:
        print(">>> SHOWING DENSE POINTS")
        # assuming the densified point cloud is saved to dense_path
        dense_path = os.path.join(args.pc_path, "dense_nusc")
        categories = os.listdir(dense_path)

        for category in categories:
            print(f"##### category={category}")
            category="truck"#"car"
            dense_full_path = os.path.join(dense_path, category)
            pc_files = os.listdir(dense_full_path)
            for pc_file in pc_files:
                print("----PC FILE: ", pc_file)
                if not save_as_pcd:
                    dense_points = np.load(os.path.join(dense_full_path, pc_file))
                    print(f"dense num_points: {dense_points.shape[0]}")
                    sparse_points = np.load(os.path.join(args.pc_path, category, pc_file))
                    print(f"sparse num_points: {sparse_points.shape[0]}")
                else:
                    dense_points = open3d.io.read_point_cloud(os.path.join(dense_full_path, pc_file)).points
                    sparse_points = open3d.io.read_point_cloud(os.path.join(args.pc_path, category, pc_file)).points

                dense_pcd = open3d.geometry.PointCloud()
                dense_pcd.points = open3d.utility.Vector3dVector(np.array(dense_points))
                pcd_colors = np.tile(np.array([[0,0,1]]), (len(dense_points), 1))
                dense_pcd.colors = open3d.utility.Vector3dVector(pcd_colors)

                sparse_pcd = open3d.geometry.PointCloud()
                sparse_pcd.points = open3d.utility.Vector3dVector(np.array(sparse_points))
                pcd_colors = np.tile(np.array([[1,0,0]]), (len(sparse_points), 1))
                sparse_pcd.colors = open3d.utility.Vector3dVector(pcd_colors)

                open3d.visualization.draw_geometries([sparse_pcd, dense_pcd.translate((10,0,0))]) 


                # mat = open3d.visualization.rendering.MaterialRecord()
                # mat.shader = 'defaultUnlit'
                # mat.point_size = 4.0
                # open3d.visualization.draw([{'name': 'pcd', 'geometry': dense_pcd, 'material': mat}, {'name': 'pcd', 'geometry': sparse_pcd.translate((1,0,0)), 'material': mat}], show_skybox=False)




            