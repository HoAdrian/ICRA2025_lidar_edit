# import copy
# import os
# import numpy as np
# import argparse

# os.system(f"pwd")

# import sys
# sys.path.append("./")
# sys.path.append("./datasets")
# sys.path.append("./models")
# from datasets.data_utils import *
# from datasets.data_utils_nuscenes import pyquaternion_from_angle, get_obj_regions
# from datasets.dataset import Voxelizer, PolarDataset, collate_fn_BEV
# from datasets.dataset_nuscenes import Nuscenes, NuscenesForeground
# import torch
# import configs.nuscenes_config as config

# from models.vqvae_transformers import VQVAETrans, voxels2points
# import open3d
# import pickle

# from pyquaternion import Quaternion
# from shapely.geometry import Polygon
# from nuscenes.utils.geometry_utils import points_in_box

# '''
# utilities for actor insertion
# '''

# def kitti2nusc(vehicle_pc):
#     '''
#     convert from kitti coordinates to nuscenes coordinates
#     vehicle_pc: (N,3)
#     '''
#     kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
#     kitti_to_nu_lidar_mat = (np.array(kitti_to_nu_lidar.rotation_matrix))
#     vehicle_pc = np.matmul(kitti_to_nu_lidar_mat, vehicle_pc.T).T
#     return vehicle_pc

# def insert_vehicle_pc(vehicle_pc, bbox, insert_xyz_pos, rotation_angle, voxels_occupancy_has, voxelizer, dataset, mode='spherical', use_ground_seg=True, center=None, kitti2nusc=False, use_dense=True):
#     '''
#     insert a completeed vehicle_pc into the scene point cloud

#     - vehicle_pc: (N,3)
#     - Nuscenes bounding box of the vehicle_pc
#     - insert_xyz_pos: (3,), first two elements are the x-y pos to place the vehicle, the third element can be any value because the z coordinate for insertion is going to be determined in this method.
#     - rotation_angle: the angle in radian to rotate the point cloud to align orientations
#     - voxels_occupancy_has: (1, #r, #theta, #z), occupancy grid of the scene
#     - voxelizer: datasets.Voxelizer
#     - dataset: datasets.PolarDataset, assuming I am using datasets.NuscenesForeground as the dataset.point_cloud_dataset
#     - mode: either "polar" or "spherical
#     - use_ground_seg: use ground segmentation or not
#     - center: center the point cloud at the center or not, either None or (3,) shape array
#     - kitti2nusc: whether vehicle_pc is from kitti
#     - use_dense: whether to apply occlusion to the vehicle as well

#     return:
#     new_scene_points_xyz, new_bbox, insert_xyz_pos, vehicle_pc
#     '''
   
#     # pass object variables by copy
#     new_bbox = copy.deepcopy(bbox)
#     vehicle_pc = np.copy(vehicle_pc)
#     insert_xyz_pos = np.copy(insert_xyz_pos)
#     voxels_occupancy_has = torch.clone(voxels_occupancy_has)
    

#     if kitti2nusc:
#         kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
#         kitti_to_nu_lidar_mat = (np.array(kitti_to_nu_lidar.rotation_matrix))
#         vehicle_pc = np.matmul(kitti_to_nu_lidar_mat, vehicle_pc.T).T

#     ##### We have assumed that the vehicle is centered at its centroid defined by its bounding box
#     if center is not None:
#         vehicle_pc = vehicle_pc-center

#     # ### We have to align the vehicle_pc orientation / edit the allocentric angle by rotation_angle
#     vehicle_pc = cart2polar(vehicle_pc, mode=mode)
#     theta = vehicle_pc[:,1]
#     theta = theta + rotation_angle
#     theta[theta<0] += 2*np.pi
#     theta = theta%(2*np.pi)
#     vehicle_pc[:,1] = theta
#     vehicle_pc = polar2cart(vehicle_pc, mode=mode)

#     #### shift vehicle to the desired position
#     vehicle_pc[:,:2] += insert_xyz_pos[:2]
    
#     #### get the non-empty grid nearest to the vehicle in BEV, and set its lowest z-value as the lowest z-value of the vehicle
#     #start_time = timeit.default_timer()
#     if not use_ground_seg:
#         print("^^^^^^ just using nearest occupied voxels to determine z")
#         nearest_polar_voxels_pos = voxelizer.get_nearest_occupied_BEV_voxel(voxels_occupancy_has[0].cpu().detach().numpy(), cart2polar(insert_xyz_pos[np.newaxis,:], mode=mode), mode=mode) #(M,3)
#     else:
#         print(f"##### USING GROUND SEGMENTATION to determine z........")
#         ground_points = dataset.point_cloud_dataset.ground_points[:,:3] #(G,3)
#         ground_idxs = voxelizer.get_grid_ind(cart2polar(ground_points, mode=mode)) #(G,3)
#         ground_occupancy = np.zeros_like(voxels_occupancy_has[0].cpu().detach().numpy())
#         ground_occupancy[ground_idxs[:,0], ground_idxs[:,1], ground_idxs[:,2]] = 1.0
#         #nearest_polar_voxels_pos = voxelizer.get_nearest_ground_BEV_voxel(ground_occupancy, cart2polar(insert_xyz_pos[np.newaxis,:], mode=mode), mode)
#         nearest_polar_voxels_pos = voxelizer.get_nearest_ground_BEV_pos(ground_points, cart2polar(insert_xyz_pos[np.newaxis,:], mode=mode), mode)
#     #mid_time = timeit.default_timer()
        
#    # print(f"!!!!! time for getting ground segmentation z: {mid_time - start_time} seconds")

#     nearest_cart_voxels_pos = polar2cart(nearest_polar_voxels_pos, mode=mode)
#     nearest_min_z = np.min(nearest_cart_voxels_pos[:,2])
#     vehicle_min_z = np.min(vehicle_pc[:,2])
#     height_diff = nearest_min_z - vehicle_min_z
#     vehicle_pc[:,2] += height_diff
#     insert_xyz_pos[2]=height_diff

#     #insert_xyz_pos[2]-=0.1#1.0

#     ############ transform the bounding box accordingly
#     new_bbox.translate(-bbox.center)
#     new_bbox.rotate(pyquaternion_from_angle(rotation_angle))
#     new_bbox.translate(insert_xyz_pos)

#     #### project to spherical grid, apply occlusion and convert back to point cloud
#     polar_vehicle = cart2polar(vehicle_pc, mode=mode)     
#     #obj_region = get_obj_regions([new_bbox], mode=mode)[0]
#     #old_occupancy = voxels_occupancy_has[0].cpu().detach().numpy()
#     #new_occupancy[voxelizer.create_mask_by_occlusion(obj_region, use_z=True)==1] = 0
#     if use_dense:
#         new_occupancy = voxelizer.voxelize_and_occlude(voxels_occupancy_has[0].cpu().detach().numpy(), polar_vehicle, insert_only=False)
#     else:
#         new_occupancy,_ = voxelizer.voxelize_and_occlude_2(voxels_occupancy_has[0].cpu().detach().numpy(), polar_vehicle, add_vehicle=True, use_margin=True)

   
#     new_scene_points_xyz = voxels2points(voxelizer, voxels=torch.tensor(new_occupancy).permute(2,0,1).unsqueeze(0), mode=mode)[0]

#     return new_scene_points_xyz, new_occupancy, new_bbox, insert_xyz_pos, vehicle_pc


# def is_overlap_shapely(box1, box2):
#     if isinstance(box2, list):
#         poly1 = Polygon(box1)
#         for box in box2:
#             poly2 = Polygon(box)
#             if poly1.intersects(poly2):
#                 return True
#         return False

#     poly1 = Polygon(box1)
#     poly2 = Polygon(box2)
#     return poly1.intersects(poly2)

# import cv2
# def is_overlap_opencv(box1, box2):
#     if isinstance(box2, list):
#         rect1 = cv2.minAreaRect(box1)
#         for box in box2:
#             rect2 = cv2.minAreaRect(box)
#             intersection = cv2.rotatedRectangleIntersection(rect1, rect2)
#             if intersection[0] != cv2.INTERSECT_NONE:
#                 return True
#         return False
#     rect1 = cv2.minAreaRect(box1)
#     rect2 = cv2.minAreaRect(box2)
#     intersection = cv2.rotatedRectangleIntersection(rect1, rect2)
#     return intersection[0] != cv2.INTERSECT_NONE

# def sample_valid_insert_pos(name, current_viewing_angle, dataset, insert_box, other_boxes):
#     '''
#     find the pos to insert the box such that it is not colliding with other boxes and it does not contain any other background points except drivable surface
#     - name: name of the vehicle being inserted: (car, bus or truck)
#     - current_viewing_angle: the viewing angle of the vehicle's bounding box
#     - dataset: PolarDataset
#     - insert_box: the bounding box of the inserted vehicle
#     - other_boxes: a list of other bounding boxes
#     '''
#     sparse_ground_points = np.copy(dataset.point_cloud_dataset.sparse_ground_points[:,:3]) #(G,3)
#     if name=="car":
#         sparse_ground_points = sparse_ground_points[np.linalg.norm(sparse_ground_points, axis=1)>5]
#     elif name=="bus":
#         sparse_ground_points = sparse_ground_points[np.linalg.norm(sparse_ground_points, axis=1)>15]
#     elif name=="truck":
#         sparse_ground_points = sparse_ground_points[np.linalg.norm(sparse_ground_points, axis=1)>10]
    
#     other_boxes = [(copy.deepcopy(other_box).corners().T[[0,1,5,4], :2]).astype(np.float32) for other_box in other_boxes]
#     target_idx = -1
#     ### pick a random allocentric angle offset
#     #rand_alpha=np.random.uniform(low=0.0, high=2*np.pi)
#     alphas = np.linspace(start=0.0, stop=2*np.pi, num=20)
#     target_rand_alpha = 0
#     # remember to shuffle
#     np.random.shuffle(sparse_ground_points)
#     np.random.shuffle(alphas)
#     got_candidate = False
#     for ground_point_idx in range(sparse_ground_points.shape[0]):
#         for rand_alpha in alphas:
#             insert_box_copy = copy.deepcopy(insert_box)
#             ground_point = sparse_ground_points[ground_point_idx, :]

#             # simulate moving the box there
#             desired_viewing_angle = compute_viewing_angle(ground_point[:2])
#             rotation_align = -(desired_viewing_angle - current_viewing_angle) # negative sign because gamma increases clockwise
#             rotation_align-=rand_alpha
#             insert_box_copy.translate(-insert_box_copy.center)
#             insert_box_copy.rotate(pyquaternion_from_angle(rotation_align))
#             insert_box_copy.translate(ground_point)

#             insert_box_2d = (insert_box_copy.corners().T[[0,1,5,4], :2]).astype(np.float32)
#             assert(isinstance(insert_box_2d, np.ndarray))

#             if np.sum(points_in_box(insert_box_copy, dataset.point_cloud_dataset.other_background_points[:,:3].T, wlh_factor = 1.0))>0:
#                 continue
            
#             if len(other_boxes)!=0:
#                 if not is_overlap_shapely(insert_box_2d, other_boxes):
#                     target_idx = ground_point_idx
#                     got_candidate=True
#                     target_rand_alpha = rand_alpha
#                     break
#             else:
#                 target_idx = np.random.randint(low=0, high=len(sparse_ground_points))
#                 got_candidate=True
#                 target_rand_alpha = 0
#                 break
#         # exit the loop if we have found a valid insert pos
#         if got_candidate:
#             break

    
#     if target_idx==-1:
#         print("WARNING: no valid insert pos")
#         return None
        

#     insert_xyz_pos = sparse_ground_points[target_idx]
#     #dataset.point_cloud_dataset.sparse_ground_points = dataset.point_cloud_dataset.sparse_ground_points[np.arange(len(sparse_ground_points))!=target_idx] #remove the random ground points
#     print(f"+++++ insert_xyz_pos: {insert_xyz_pos}")

#     return insert_xyz_pos, target_rand_alpha

# from datasets.dataset_nuscenes import vehicle_names, plot_obj_regions
# import timeit

# def angles_from_box(box):
#     '''
#     Get the viewing angle and allocentric angle of the box
#     '''
#     corners = box.corners()
#     center2D = box.center[:2]
#     corner1, corner2 = corners[:,0][:2], corners[:,1][:2]  # top front corners (left and right)
#     corner7, corner6 = corners[:,6][:2], corners[:,5][:2] # bottom back corner (right), top back corner (right)
#     center2D = (corner1 + corner6)/2

#     center3D = np.array([center2D[0], center2D[1], (corners[2,2]+corners[2,1])/2])

#     right_pointing_vector = (corner2 + corner6)/2.0 - center2D
#     front_pointing_vector = (corner1 + corner2)/2.0 - center2D
#     obj2cam_vector = -center2D
#     #print(np.arccos(np.dot(right_pointing_vector/np.linalg.norm(right_pointing_vector), front_pointing_vector/np.linalg.norm(front_pointing_vector)))/np.pi*180)
    
#     ### compute allocentric angle and viewing angle gamma
#     alpha = compute_allocentric_angle(obj2cam_pos=obj2cam_vector, obj_right_axis=right_pointing_vector, obj_front_axis=front_pointing_vector)
#     gamma = compute_viewing_angle(-obj2cam_vector)

#     return alpha, gamma, center3D


# from scipy.spatial import KDTree

# def insertion_vehicles_driver(allocentric_dict, voxels_occupancy_has, names, dataset, voxelizer, token2sample_dict, args, save_lidar_path, mode="spherical"):
#     '''
#     Driver method of inserting vehicle
#     - allocentric dict: contains all information of the completed point clouds
#     - voxels_occupancy_has: (1, #r, #theta, #z), occupancy grid of the scene
#     #- names: list of sampled vehicle names
#     - dataset: PolarDataset
#     - voxelizer: Voxelizer
#     - token2sample_dict: modify this in place and save it, maps from lidar sample token to generated data we will use
#     - voxels_occupancy_has: occupancy grid of shape (1,#r, #theta, #z), make a copy and modify out of place
#     - save_lidar_path: the full path to the where the data is save
#     - mode: "spherical"
#     '''
#     assert(mode=="spherical")
#     assert(names is None)

#     # make a copy of the voxel occupancy
#     voxels_occupancy_has = torch.clone(voxels_occupancy_has)

#     new_bboxes = []
#     new_obj_ann_token_list = []
#     new_ann_info_list=[]

#     #(obj_point_cloud_list, obj_name_list, points_3D, obj_allocentric_list, obj_centers_list, obj_boxes_list, obj_gamma_list, kitti_boxes_list, lidar_sample_token, boxes, obj_ann_token_list, sample_annotation_tokens, sample_records, obj_ann_info_list, curr_sample_table_token)
#     dataset_obj_boxes_list = dataset.obj_properties[5][:3] #5
#     original_box_idxs = [i for i, box in enumerate(dataset_obj_boxes_list) if (box.name in vehicle_names and vehicle_names[box.name] in {"bus", "car", "truck"})]
#     dataset_obj_ann_token_list = [dataset.obj_properties[10][i] for i in original_box_idxs] #10
#     original_vehicle_boxes = [dataset_obj_boxes_list[i] for i in original_box_idxs]

#     names = [vehicle_names[box.name] for box in original_vehicle_boxes]
#     assert(len(dataset_obj_ann_token_list)==len(original_vehicle_boxes))
#     print(f"##### insert vehicle name: {names}")


#     kd_tree = KDTree(dataset.points_xyz[:,:3])

#     # prevent error when not entering the loop (empty names)
#     new_scene_points_xyz = dataset.points_xyz[:,:3]
    
#     print("length_names", len(names))

#     # iterate over each object we want to insert
#     for i, name in enumerate(names):
        
#         obj_properties = allocentric_dict[name] # the properties of objects belonging to the specified name
#         allocentric_angles = (obj_properties[0]) #np.array(obj_properties[0])
#         pc_filenames = obj_properties[1]
#         viewing_angles = obj_properties[2] #np.array(obj_properties[2])
#         boxes = obj_properties[3]
#         center3Ds = obj_properties[4] #np.array(obj_properties[4])
#         obj_ann_tokens = obj_properties[7]
#         is_trains = obj_properties[8]
#         ann_info_list = obj_properties[10]
#         obj_lidar_sample_token_list = obj_properties[6]
#         # is_good_completion_indicator_list = obj_properties[11]
#         box_volume_list = obj_properties[12]
#         print(name)

#         N = len(allocentric_angles)
#         assert(len(allocentric_angles)==len(pc_filenames)==len(viewing_angles)==len(boxes)==len(center3Ds))

#         ####### choose an object
#         original_box = original_vehicle_boxes[i]
#         original_volume = np.prod(original_box.wlh)
#         chosen_idx = np.argmin(np.abs((box_volume_list)-original_volume))


#         pc_filename = pc_filenames[chosen_idx]
#         bbox = boxes[chosen_idx]
#         center3D = center3Ds[chosen_idx]
#         pc_path = os.path.join(args.pc_path, name)
#         new_obj_ann_token = obj_ann_tokens[chosen_idx]
#         new_ann_info = ann_info_list[chosen_idx]

#         use_dense = args.dense==1
#         #if use_dense and is_good_completion_indicator_list[chosen_idx]:
#         if use_dense:
#             pc_full_path = os.path.join(args.pc_path, "dense_nusc", name, pc_filename)
#         else:
#             print("using incomplete point cloud......")
#             pc_full_path = os.path.join(pc_path, pc_filename)
#             #raise Exception("I prefer completed point cloud LOL")
#         vehicle_pc = np.asarray(open3d.io.read_point_cloud(pc_full_path).points) #np.load(pc_full_path)

#         ###### insert it right at the position of the original box in the scene before foreground removal
#         # original_box = original_vehicle_boxes[i]
#         original_alpha, original_gamma, original_center = angles_from_box(original_box)
#         insert_xyz_pos = original_center.reshape(-1)
#         insert_xyz_pos[2]-= float(original_box.wlh[2])/2.0 # bottom of the box center


#         desired_viewing_angle = original_gamma
#         desired_allocentric_angle = original_alpha
#         current_viewing_angle = viewing_angles[chosen_idx]
#         current_allocentric_angle = allocentric_angles[chosen_idx]
#         rotation_align = -(desired_viewing_angle - current_viewing_angle) # negative sign because gamma increases clockwise
#         # align allocentric as well
#         rotation_align-= (desired_allocentric_angle - current_allocentric_angle)

#         new_scene_points_xyz, new_occupancy, new_bbox, insert_xyz_pos, vehicle_pc = insert_vehicle_pc(vehicle_pc, bbox, insert_xyz_pos, rotation_align, voxels_occupancy_has, voxelizer, dataset, mode=mode, use_ground_seg=True, center=center3D, kitti2nusc=False, use_dense=use_dense)
        
#         voxels_occupancy_has = torch.tensor(new_occupancy).unsqueeze(0)
#         new_bboxes.append(new_bbox)
#         new_obj_ann_token_list.append(new_obj_ann_token)
#         new_ann_info_list.append(new_ann_info)

    
#     ## get nearest neighbor intensity
#     original_points = dataset.points_xyz #(N,5)
#     _, nearest_idxs = kd_tree.query(new_scene_points_xyz[:,:3], k=1)
#     # make the point dim be 5
#     extras = np.zeros((len(new_scene_points_xyz), 2))
#     new_points_xyz = np.concatenate((new_scene_points_xyz, extras), axis=1)
#     assert(new_points_xyz.shape[-1]==5)

#     # new_point_in_box_mask = np.ones((len(new_points_xyz),))==0
#     # for b_idx, new_box in enumerate(new_bboxes):
#     #     original_mean_intensity = original_vehicle_mean_intensities[i]
#     #     mask = points_in_box(new_box, new_points_xyz[:,:3].T, wlh_factor = 1.0)
#     #     new_point_in_box_mask = new_point_in_box_mask | mask
#     #     new_points_xyz[mask, 3] = original_mean_intensity
    
#     # new_points_xyz[np.logical_not(new_point_in_box_mask), 3] = mean_original_background_intensity

#     new_points_xyz[:,3] = original_points[:,3][nearest_idxs].astype(np.float64)
#     #new_points_xyz[:,3] = 0.0 #original_points[:,3][nearest_idxs].astype(np.float64)
#     new_points_xyz[:,4] = 0.0 #original_points[0,4]

#     points_xyz = new_scene_points_xyz
#     bounding_boxes = new_bboxes
#     lidar_sample_token = dataset.lidar_sample_token

#     return new_scene_points_xyz, new_bboxes, token2sample_dict, voxels_occupancy_has, original_vehicle_boxes



# def insertion_vehicles_driver_random(voxels_occupancy_has, names, dataset, voxelizer, token2sample_dict, args, save_lidar_path, mode="spherical"):
#     '''
#     Driver method of inserting vehicle with everything random
#     - voxels_occupancy_has: (1, #r, #theta, #z), occupancy grid of the scene
#     #- names: list of sampled vehicle names
#     - dataset: PolarDataset
#     - voxelizer: Voxelizer
#     - token2sample_dict: modify this in place and save it, maps from lidar sample token to generated data we will use
#     - voxels_occupancy_has: occupancy grid of shape (1,#r, #theta, #z), make a copy and modify out of place
#     - save_lidar_path: the full path to the where the data is save
#     - mode: "spherical"
#     '''
#     assert(mode=="spherical")

#     # make a copy of the voxel occupancy
#     voxels_occupancy_has = torch.clone(voxels_occupancy_has)

#     new_bboxes = []
#     new_obj_ann_token_list = []
#     new_ann_info_list=[]

#     # iterate over each object we want to insert
#     for i, name in enumerate(names):
#         with open(os.path.join(args.pc_path, "allocentric.pickle"), 'rb') as handle:
#             allocentric_dict = pickle.load(handle)

#         obj_properties = allocentric_dict[name] # the properties of objects belonging to the specified name
#         allocentric_angles = np.array(obj_properties[0])
#         pc_filenames = obj_properties[1]
#         viewing_angles = np.array(obj_properties[2])
#         boxes = obj_properties[3]
#         center3Ds = np.array(obj_properties[4])
#         obj_ann_tokens = obj_properties[7]
#         is_trains = obj_properties[8]
#         ann_info_list = obj_properties[10]
#         print(name)

#         N = len(allocentric_angles)
#         assert(len(allocentric_angles)==len(pc_filenames)==len(viewing_angles)==len(boxes)==len(center3Ds))

#         ####### choose an object
#         chosen_idx = np.random.randint(low=0, high=N)
        
#         pc_filename = pc_filenames[chosen_idx]
#         bbox = boxes[chosen_idx]
#         center3D = center3Ds[chosen_idx]
#         pc_path = os.path.join(args.pc_path, name)
#         new_obj_ann_token = obj_ann_tokens[chosen_idx]
#         new_ann_info = ann_info_list[chosen_idx]

#         use_dense = args.dense==1
#         if use_dense:
#             pc_full_path = os.path.join(args.pc_path, "dense_nusc", name, pc_filename)
#         else:
#             pc_full_path = os.path.join(pc_path, pc_filename)
#             #raise Exception("I prefer completed point cloud LOL")
#         vehicle_pc = np.asarray(open3d.io.read_point_cloud(pc_full_path).points) #np.load(pc_full_path)

#         ###### choose where to insert vehicle
#         valid_pos_info = sample_valid_insert_pos(name, viewing_angles[chosen_idx], dataset, bbox, new_bboxes)
        
#         if valid_pos_info is not None:
#             insert_xyz_pos, rand_alpha = valid_pos_info
#         else:
#             print("warning: skip this vehicle")
#             continue

#         desired_viewing_angle = compute_viewing_angle(insert_xyz_pos[:2])
#         current_viewing_angle = viewing_angles[chosen_idx]
#         rotation_align = -(desired_viewing_angle - current_viewing_angle) # negative sign because gamma increases clockwise
#         # random allocentric offset
#         rotation_align-=rand_alpha

#         new_scene_points_xyz, new_occupancy, new_bbox, insert_xyz_pos, vehicle_pc = insert_vehicle_pc(vehicle_pc, bbox, insert_xyz_pos, rotation_align, voxels_occupancy_has, voxelizer, dataset, mode=mode, use_ground_seg=True, center=center3D, kitti2nusc=False, use_dense=use_dense)
        
        
#         voxels_occupancy_has = torch.tensor(new_occupancy).unsqueeze(0)
#         new_bboxes.append(new_bbox)
#         new_obj_ann_token_list.append(new_obj_ann_token)
#         new_ann_info_list.append(new_ann_info)

#     ### remove new bboxes that contain no points after applying occlusion
#     new_bboxes_copy = [copy.deepcopy(box) for box in new_bboxes]
#     new_obj_ann_token_list_copy = [ann_token for ann_token in new_obj_ann_token_list]
#     new_ann_info_list_copy = [copy.deepcopy(ann_info) for ann_info in new_ann_info_list]
#     new_bboxes = []
#     new_obj_ann_token_list = []
#     new_ann_info_list = []
#     for i, box in enumerate(new_bboxes_copy):
#         mask = points_in_box(box, new_scene_points_xyz.T, wlh_factor = 1.0)
#         if np.sum(mask)!=0: # and np.sum(mask)>=50:
#             new_bboxes.append(box)
#             new_obj_ann_token_list.append(new_obj_ann_token_list_copy[i])
#             new_ann_info_list.append(new_ann_info_list_copy[i])
#     assert(len(new_bboxes)==len(new_obj_ann_token_list)==len(new_ann_info_list))
    
#     ## get nearest neighbor intensity
#     original_points = dataset.points_xyz #(N,5)
#     nearest_idxs = np.argmin(np.linalg.norm(new_scene_points_xyz[:,:3][np.newaxis,...] - original_points[:,:3][:,np.newaxis, :], axis=-1), axis=0) #(1,M,3) - (N,1,3) = (N,M,3) => (M,3)
#     # make the point dim be 5
#     extras = np.zeros((len(new_scene_points_xyz), 2))
#     new_points_xyz = np.concatenate((new_scene_points_xyz, extras), axis=1)
#     new_points_xyz[:,3] = original_points[:,3][nearest_idxs].astype(np.float64)
#     new_points_xyz[:,4] = 0.0 #original_points[0,4]

#     points_xyz = new_scene_points_xyz
#     bounding_boxes = new_bboxes
#     lidar_sample_token = dataset.lidar_sample_token

#     return new_scene_points_xyz, new_bboxes, token2sample_dict, voxels_occupancy_has






# def save_reconstruct_data(rec_voxels_occupancy_has, dataset, voxelizer, token2sample_dict, args, save_lidar_path, mode="spherical"):
#     '''
#     Just save the reconstructed voxels with foreground points and the corresponding bounding boxes, annotations and sample_records
    
#     - voxels occupancy has: shape (1, #z, #r, #theta)
#     '''
#     new_scene_points_xyz = voxels2points(voxelizer, rec_voxels_occupancy_has, mode=mode)[0]
#     bounding_boxes = dataset.obj_properties[9]
#     lidar_sample_token = dataset.lidar_sample_token
#     sample_records = dataset.obj_properties[12]
#     obj_ann_token_list = dataset.obj_properties[11]
    
#      ## get nearest neighbor intensity
#     original_points = dataset.points_xyz #(N,5)
#     nearest_idxs = np.argmin(np.linalg.norm(new_scene_points_xyz[:,:3][np.newaxis,...] - original_points[:,:3][:,np.newaxis, :], axis=-1), axis=0) #(1,M,3) - (N,1,3) = (N,M,3) => (M,3)
#     # make the point dim be 5
#     extras = np.zeros((len(new_scene_points_xyz), 2))
#     new_points_xyz = np.concatenate((new_scene_points_xyz, extras), axis=1)
#     new_points_xyz[:,3] = original_points[:,3][nearest_idxs].astype(np.float64)
#     new_points_xyz[:,4] = 0.0 #original_points[0,4]
    
#     ############# save point cloud 
#     pc_name = f'{args.split}_{lidar_sample_token}.bin'
#     os.makedirs(os.path.join(save_lidar_path, "lidar_point_clouds"), exist_ok=True)
#     lidar_full_path = os.path.join(save_lidar_path, "lidar_point_clouds", pc_name)
#     assert(not os.path.exists(lidar_full_path))
#     new_points_xyz.astype(np.float32).tofile(lidar_full_path)

#     ############## Save the data needed to build the new database
#     token2sample_dict[lidar_sample_token] = (lidar_full_path, bounding_boxes, obj_ann_token_list, sample_records)

#     token2sample_dict_full_path = os.path.join(save_lidar_path, "token2sample.pickle")
#     with open(token2sample_dict_full_path, 'wb') as handle:
#         pickle.dump(token2sample_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#     return new_scene_points_xyz, bounding_boxes

    


# def copy_and_paste_method(voxels_occupancy_has, dataset, voxelizer):
#     '''
#     Generate foreground object-free point cloud by a naive copy and paste method
#     voxels_occupancy_has: (1, C, H, W) or equivalently (1, #z, #r, #theta)
#     dataset: PolarDataset
#     voxelizer: Voxelizer

#     return:
#     generated occupancy grid of shape (1, C, H, W)
#     '''
#     gen_binary_voxels = torch.clone(voxels_occupancy_has.permute(0,2,3,1)[0].detach().cpu())
#     for mask in dataset.obj_voxels_mask_list:
#         if np.sum(mask)==0:
#             continue
#         # print("mask sum: ", np.sum(mask))
#         gen_binary_voxels = voxelizer.copy_and_paste_neighborhood(gen_binary_voxels, voxels_mask=torch.tensor(mask)) #(H,W,in_chans)
#     gen_binary_voxels = gen_binary_voxels.unsqueeze(0).permute(0,3,1,2) #(1, C, H, W)
#     return gen_binary_voxels


# def count_vehicle_name_in_box_list(boxes, vehicle_names_dict):
#     '''
#     maps the raw vehicle name to the name we use for categorizing vehicles
#     '''
#     count = {v_name:0 for v_name in vehicle_names_dict.values()}
#     for box in boxes:
#         #print(box.name)
        
#         if box.name in vehicle_names_dict.keys():
#             name = vehicle_names_dict[box.name]
#             count[name]+=1
#     return count
            

# ###################################################################################################################################
# def perturb_2d(point, radius):
#     '''
#     point: (2,)
#     '''
#     theta = np.random.uniform(0, 2 * np.pi)
#     # Random radius within the given limit
#     r = np.random.uniform(0, radius)
#     r = 8.0
#     dx = r * np.cos(theta)
#     dy = r * np.sin(theta)

#     perturbed_point = point + np.array([dx, dy])
#     return perturbed_point


# def sample_perturb_insert_pos(original_insert_xyz, current_viewing_angle, dataset, insert_box, other_boxes):
#     '''
#     TODO: implement this
#     find the pos perturbed from an original position to insert the box such that it is not colliding with other boxes and it does not contain any other background points except drivable surface

#     - original_insert_xyz: original xyz-pos to insert the vehicle
#     - current_viewing_angle: the viewing angle of the vehicle's bounding box
#     - dataset: PolarDataset
#     - insert_box: the bounding box of the inserted vehicle
#     - other_boxes: a list of other bounding boxes
#     '''
#     other_boxes = [(copy.deepcopy(other_box).corners().T[[0,1,5,4], :2]).astype(np.float32) for other_box in other_boxes]
#     ### pick a random allocentric angle offset with 45 degrees
#     alphas = np.linspace(start=-np.pi/4, stop=np.pi/4, num=20)
#     target_rand_alpha = 0
#     # remember to shuffle
#     np.random.shuffle(alphas)
#     got_candidate = False
#     perturbed_insert_xyz_copy = np.copy(original_insert_xyz)

#     for num_try in range(10): # try perturb at most 10 times, if it does not succeed, return None
#         # sample a perturbed insert_pos
#         perturbed_insert_xyz_copy[:2] = perturb_2d(original_insert_xyz[:2], radius=2.5)

#         for rand_alpha in alphas:
#             #rand_alpha = np.pi/4
#             insert_box_copy = copy.deepcopy(insert_box)
    
#             # simulate moving the box there
#             desired_viewing_angle = compute_viewing_angle(perturbed_insert_xyz_copy[:2])
#             rotation_align = -(desired_viewing_angle - current_viewing_angle) # negative sign because gamma increases clockwise
#             rotation_align-=rand_alpha
#             insert_box_copy.translate(-insert_box_copy.center)
#             insert_box_copy.rotate(pyquaternion_from_angle(rotation_align))
#             insert_box_copy.translate(np.array([perturbed_insert_xyz_copy[0], perturbed_insert_xyz_copy[1], insert_box.center.reshape(-1)[2]]))

#             insert_box_2d = (insert_box_copy.corners().T[[0,1,5,4], :2]).astype(np.float32)
#             assert(isinstance(insert_box_2d, np.ndarray))

#             if np.sum(points_in_box(insert_box_copy, dataset.point_cloud_dataset.other_background_points[:,:3].T, wlh_factor = 1.0))>0:
#                 continue
            
#             if len(other_boxes)!=0:
#                 if not is_overlap_shapely(insert_box_2d, other_boxes):
#                     got_candidate=True
#                     target_rand_alpha = rand_alpha
#                     break
#             else:
#                 got_candidate=True
#                 target_rand_alpha = 0
#                 break
#         # exit the loop if we have found a valid insert pos
#         if got_candidate:
#             break

    
#     if not got_candidate:
#         print("WARNING: no valid insert pos")
#         return None
        

#     perturbed_insert_xyz_pos = perturbed_insert_xyz_copy
#     print(f"+++++ perturbed_insert_xyz_pos: {perturbed_insert_xyz_pos}")

#     return perturbed_insert_xyz_pos, target_rand_alpha


# def insertion_vehicles_driver_perturbed(inpainted_points, allocentric_dict, voxels_occupancy_has, names, dataset, voxelizer, token2sample_dict, args, save_lidar_path, mode="spherical"):
#     '''
#     Driver method of inserting vehicle with slightly perturbed position and allocentric angle


#     - allocentric dict: contains all information of the completed point clouds
#     - voxels_occupancy_has: (1, #r, #theta, #z), occupancy grid of the scene
#     #- names: list of sampled vehicle names
#     - dataset: PolarDataset
#     - voxelizer: Voxelizer
#     - token2sample_dict: modify this in place and save it, maps from lidar sample token to generated data we will use
#     - voxels_occupancy_has: occupancy grid of shape (1,#r, #theta, #z), make a copy and modify out of place
#     - save_lidar_path: the full path to the where the data is save
#     - mode: "spherical"
#     '''
#     assert(mode=="spherical")
#     assert(names is None)

#     # make a copy of the voxel occupancy
#     voxels_occupancy_has = torch.clone(voxels_occupancy_has)

#     new_bboxes = []
#     new_obj_ann_token_list = []
#     new_ann_info_list=[]

#     #(obj_point_cloud_list, obj_name_list, points_3D, obj_allocentric_list, obj_centers_list, obj_boxes_list, obj_gamma_list, kitti_boxes_list, lidar_sample_token, boxes, obj_ann_token_list, sample_annotation_tokens, sample_records, obj_ann_info_list, curr_sample_table_token)
#     print("num objs: ", len(dataset.obj_properties[5]))
#     dataset_obj_boxes_list = dataset.obj_properties[5]
#     original_box_idxs = [i for i, box in enumerate(dataset_obj_boxes_list) if (box.name in vehicle_names and vehicle_names[box.name] in {"bus", "car", "truck"})]
#     # close_to_zero = np.array([np.linalg.norm(box.center.reshape(-1)) for box in dataset.obj_properties[5]])
#     # idxs = np.argsort(close_to_zero)
#     # original_box_idxs = [idx for idx in idxs if (dataset_obj_boxes_list[idx].name in vehicle_names and vehicle_names[dataset_obj_boxes_list[idx].name] in {"bus", "car", "truck"})]
#     # original_box_idxs = original_box_idxs[:3]
#     dataset_obj_ann_token_list = [dataset.obj_properties[10][i] for i in original_box_idxs] #10
#     original_vehicle_boxes = [dataset_obj_boxes_list[i] for i in original_box_idxs]

#     names = [vehicle_names[box.name] for box in original_vehicle_boxes]
#     assert(len(dataset_obj_ann_token_list)==len(original_vehicle_boxes))
#     print(f"##### insert vehicle name: {names}")


#     kd_tree = KDTree(dataset.points_xyz[:,:3])

#     # prevent error when not entering the loop (empty names)
#     new_scene_points_xyz = dataset.points_xyz[:,:3]

#     new_points_xyz_no_resampling_occlusion = inpainted_points
    
#     print("length_names", len(names))

#     # iterate over each object we want to insert
#     for i, name in enumerate(names):
        
#         obj_properties = allocentric_dict[name] # the properties of objects belonging to the specified name
#         allocentric_angles = (obj_properties[0]) #np.array(obj_properties[0])
#         pc_filenames = obj_properties[1]
#         viewing_angles = obj_properties[2] #np.array(obj_properties[2])
#         boxes = obj_properties[3]
#         center3Ds = obj_properties[4] #np.array(obj_properties[4])
#         obj_ann_tokens = obj_properties[7]
#         is_trains = obj_properties[8]
#         ann_info_list = obj_properties[10]
#         obj_lidar_sample_token_list = obj_properties[6]
#         # is_good_completion_indicator_list = obj_properties[11]
#         box_volume_list = obj_properties[12]
#         print(name)

#         N = len(allocentric_angles)
#         assert(len(allocentric_angles)==len(pc_filenames)==len(viewing_angles)==len(boxes)==len(center3Ds))

#         ####### choose an object
#         original_box = original_vehicle_boxes[i]
#         original_volume = np.prod(original_box.wlh)
#         chosen_idx = np.argmin(np.abs((box_volume_list)-original_volume))

#         pc_filename = pc_filenames[chosen_idx]
#         bbox = boxes[chosen_idx]
#         center3D = center3Ds[chosen_idx]
#         pc_path = os.path.join(args.pc_path, name)
#         new_obj_ann_token = obj_ann_tokens[chosen_idx]
#         new_ann_info = ann_info_list[chosen_idx]

#         use_dense = args.dense==1
#         #if use_dense and is_good_completion_indicator_list[chosen_idx]:
#         if use_dense:
#             pc_full_path = os.path.join(args.pc_path, "dense_nusc", name, pc_filename)
#         else:
#             print("using incomplete point cloud......")
#             pc_full_path = os.path.join(pc_path, pc_filename)
#             #raise Exception("I prefer completed point cloud LOL")
#         vehicle_pc = np.asarray(open3d.io.read_point_cloud(pc_full_path).points) #np.load(pc_full_path)

#         ###### insert it right at the position of the original box in the scene before foreground removal
#         original_alpha, original_gamma, original_center = angles_from_box(original_box)
#         insert_xyz_pos = original_center.reshape(-1)
#         insert_xyz_pos[2]-= float(original_box.wlh[2])/2.0 # bottom of the box center

#         ####### align the allocentric angle with the original box first
#         desired_allocentric_angle = original_alpha
#         current_allocentric_angle = allocentric_angles[chosen_idx]
#         tmp_new_center = np.copy(bbox.center)
#         bbox.translate(-tmp_new_center)
#         bbox.rotate(pyquaternion_from_angle(-(desired_allocentric_angle - current_allocentric_angle)))
#         bbox.translate(tmp_new_center)
#         current_viewing_angle = viewing_angles[chosen_idx]

#         ##### also do this on the vehicle Pc
#         vehicle_pc = vehicle_pc-center3D
#         # ### We have to align the vehicle_pc orientation / edit the allocentric angle by rotation_angle
#         vehicle_pc = cart2polar(vehicle_pc, mode=mode)
#         theta = vehicle_pc[:,1]
#         theta = theta - (desired_allocentric_angle - current_allocentric_angle)
#         theta[theta<0] += 2*np.pi
#         theta = theta%(2*np.pi)
#         vehicle_pc[:,1] = theta
#         vehicle_pc = polar2cart(vehicle_pc, mode=mode)
#         vehicle_pc += center3D


#         ###### sample a perturbed position and a small random alpha offset
#         preturbed_result = sample_perturb_insert_pos(insert_xyz_pos, current_viewing_angle, dataset, bbox, new_bboxes)
#         if preturbed_result is None:
#             print("**** warning: skip current vehicle due to unsuccessful perturbation")
#             continue
#         insert_xyz_pos, rand_alpha_offset = preturbed_result

#         ###### get the rotation needed to align the viewing angle and to add the random alpha offset
#         desired_viewing_angle = compute_viewing_angle(insert_xyz_pos[:2])
#         current_viewing_angle = viewing_angles[chosen_idx]
#         rotation_align = -(desired_viewing_angle - current_viewing_angle) # negative sign because gamma increases clockwise
#         rotation_align-= (rand_alpha_offset)

#         new_scene_points_xyz, new_occupancy, new_bbox, insert_xyz_pos, vehicle_pc = insert_vehicle_pc(vehicle_pc, bbox, insert_xyz_pos, rotation_align, voxels_occupancy_has, voxelizer, dataset, mode=mode, use_ground_seg=True, center=center3D, kitti2nusc=False, use_dense=use_dense)
        
#         ### visualize without occlusion nor resampling
#         new_points_xyz_no_resampling_occlusion = np.concatenate((new_points_xyz_no_resampling_occlusion, vehicle_pc), axis=0)

#         voxels_occupancy_has = torch.tensor(new_occupancy).unsqueeze(0)
#         new_bboxes.append(new_bbox)
#         new_obj_ann_token_list.append(new_obj_ann_token)
#         new_ann_info_list.append(new_ann_info)

#     ######## remove new bboxes that contain no points after applying occlusion
#     new_bboxes_copy = [copy.deepcopy(box) for box in new_bboxes]
#     new_obj_ann_token_list_copy = [ann_token for ann_token in new_obj_ann_token_list]
#     new_ann_info_list_copy = [copy.deepcopy(ann_info) for ann_info in new_ann_info_list]
#     new_bboxes = []
#     new_obj_ann_token_list = []
#     new_ann_info_list = []
#     for i, box in enumerate(new_bboxes_copy):
#         mask = points_in_box(box, new_scene_points_xyz.T, wlh_factor = 1.0)
#         if np.sum(mask)!=0: # and np.sum(mask)>=50:
#             new_bboxes.append(box)
#             new_obj_ann_token_list.append(new_obj_ann_token_list_copy[i])
#             new_ann_info_list.append(new_ann_info_list_copy[i])
#     assert(len(new_bboxes)==len(new_obj_ann_token_list)==len(new_ann_info_list))

#     ######## visualize without resampling and occlusion
#     # print("############## visualizing inserted cars with no resampling nor occlusion")
#     # pcd = open3d.geometry.PointCloud()
#     # pcd.points = open3d.utility.Vector3dVector(np.array(new_points_xyz_no_resampling_occlusion))
#     # pcd_colors = np.tile(np.array([[0,0,1]]), (len(new_points_xyz_no_resampling_occlusion), 1))
#     # mask_vehicle = np.ones((len(new_points_xyz_no_resampling_occlusion),))==0
#     # for i, box in enumerate(new_bboxes_copy):
#     #     mask = points_in_box(box, new_points_xyz_no_resampling_occlusion.T, wlh_factor = 1.0)
#     #     mask_vehicle = mask_vehicle | mask
#     # pcd_colors[mask_vehicle==1, 0] = 1
#     # pcd_colors[mask_vehicle==1, 2] = 0
#     # pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
#     # mat = open3d.visualization.rendering.MaterialRecord()
#     # mat.shader = 'defaultUnlit'
#     # mat.point_size = 3.0

#     # ground_pcd = open3d.geometry.PointCloud()
#     # ground_pcd.points = open3d.utility.Vector3dVector(dataset.point_cloud_dataset.other_background_points[:,:3][:,:3])
#     # ground_pcd_colors = np.tile(np.array([[0,1,0]]), (len(dataset.point_cloud_dataset.other_background_points[:,:3][:,:3]), 1))
#     # ground_pcd.colors = open3d.utility.Vector3dVector(ground_pcd_colors)
#     # open3d.visualization.draw_geometries([pcd, ground_pcd])
#     # #open3d.visualization.draw([{'name': 'pcd', 'geometry': [pcd, ground_pcd], 'material': mat}], show_skybox=False)

#     print("############## visualizing inserted cars with no resampling nor occlusion")
#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(np.array(new_points_xyz_no_resampling_occlusion))
#     pcd_colors = np.tile(np.array([[0,0,1]]), (len(new_points_xyz_no_resampling_occlusion), 1))
#     mask_vehicle = np.ones((len(new_points_xyz_no_resampling_occlusion),))==0
#     for i, box in enumerate(new_bboxes_copy):
#         mask = points_in_box(box, new_points_xyz_no_resampling_occlusion.T, wlh_factor = 1.0)
#         mask_vehicle = mask_vehicle | mask
#     pcd_colors[mask_vehicle==1, 0] = 1
#     pcd_colors[mask_vehicle==1, 2] = 0
#     pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
#     mat = open3d.visualization.rendering.MaterialRecord()
#     mat.shader = 'defaultUnlit'
#     mat.point_size = 3.0

#     ground_pcd = open3d.geometry.PointCloud()
#     ground_pcd.points = open3d.utility.Vector3dVector(dataset.point_cloud_dataset.ground_points[:,:3][:,:3])
#     ground_pcd_colors = np.tile(np.array([[0,1,0]]), (len(dataset.point_cloud_dataset.ground_points[:,:3][:,:3]), 1))
#     ground_pcd.colors = open3d.utility.Vector3dVector(ground_pcd_colors)

#     lines = [[0, 1], [1, 2], [2, 3], [0, 3],
#          [4, 5], [5, 6], [6, 7], [4, 7],
#          [0, 4], [1, 5], [2, 6], [3, 7]]
#     visboxes = []
#     for box in new_bboxes_copy:
#         line_set = open3d.geometry.LineSet()
#         line_set.points = open3d.utility.Vector3dVector(box.corners().T)
#         line_set.lines = open3d.utility.Vector2iVector(lines)
#         colors = [[1, 0, 0] for _ in range(len(lines))]
#         line_set.colors = open3d.utility.Vector3dVector(colors)
#         visboxes.append(line_set)

#     open3d.visualization.draw_geometries([pcd, ground_pcd]+visboxes)


#     print("############## visualizing inserted cars with POST-PROCESSING i.e. OCCLUSION")
#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(np.array(new_scene_points_xyz))
#     pcd_colors = np.tile(np.array([[0,0,1]]), (len(new_scene_points_xyz), 1))
#     mask_vehicle = np.ones((len(new_scene_points_xyz),))==0
#     for i, box in enumerate(new_bboxes):
#         mask = points_in_box(box, new_scene_points_xyz.T, wlh_factor = 1.0)
#         mask_vehicle = mask_vehicle | mask
#     pcd_colors[mask_vehicle==1, 0] = 1
#     pcd_colors[mask_vehicle==1, 2] = 0
#     pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
#     mat = open3d.visualization.rendering.MaterialRecord()
#     mat.shader = 'defaultUnlit'
#     mat.point_size = 3.0
#     open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], show_skybox=False)



#     return new_scene_points_xyz, new_bboxes, token2sample_dict, voxels_occupancy_has, original_vehicle_boxes















































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
from datasets.data_utils_nuscenes import pyquaternion_from_angle, get_obj_regions
from datasets.dataset import Voxelizer, PolarDataset, collate_fn_BEV
from datasets.dataset_nuscenes import Nuscenes, NuscenesForeground
import torch
import configs.nuscenes_config as config

from models.vqvae_transformers import VQVAETrans, voxels2points
import open3d
import pickle

from pyquaternion import Quaternion
from shapely.geometry import Polygon
from nuscenes.utils.geometry_utils import points_in_box

'''
utilities for actor insertion
'''

def kitti2nusc(vehicle_pc):
    '''
    convert from kitti coordinates to nuscenes coordinates
    vehicle_pc: (N,3)
    '''
    kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
    kitti_to_nu_lidar_mat = (np.array(kitti_to_nu_lidar.rotation_matrix))
    vehicle_pc = np.matmul(kitti_to_nu_lidar_mat, vehicle_pc.T).T
    return vehicle_pc

def insert_vehicle_pc(vehicle_pc, bbox, insert_xyz_pos, rotation_angle, voxels_occupancy_has, voxelizer, dataset, mode='spherical', use_ground_seg=True, center=None, kitti2nusc=False, use_dense=True):
    '''
    insert a completeed vehicle_pc into the scene point cloud

    - vehicle_pc: (N,3)
    - Nuscenes bounding box of the vehicle_pc
    - insert_xyz_pos: (3,), first two elements are the x-y pos to place the vehicle, the third element can be any value because the z coordinate for insertion is going to be determined in this method.
    - rotation_angle: the angle in radian to rotate the point cloud to align orientations
    - voxels_occupancy_has: (1, #r, #theta, #z), occupancy grid of the scene
    - voxelizer: datasets.Voxelizer
    - dataset: datasets.PolarDataset, assuming I am using datasets.NuscenesForeground as the dataset.point_cloud_dataset
    - mode: either "polar" or "spherical
    - use_ground_seg: use ground segmentation or not
    - center: center the point cloud at the center or not, either None or (3,) shape array
    - kitti2nusc: whether vehicle_pc is from kitti
    - use_dense: whether to apply occlusion to the vehicle as well

    return:
    new_scene_points_xyz, new_bbox, insert_xyz_pos, vehicle_pc
    '''
   
    # pass object variables by copy
    new_bbox = copy.deepcopy(bbox)
    vehicle_pc = np.copy(vehicle_pc)
    insert_xyz_pos = np.copy(insert_xyz_pos)
    voxels_occupancy_has = torch.clone(voxels_occupancy_has)
    

    if kitti2nusc:
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_mat = (np.array(kitti_to_nu_lidar.rotation_matrix))
        vehicle_pc = np.matmul(kitti_to_nu_lidar_mat, vehicle_pc.T).T

    ##### We have assumed that the vehicle is centered at its centroid defined by its bounding box
    if center is not None:
        vehicle_pc = vehicle_pc-center

    # ### We have to align the vehicle_pc orientation / edit the allocentric angle by rotation_angle
    vehicle_pc = cart2polar(vehicle_pc, mode=mode)
    theta = vehicle_pc[:,1]
    theta = theta + rotation_angle
    theta[theta<0] += 2*np.pi
    theta = theta%(2*np.pi)
    vehicle_pc[:,1] = theta
    vehicle_pc = polar2cart(vehicle_pc, mode=mode)

    #### shift vehicle to the desired position
    vehicle_pc[:,:2] += insert_xyz_pos[:2]
    
    #### get the non-empty grid nearest to the vehicle in BEV, and set its lowest z-value as the lowest z-value of the vehicle
    #start_time = timeit.default_timer()
    if not use_ground_seg:
        print("^^^^^^ just using nearest occupied voxels to determine z")
        nearest_polar_voxels_pos = voxelizer.get_nearest_occupied_BEV_voxel(voxels_occupancy_has[0].cpu().detach().numpy(), cart2polar(insert_xyz_pos[np.newaxis,:], mode=mode), mode=mode) #(M,3)
    else:
        print(f"##### USING GROUND SEGMENTATION to determine z........")
        ground_points = dataset.point_cloud_dataset.ground_points[:,:3] #(G,3)
        ground_idxs = voxelizer.get_grid_ind(cart2polar(ground_points, mode=mode)) #(G,3)
        ground_occupancy = np.zeros_like(voxels_occupancy_has[0].cpu().detach().numpy())
        ground_occupancy[ground_idxs[:,0], ground_idxs[:,1], ground_idxs[:,2]] = 1.0
        #nearest_polar_voxels_pos = voxelizer.get_nearest_ground_BEV_voxel(ground_occupancy, cart2polar(insert_xyz_pos[np.newaxis,:], mode=mode), mode)
        nearest_polar_voxels_pos = voxelizer.get_nearest_ground_BEV_pos(ground_points, cart2polar(insert_xyz_pos[np.newaxis,:], mode=mode), mode)
    #mid_time = timeit.default_timer()
        
   # print(f"!!!!! time for getting ground segmentation z: {mid_time - start_time} seconds")

    nearest_cart_voxels_pos = polar2cart(nearest_polar_voxels_pos, mode=mode)
    nearest_min_z = np.min(nearest_cart_voxels_pos[:,2])
    vehicle_min_z = np.min(vehicle_pc[:,2])
    height_diff = nearest_min_z - vehicle_min_z
    vehicle_pc[:,2] += height_diff
    insert_xyz_pos[2]=height_diff

    #insert_xyz_pos[2]-=0.1#1.0

    ############ transform the bounding box accordingly
    new_bbox.translate(-bbox.center)
    new_bbox.rotate(pyquaternion_from_angle(rotation_angle))
    new_bbox.translate(insert_xyz_pos)

    #### project to spherical grid, apply occlusion and convert back to point cloud
    polar_vehicle = cart2polar(vehicle_pc, mode=mode)     
    #obj_region = get_obj_regions([new_bbox], mode=mode)[0]
    #old_occupancy = voxels_occupancy_has[0].cpu().detach().numpy()
    #new_occupancy[voxelizer.create_mask_by_occlusion(obj_region, use_z=True)==1] = 0
    if use_dense:
        new_occupancy = voxelizer.voxelize_and_occlude(voxels_occupancy_has[0].cpu().detach().numpy(), polar_vehicle, insert_only=False)
    else:
        new_occupancy,_ = voxelizer.voxelize_and_occlude_2(voxels_occupancy_has[0].cpu().detach().numpy(), polar_vehicle, add_vehicle=True, use_margin=True)

   
    new_scene_points_xyz = voxels2points(voxelizer, voxels=torch.tensor(new_occupancy).permute(2,0,1).unsqueeze(0), mode=mode)[0]

    return new_scene_points_xyz, new_occupancy, new_bbox, insert_xyz_pos, vehicle_pc


def is_overlap_shapely(box1, box2):
    if isinstance(box2, list):
        poly1 = Polygon(box1)
        for box in box2:
            poly2 = Polygon(box)
            if poly1.intersects(poly2):
                return True
        return False

    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    return poly1.intersects(poly2)

import cv2
def is_overlap_opencv(box1, box2):
    if isinstance(box2, list):
        rect1 = cv2.minAreaRect(box1)
        for box in box2:
            rect2 = cv2.minAreaRect(box)
            intersection = cv2.rotatedRectangleIntersection(rect1, rect2)
            if intersection[0] != cv2.INTERSECT_NONE:
                return True
        return False
    rect1 = cv2.minAreaRect(box1)
    rect2 = cv2.minAreaRect(box2)
    intersection = cv2.rotatedRectangleIntersection(rect1, rect2)
    return intersection[0] != cv2.INTERSECT_NONE

def sample_valid_insert_pos(name, current_viewing_angle, dataset, insert_box, other_boxes):
    '''
    find the pos to insert the box such that it is not colliding with other boxes and it does not contain any other background points except drivable surface
    - name: name of the vehicle being inserted: (car, bus or truck)
    - current_viewing_angle: the viewing angle of the vehicle's bounding box
    - dataset: PolarDataset
    - insert_box: the bounding box of the inserted vehicle
    - other_boxes: a list of other bounding boxes
    '''
    sparse_ground_points = np.copy(dataset.point_cloud_dataset.sparse_ground_points[:,:3]) #(G,3)
    if name=="car":
        sparse_ground_points = sparse_ground_points[np.linalg.norm(sparse_ground_points, axis=1)>5]
    elif name=="bus":
        sparse_ground_points = sparse_ground_points[np.linalg.norm(sparse_ground_points, axis=1)>15]
    elif name=="truck":
        sparse_ground_points = sparse_ground_points[np.linalg.norm(sparse_ground_points, axis=1)>10]
    
    other_boxes = [(copy.deepcopy(other_box).corners().T[[0,1,5,4], :2]).astype(np.float32) for other_box in other_boxes]
    target_idx = -1
    ### pick a random allocentric angle offset
    #rand_alpha=np.random.uniform(low=0.0, high=2*np.pi)
    alphas = np.linspace(start=0.0, stop=2*np.pi, num=20)
    target_rand_alpha = 0
    # remember to shuffle
    np.random.shuffle(sparse_ground_points)
    np.random.shuffle(alphas)
    got_candidate = False
    for ground_point_idx in range(sparse_ground_points.shape[0]):
        for rand_alpha in alphas:
            insert_box_copy = copy.deepcopy(insert_box)
            ground_point = sparse_ground_points[ground_point_idx, :]

            # simulate moving the box there
            desired_viewing_angle = compute_viewing_angle(ground_point[:2])
            rotation_align = -(desired_viewing_angle - current_viewing_angle) # negative sign because gamma increases clockwise
            rotation_align-=rand_alpha
            insert_box_copy.translate(-insert_box_copy.center)
            insert_box_copy.rotate(pyquaternion_from_angle(rotation_align))
            insert_box_copy.translate(ground_point)

            insert_box_2d = (insert_box_copy.corners().T[[0,1,5,4], :2]).astype(np.float32)
            assert(isinstance(insert_box_2d, np.ndarray))

            if np.sum(points_in_box(insert_box_copy, dataset.point_cloud_dataset.other_background_points[:,:3].T, wlh_factor = 1.0))>0:
                continue
            
            if len(other_boxes)!=0:
                if not is_overlap_shapely(insert_box_2d, other_boxes):
                    target_idx = ground_point_idx
                    got_candidate=True
                    target_rand_alpha = rand_alpha
                    break
            else:
                target_idx = np.random.randint(low=0, high=len(sparse_ground_points))
                got_candidate=True
                target_rand_alpha = 0
                break
        # exit the loop if we have found a valid insert pos
        if got_candidate:
            break

    
    if target_idx==-1:
        print("WARNING: no valid insert pos")
        return None
        

    insert_xyz_pos = sparse_ground_points[target_idx]
    #dataset.point_cloud_dataset.sparse_ground_points = dataset.point_cloud_dataset.sparse_ground_points[np.arange(len(sparse_ground_points))!=target_idx] #remove the random ground points
    print(f"+++++ insert_xyz_pos: {insert_xyz_pos}")

    return insert_xyz_pos, target_rand_alpha

from datasets.dataset_nuscenes import vehicle_names, plot_obj_regions
import timeit

def angles_from_box(box):
    '''
    Get the viewing angle and allocentric angle of the box
    '''
    corners = box.corners()
    center2D = box.center[:2]
    corner1, corner2 = corners[:,0][:2], corners[:,1][:2]  # top front corners (left and right)
    corner7, corner6 = corners[:,6][:2], corners[:,5][:2] # bottom back corner (right), top back corner (right)
    center2D = (corner1 + corner6)/2

    center3D = np.array([center2D[0], center2D[1], (corners[2,2]+corners[2,1])/2])

    right_pointing_vector = (corner2 + corner6)/2.0 - center2D
    front_pointing_vector = (corner1 + corner2)/2.0 - center2D
    obj2cam_vector = -center2D
    #print(np.arccos(np.dot(right_pointing_vector/np.linalg.norm(right_pointing_vector), front_pointing_vector/np.linalg.norm(front_pointing_vector)))/np.pi*180)
    
    ### compute allocentric angle and viewing angle gamma
    alpha = compute_allocentric_angle(obj2cam_pos=obj2cam_vector, obj_right_axis=right_pointing_vector, obj_front_axis=front_pointing_vector)
    gamma = compute_viewing_angle(-obj2cam_vector)

    return alpha, gamma, center3D


from scipy.spatial import KDTree

def insertion_vehicles_driver(inpainted_points, allocentric_dict, voxels_occupancy_has, names, dataset, voxelizer, token2sample_dict, args, save_lidar_path, mode="spherical"):
    '''
    Driver method of inserting vehicle
    - allocentric dict: contains all information of the completed point clouds
    - voxels_occupancy_has: (1, #r, #theta, #z), occupancy grid of the scene
    #- names: list of sampled vehicle names
    - dataset: PolarDataset
    - voxelizer: Voxelizer
    - token2sample_dict: modify this in place and save it, maps from lidar sample token to generated data we will use
    - voxels_occupancy_has: occupancy grid of shape (1,#r, #theta, #z), make a copy and modify out of place
    - save_lidar_path: the full path to the where the data is save
    - mode: "spherical"
    '''
    assert(mode=="spherical")
    assert(names is None)

    # make a copy of the voxel occupancy
    voxels_occupancy_has = torch.clone(voxels_occupancy_has)

    new_bboxes = []
    new_obj_ann_token_list = []
    new_ann_info_list=[]

    #(obj_point_cloud_list, obj_name_list, points_3D, obj_allocentric_list, obj_centers_list, obj_boxes_list, obj_gamma_list, kitti_boxes_list, lidar_sample_token, boxes, obj_ann_token_list, sample_annotation_tokens, sample_records, obj_ann_info_list, curr_sample_table_token)
    dataset_obj_boxes_list = dataset.obj_properties[5] #5
    original_box_idxs = [i for i, box in enumerate(dataset_obj_boxes_list) if (box.name in vehicle_names and vehicle_names[box.name] in {"bus", "car", "truck"})]
    dataset_obj_ann_token_list = [dataset.obj_properties[10][i] for i in original_box_idxs] #10
    original_vehicle_boxes = [dataset_obj_boxes_list[i] for i in original_box_idxs]

    names = [vehicle_names[box.name] for box in original_vehicle_boxes]
    assert(len(dataset_obj_ann_token_list)==len(original_vehicle_boxes))
    print(f"##### insert vehicle name: {names}")


    kd_tree = KDTree(dataset.points_xyz[:,:3])

    # prevent error when not entering the loop (empty names)
    new_scene_points_xyz = dataset.points_xyz[:,:3]

    new_points_xyz_no_resampling_occlusion = inpainted_points
    
    
    print("length_names", len(names))

    # iterate over each object we want to insert
    for i, name in enumerate(names):
        
        obj_properties = allocentric_dict[name] # the properties of objects belonging to the specified name
        allocentric_angles = (obj_properties[0]) #np.array(obj_properties[0])
        pc_filenames = obj_properties[1]
        viewing_angles = obj_properties[2] #np.array(obj_properties[2])
        boxes = obj_properties[3]
        center3Ds = obj_properties[4] #np.array(obj_properties[4])
        obj_ann_tokens = obj_properties[7]
        is_trains = obj_properties[8]
        ann_info_list = obj_properties[10]
        obj_lidar_sample_token_list = obj_properties[6]
        # is_good_completion_indicator_list = obj_properties[11]
        box_volume_list = obj_properties[12]
        print(name)

        N = len(allocentric_angles)
        assert(len(allocentric_angles)==len(pc_filenames)==len(viewing_angles)==len(boxes)==len(center3Ds))

        ####### choose an object
        original_box = original_vehicle_boxes[i]
        original_volume = np.prod(original_box.wlh)
        chosen_idx = np.argmin(np.abs((box_volume_list)-original_volume))


        pc_filename = pc_filenames[chosen_idx]
        bbox = boxes[chosen_idx]
        center3D = center3Ds[chosen_idx]
        pc_path = os.path.join(args.pc_path, name)
        new_obj_ann_token = obj_ann_tokens[chosen_idx]
        new_ann_info = ann_info_list[chosen_idx]

        use_dense = args.dense==1
        #if use_dense and is_good_completion_indicator_list[chosen_idx]:
        if use_dense:
            pc_full_path = os.path.join(args.pc_path, "dense_nusc", name, pc_filename)
        else:
            print("using incomplete point cloud......")
            pc_full_path = os.path.join(pc_path, pc_filename)
            #raise Exception("I prefer completed point cloud LOL")
        vehicle_pc = np.asarray(open3d.io.read_point_cloud(pc_full_path).points) #np.load(pc_full_path)

        ###### insert it right at the position of the original box in the scene before foreground removal
        # original_box = original_vehicle_boxes[i]
        original_alpha, original_gamma, original_center = angles_from_box(original_box)
        insert_xyz_pos = original_center.reshape(-1)
        insert_xyz_pos[2]-= float(original_box.wlh[2])/2.0 # bottom of the box center

        # if use_dense and is_good_completion_indicator_list[chosen_idx]:
        #     print(f"((((( {name} )))))")
        #     print("################ saved bbox center: ", bbox.center.reshape(-1))
        #     print("################ saved bbox center: ", original_box.center.reshape(-1))

        desired_viewing_angle = original_gamma
        desired_allocentric_angle = original_alpha
        current_viewing_angle = viewing_angles[chosen_idx]
        current_allocentric_angle = allocentric_angles[chosen_idx]
        rotation_align = -(desired_viewing_angle - current_viewing_angle) # negative sign because gamma increases clockwise
        # align allocentric as well
        rotation_align-= (desired_allocentric_angle - current_allocentric_angle)

        ############## IDK WHY THIS ALLOCENTRIC ANGLE ALIGN IS WRONG ###########
        # print("############## visualizing insert bbox")
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(vehicle_pc)
        # pcd_colors = np.tile(np.array([[0,0,1]]), (len(vehicle_pc), 1))
        # pcd.colors = open3d.utility.Vector3dVector(pcd_colors)

        # lines = [[0, 1], [1, 2], [2, 3], [0, 3],
        #     [4, 5], [5, 6], [6, 7], [4, 7],
        #     [0, 4], [1, 5], [2, 6], [3, 7]]
        # visboxes = []
        # for box in [bbox]:
        #     line_set = open3d.geometry.LineSet()
        #     line_set.points = open3d.utility.Vector3dVector(box.corners().T)
        #     line_set.lines = open3d.utility.Vector2iVector(lines)
        #     colors = [[1, 0, 0] for _ in range(len(lines))]
        #     line_set.colors = open3d.utility.Vector3dVector(colors)
        #     visboxes.append(line_set)

        # open3d.visualization.draw_geometries([pcd]+visboxes)
        

        ##### warning: overriding points and boxes
        # vehicle_pc = original_points[i]
        # bbox = original_vehicle_boxes[i]
        # center3D = bbox.center.reshape(-1)
        # rotation_align = 0.0

        #start_time = timeit.default_timer()
        new_scene_points_xyz, new_occupancy, new_bbox, insert_xyz_pos, vehicle_pc = insert_vehicle_pc(vehicle_pc, bbox, insert_xyz_pos, rotation_align, voxels_occupancy_has, voxelizer, dataset, mode=mode, use_ground_seg=True, center=center3D, kitti2nusc=False, use_dense=use_dense)
        #mid_time = timeit.default_timer()
        #print(f"!!!!! time for occlusion: {mid_time - start_time} seconds")
        #voxelizer.verify_occlusion(new_occupancy)

        # ############## IDK WHY THIS ALLOCENTRIC ANGLE ALIGN IS WRONG ###########
        # print("############## visualizing insert bbox")
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(vehicle_pc)
        # pcd_colors = np.tile(np.array([[0,0,1]]), (len(vehicle_pc), 1))
        # pcd.colors = open3d.utility.Vector3dVector(pcd_colors)

        # lines = [[0, 1], [1, 2], [2, 3], [0, 3],
        #     [4, 5], [5, 6], [6, 7], [4, 7],
        #     [0, 4], [1, 5], [2, 6], [3, 7]]
        # visboxes = []
        # for box in [new_bbox]:
        #     line_set = open3d.geometry.LineSet()
        #     line_set.points = open3d.utility.Vector3dVector(box.corners().T)
        #     line_set.lines = open3d.utility.Vector2iVector(lines)
        #     colors = [[1, 0, 0] for _ in range(len(lines))]
        #     line_set.colors = open3d.utility.Vector3dVector(colors)
        #     visboxes.append(line_set)

        # open3d.visualization.draw_geometries([pcd]+visboxes)



         ### visualize without occlusion nor resampling
        new_points_xyz_no_resampling_occlusion = np.concatenate((new_points_xyz_no_resampling_occlusion, vehicle_pc), axis=0)
        
        voxels_occupancy_has = torch.tensor(new_occupancy).unsqueeze(0)
        new_bboxes.append(new_bbox)
        new_obj_ann_token_list.append(new_obj_ann_token)
        new_ann_info_list.append(new_ann_info)

    ## remove new bboxes that contain no points after applying occlusion
    new_bboxes_copy = [copy.deepcopy(box) for box in new_bboxes]
    new_obj_ann_token_list_copy = [ann_token for ann_token in new_obj_ann_token_list]
    new_ann_info_list_copy = [copy.deepcopy(ann_info) for ann_info in new_ann_info_list]
    new_bboxes = []
    new_obj_ann_token_list = []
    new_ann_info_list = []
    for i, box in enumerate(new_bboxes_copy):
        mask = points_in_box(box, new_scene_points_xyz.T, wlh_factor = 1.0)
        if np.sum(mask)!=0: # and np.sum(mask)>=50:
            new_bboxes.append(box)
            new_obj_ann_token_list.append(new_obj_ann_token_list_copy[i])
            new_ann_info_list.append(new_ann_info_list_copy[i])
    assert(len(new_bboxes)==len(new_obj_ann_token_list)==len(new_ann_info_list))

    ### add nosiy points to duplicate
    # noises = np.random.normal(loc=0.0, scale=0.001, size=(len(new_scene_points_xyz),3))
    # new_scene_points_add = np.copy(new_scene_points_xyz) + noises
    # new_scene_points_xyz = np.concatenate((new_scene_points_xyz, new_scene_points_add), axis=0)

    
    ## get nearest neighbor intensity
    original_points = dataset.points_xyz #(N,5)
    _, nearest_idxs = kd_tree.query(new_scene_points_xyz[:,:3], k=1)
    
    # make the point dim be 5
    extras = np.zeros((len(new_scene_points_xyz), 2))
    new_points_xyz = np.concatenate((new_scene_points_xyz, extras), axis=1)
    assert(new_points_xyz.shape[-1]==5)

    new_points_xyz[:,3] = original_points[:,3][nearest_idxs].astype(np.float64)
    #new_points_xyz[:,3] = 0.0 #original_points[:,3][nearest_idxs].astype(np.float64)
    new_points_xyz[:,4] = 0.0 #original_points[0,4]

    # ######## visualize without resampling and occlusion
    # print("############## visualizing inserted cars with no resampling nor occlusion")
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(np.array(new_points_xyz_no_resampling_occlusion))
    # pcd_colors = np.tile(np.array([[0,0,1]]), (len(new_points_xyz_no_resampling_occlusion), 1))
    # mask_vehicle = np.ones((len(new_points_xyz_no_resampling_occlusion),))==0
    # for i, box in enumerate(new_bboxes_copy):
    #     mask = points_in_box(box, new_points_xyz_no_resampling_occlusion.T, wlh_factor = 1.0)
    #     mask_vehicle = mask_vehicle | mask
    # pcd_colors[mask_vehicle==1, 0] = 1
    # pcd_colors[mask_vehicle==1, 2] = 0
    # pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
    # mat = open3d.visualization.rendering.MaterialRecord()
    # mat.shader = 'defaultUnlit'
    # mat.point_size = 3.0

    # ground_pcd = open3d.geometry.PointCloud()
    # ground_pcd.points = open3d.utility.Vector3dVector(dataset.point_cloud_dataset.ground_points[:,:3][:,:3])
    # ground_pcd_colors = np.tile(np.array([[0,1,0]]), (len(dataset.point_cloud_dataset.ground_points[:,:3][:,:3]), 1))
    # ground_pcd.colors = open3d.utility.Vector3dVector(ground_pcd_colors)

    # lines = [[0, 1], [1, 2], [2, 3], [0, 3],
    #      [4, 5], [5, 6], [6, 7], [4, 7],
    #      [0, 4], [1, 5], [2, 6], [3, 7]]
    # visboxes = []
    # for box in new_bboxes_copy:
    #     line_set = open3d.geometry.LineSet()
    #     line_set.points = open3d.utility.Vector3dVector(box.corners().T)
    #     line_set.lines = open3d.utility.Vector2iVector(lines)
    #     colors = [[1, 0, 0] for _ in range(len(lines))]
    #     line_set.colors = open3d.utility.Vector3dVector(colors)
    #     visboxes.append(line_set)

    # open3d.visualization.draw_geometries([pcd, ground_pcd]+visboxes)


    # # print("############## visualizing inserted cars with POST-PROCESSING i.e. OCCLUSION")
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(np.array(new_scene_points_xyz))
    # pcd_colors = np.tile(np.array([[0,0,1]]), (len(new_scene_points_xyz), 1))
    # mask_vehicle = np.ones((len(new_scene_points_xyz),))==0
    # for i, box in enumerate(new_bboxes):
    #     mask = points_in_box(box, new_scene_points_xyz.T, wlh_factor = 1.0)
    #     mask_vehicle = mask_vehicle | mask
    # pcd_colors[mask_vehicle==1, 0] = 1
    # pcd_colors[mask_vehicle==1, 2] = 0
    # pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
    
    # lines = [[0, 1], [1, 2], [2, 3], [0, 3],
    #      [4, 5], [5, 6], [6, 7], [4, 7],
    #      [0, 4], [1, 5], [2, 6], [3, 7]]
    # visboxes = []
    # for box in new_bboxes:
    #     line_set = open3d.geometry.LineSet()
    #     line_set.points = open3d.utility.Vector3dVector(box.corners().T)
    #     line_set.lines = open3d.utility.Vector2iVector(lines)
    #     colors = [[1, 0, 0] for _ in range(len(lines))]
    #     line_set.colors = open3d.utility.Vector3dVector(colors)
    #     visboxes.append(line_set)

    # open3d.visualization.draw_geometries([pcd]+visboxes)

    


    points_xyz = new_scene_points_xyz
    bounding_boxes = new_bboxes
    lidar_sample_token = dataset.lidar_sample_token

    ########### nuscens get_sample_data method has converted the box from global coodinates to lidar coordinates as follows:
    ## # Move box to ego vehicle coord system.
    ## box.translate(-np.array(pose_record['translation']))
    ## box.rotate(Quaternion(pose_record['rotation']).inverse)
    ## #  Move box to sensor coord system.
    ## box.translate(-np.array(cs_record['translation']))
    ## box.rotate(Quaternion(cs_record['rotation']).inverse)
    ######### This may be useful: you can convert the box coordinates back to global coordinates instead of lidar coordinates
    sample_records = dataset.obj_properties[12]
    cs_record, sensor_record, pose_record = sample_records["cs_record"], sample_records["sensor_record"], sample_records["pose_record"]
    # for new_box in new_bboxes:
    #     box_tmp = copy.deepcopy(new_box)
    #     # undo the transformation done by nuscenes get_sample_data method
    #     box_tmp.rotate(Quaternion(cs_record['rotation']))
    #     box_tmp.translate(np.array(cs_record['translation']))
    #     box_tmp.rotate(Quaternion(pose_record['rotation']))
    #     box_tmp.translate(np.array(pose_record['translation']))
    #     print(" box center in global coordinate: ", box_tmp.center)

    ############# save point cloud 
    #start_pc_time = timeit.default_timer()
    pc_name = f'{args.split}_{lidar_sample_token}.bin'
    os.makedirs(os.path.join(save_lidar_path, "lidar_point_clouds"), exist_ok=True)
    lidar_full_path = os.path.join(save_lidar_path, "lidar_point_clouds", pc_name)
    #assert(not os.path.exists(lidar_full_path))
    new_points_xyz.astype(np.float32).tofile(lidar_full_path)
    #end_pc_time = timeit.default_timer()
    #print(f"$$$$$ .bin file point cloud saving time: {end_pc_time - start_pc_time} seconds")


    ############## Save the data needed to build the new database
    token2sample_dict[lidar_sample_token] = (lidar_full_path, bounding_boxes, new_obj_ann_token_list, sample_records, new_ann_info_list)

    # start_dict_time = timeit.default_timer()
    # print(f"========= SAVING DATA TO {save_lidar_path}")
    # token2sample_dict_full_path = os.path.join(save_lidar_path, "token2sample.pickle")
    # with open(token2sample_dict_full_path, 'wb') as handle:
    #     pickle.dump(token2sample_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # end_dict_time = timeit.default_timer()
    # print(f"$$$$$ token2sample dict saving time: {end_dict_time - start_dict_time} seconds")

    return new_scene_points_xyz, new_bboxes, token2sample_dict, voxels_occupancy_has, original_vehicle_boxes



def insertion_vehicles_driver_random(voxels_occupancy_has, names, dataset, voxelizer, token2sample_dict, args, save_lidar_path, mode="spherical"):
    '''
    Driver method of inserting vehicle with everything random
    - voxels_occupancy_has: (1, #r, #theta, #z), occupancy grid of the scene
    #- names: list of sampled vehicle names
    - dataset: PolarDataset
    - voxelizer: Voxelizer
    - token2sample_dict: modify this in place and save it, maps from lidar sample token to generated data we will use
    - voxels_occupancy_has: occupancy grid of shape (1,#r, #theta, #z), make a copy and modify out of place
    - save_lidar_path: the full path to the where the data is save
    - mode: "spherical"
    '''
    assert(mode=="spherical")

    # make a copy of the voxel occupancy
    voxels_occupancy_has = torch.clone(voxels_occupancy_has)

    new_bboxes = []
    new_obj_ann_token_list = []
    new_ann_info_list=[]

    # iterate over each object we want to insert
    for i, name in enumerate(names):
        with open(os.path.join(args.pc_path, "allocentric.pickle"), 'rb') as handle:
            allocentric_dict = pickle.load(handle)

        obj_properties = allocentric_dict[name] # the properties of objects belonging to the specified name
        allocentric_angles = np.array(obj_properties[0])
        pc_filenames = obj_properties[1]
        viewing_angles = np.array(obj_properties[2])
        boxes = obj_properties[3]
        center3Ds = np.array(obj_properties[4])
        obj_ann_tokens = obj_properties[7]
        is_trains = obj_properties[8]
        ann_info_list = obj_properties[10]
        print(name)

        N = len(allocentric_angles)
        assert(len(allocentric_angles)==len(pc_filenames)==len(viewing_angles)==len(boxes)==len(center3Ds))

        ####### choose an object
        chosen_idx = np.random.randint(low=0, high=N)
        
        pc_filename = pc_filenames[chosen_idx]
        bbox = boxes[chosen_idx]
        center3D = center3Ds[chosen_idx]
        pc_path = os.path.join(args.pc_path, name)
        new_obj_ann_token = obj_ann_tokens[chosen_idx]
        new_ann_info = ann_info_list[chosen_idx]

        use_dense = args.dense==1
        if use_dense:
            pc_full_path = os.path.join(args.pc_path, "dense_nusc", name, pc_filename)
        else:
            pc_full_path = os.path.join(pc_path, pc_filename)
            #raise Exception("I prefer completed point cloud LOL")
        vehicle_pc = np.asarray(open3d.io.read_point_cloud(pc_full_path).points) #np.load(pc_full_path)

        ###### choose where to insert vehicle
        
        # simulate inserting the vehicle at each ground point until we find a valid pos that does not contain background points and does not collide with other bounding boxes
        # start_time = timeit.default_timer()
        valid_pos_info = sample_valid_insert_pos(name, viewing_angles[chosen_idx], dataset, bbox, new_bboxes)
        # mid_time = timeit.default_timer()
        # print(f"!!!!! time for sample valid pos: {mid_time - start_time} seconds")
        
        if valid_pos_info is not None:
            insert_xyz_pos, rand_alpha = valid_pos_info
        else:
            print("warning: skip this vehicle")
            continue

        desired_viewing_angle = compute_viewing_angle(insert_xyz_pos[:2])
        current_viewing_angle = viewing_angles[chosen_idx]
        rotation_align = -(desired_viewing_angle - current_viewing_angle) # negative sign because gamma increases clockwise
        # random allocentric offset
        rotation_align-=rand_alpha

        #start_time = timeit.default_timer()
        new_scene_points_xyz, new_occupancy, new_bbox, insert_xyz_pos, vehicle_pc = insert_vehicle_pc(vehicle_pc, bbox, insert_xyz_pos, rotation_align, voxels_occupancy_has, voxelizer, dataset, mode=mode, use_ground_seg=True, center=center3D, kitti2nusc=False, use_dense=use_dense)
        #mid_time = timeit.default_timer()
        #print(f"!!!!! time for occlusion: {mid_time - start_time} seconds")
        #voxelizer.verify_occlusion(new_occupancy)
        
        voxels_occupancy_has = torch.tensor(new_occupancy).unsqueeze(0)
        new_bboxes.append(new_bbox)
        new_obj_ann_token_list.append(new_obj_ann_token)
        new_ann_info_list.append(new_ann_info)

    ### remove new bboxes that contain no points after applying occlusion
    new_bboxes_copy = [copy.deepcopy(box) for box in new_bboxes]
    new_obj_ann_token_list_copy = [ann_token for ann_token in new_obj_ann_token_list]
    new_ann_info_list_copy = [copy.deepcopy(ann_info) for ann_info in new_ann_info_list]
    new_bboxes = []
    new_obj_ann_token_list = []
    new_ann_info_list = []
    for i, box in enumerate(new_bboxes_copy):
        mask = points_in_box(box, new_scene_points_xyz.T, wlh_factor = 1.0)
        if np.sum(mask)!=0: # and np.sum(mask)>=50:
            new_bboxes.append(box)
            new_obj_ann_token_list.append(new_obj_ann_token_list_copy[i])
            new_ann_info_list.append(new_ann_info_list_copy[i])
    assert(len(new_bboxes)==len(new_obj_ann_token_list)==len(new_ann_info_list))
    
    # colors = []
    # for box in new_bboxes:
    #     if vehicle_names[box.name]=="bus":
    #         colors.append('b')
    #     elif vehicle_names[box.name]=="truck":
    #         colors.append('g')
    #     else:
    #         colors.append('r')
    
    # plot_obj_regions([], [], new_scene_points_xyz, 40, new_bboxes, xlim=[-40,40], ylim=[-40,40], title=f"check_generated", path="./actor_insertion/vis_insert_buses", name=f"buses & trucks", \
    #     vis=False, colors=colors)



    ## get nearest neighbor intensity
    original_points = dataset.points_xyz #(N,5)
    nearest_idxs = np.argmin(np.linalg.norm(new_scene_points_xyz[:,:3][np.newaxis,...] - original_points[:,:3][:,np.newaxis, :], axis=-1), axis=0) #(1,M,3) - (N,1,3) = (N,M,3) => (M,3)
    # make the point dim be 5
    extras = np.zeros((len(new_scene_points_xyz), 2))
    new_points_xyz = np.concatenate((new_scene_points_xyz, extras), axis=1)
    new_points_xyz[:,3] = original_points[:,3][nearest_idxs].astype(np.float64)
    new_points_xyz[:,4] = 0.0 #original_points[0,4]

    points_xyz = new_scene_points_xyz
    bounding_boxes = new_bboxes
    lidar_sample_token = dataset.lidar_sample_token

    ############### visualize with intensity
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(np.array(new_points_xyz[:,:3]))
    # pcd_colors = np.tile(np.array([[0,0,1]]), (len(new_points_xyz[:,:3]), 1)).astype(np.float64)
    # pcd_colors[:,2] *= new_points_xyz[:, 3]/255.0
    # pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
    # mat = open3d.visualization.rendering.MaterialRecord()
    # mat.shader = 'defaultUnlit'
    # mat.point_size = 3.0
    # open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], show_skybox=False)

    
    ########### nuscens get_sample_data method has converted the box from global coodinates to lidar coordinates as follows:
    ## # Move box to ego vehicle coord system.
    ## box.translate(-np.array(pose_record['translation']))
    ## box.rotate(Quaternion(pose_record['rotation']).inverse)
    ## #  Move box to sensor coord system.
    ## box.translate(-np.array(cs_record['translation']))
    ## box.rotate(Quaternion(cs_record['rotation']).inverse)
    ######### This may be useful: you can convert the box coordinates back to global coordinates instead of lidar coordinates
    sample_records = dataset.obj_properties[12]
    cs_record, sensor_record, pose_record = sample_records["cs_record"], sample_records["sensor_record"], sample_records["pose_record"]
    # for new_box in new_bboxes:
    #     box_tmp = copy.deepcopy(new_box)
    #     # undo the transformation done by nuscenes get_sample_data method
    #     box_tmp.rotate(Quaternion(cs_record['rotation']))
    #     box_tmp.translate(np.array(cs_record['translation']))
    #     box_tmp.rotate(Quaternion(pose_record['rotation']))
    #     box_tmp.translate(np.array(pose_record['translation']))
    #     print(" box center in global coordinate: ", box_tmp.center)

    ############# save point cloud 
    pc_name = f'{args.split}_{lidar_sample_token}.bin'
    os.makedirs(os.path.join(save_lidar_path, "lidar_point_clouds"), exist_ok=True)
    lidar_full_path = os.path.join(save_lidar_path, "lidar_point_clouds", pc_name)
    assert(not os.path.exists(lidar_full_path))
    new_points_xyz.astype(np.float32).tofile(lidar_full_path)

    ############## Save the data needed to build the new database
    token2sample_dict[lidar_sample_token] = (lidar_full_path, bounding_boxes, new_obj_ann_token_list, sample_records, new_ann_info_list)

    print(f"========= SAVING DATA TO {save_lidar_path}")
    token2sample_dict_full_path = os.path.join(save_lidar_path, "token2sample.pickle")
    with open(token2sample_dict_full_path, 'wb') as handle:
        pickle.dump(token2sample_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return new_scene_points_xyz, new_bboxes, token2sample_dict, voxels_occupancy_has






def save_reconstruct_data(rec_voxels_occupancy_has, dataset, voxelizer, token2sample_dict, args, save_lidar_path, mode="spherical"):
    '''
    Just save the reconstructed voxels with foreground points and the corresponding bounding boxes, annotations and sample_records
    
    - voxels occupancy has: shape (1, #z, #r, #theta)
    '''
    new_scene_points_xyz = voxels2points(voxelizer, rec_voxels_occupancy_has, mode=mode)[0]
    bounding_boxes = dataset.obj_properties[9]
    lidar_sample_token = dataset.lidar_sample_token
    sample_records = dataset.obj_properties[12]
    obj_ann_token_list = dataset.obj_properties[11]
    
     ## get nearest neighbor intensity
    original_points = dataset.points_xyz #(N,5)
    nearest_idxs = np.argmin(np.linalg.norm(new_scene_points_xyz[:,:3][np.newaxis,...] - original_points[:,:3][:,np.newaxis, :], axis=-1), axis=0) #(1,M,3) - (N,1,3) = (N,M,3) => (M,3)
    # make the point dim be 5
    extras = np.zeros((len(new_scene_points_xyz), 2))
    new_points_xyz = np.concatenate((new_scene_points_xyz, extras), axis=1)
    new_points_xyz[:,3] = original_points[:,3][nearest_idxs].astype(np.float64)
    new_points_xyz[:,4] = 0.0 #original_points[0,4]
    
    ############# save point cloud 
    pc_name = f'{args.split}_{lidar_sample_token}.bin'
    os.makedirs(os.path.join(save_lidar_path, "lidar_point_clouds"), exist_ok=True)
    lidar_full_path = os.path.join(save_lidar_path, "lidar_point_clouds", pc_name)
    assert(not os.path.exists(lidar_full_path))
    new_points_xyz.astype(np.float32).tofile(lidar_full_path)

    ############## Save the data needed to build the new database
    token2sample_dict[lidar_sample_token] = (lidar_full_path, bounding_boxes, obj_ann_token_list, sample_records)

    token2sample_dict_full_path = os.path.join(save_lidar_path, "token2sample.pickle")
    with open(token2sample_dict_full_path, 'wb') as handle:
        pickle.dump(token2sample_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return new_scene_points_xyz, bounding_boxes

    


def copy_and_paste_method(voxels_occupancy_has, dataset, voxelizer):
    '''
    Generate foreground object-free point cloud by a naive copy and paste method
    voxels_occupancy_has: (1, C, H, W) or equivalently (1, #z, #r, #theta)
    dataset: PolarDataset
    voxelizer: Voxelizer

    return:
    generated occupancy grid of shape (1, C, H, W)
    '''
    gen_binary_voxels = torch.clone(voxels_occupancy_has.permute(0,2,3,1)[0].detach().cpu())
    for mask in dataset.obj_voxels_mask_list:
        if np.sum(mask)==0:
            continue
        # print("mask sum: ", np.sum(mask))
        gen_binary_voxels = voxelizer.copy_and_paste_neighborhood(gen_binary_voxels, voxels_mask=torch.tensor(mask)) #(H,W,in_chans)
    gen_binary_voxels = gen_binary_voxels.unsqueeze(0).permute(0,3,1,2) #(1, C, H, W)
    return gen_binary_voxels


def count_vehicle_name_in_box_list(boxes, vehicle_names_dict):
    '''
    maps the raw vehicle name to the name we use for categorizing vehicles
    '''
    count = {v_name:0 for v_name in vehicle_names_dict.values()}
    for box in boxes:
        #print(box.name)
        
        if box.name in vehicle_names_dict.keys():
            name = vehicle_names_dict[box.name]
            count[name]+=1
    return count
            

###################################################################################################################################
def perturb_2d(point, radius):
    '''
    point: (2,)
    '''
    theta = np.random.uniform(0, 2 * np.pi)
    # Random radius within the given limit
    r = np.random.uniform(0, radius)

    dx = r * np.cos(theta)
    dy = r * np.sin(theta)

    perturbed_point = point + np.array([dx, dy])
    return perturbed_point


def sample_perturb_insert_pos(original_insert_xyz, current_viewing_angle, dataset, insert_box, other_boxes):
    '''
    TODO: implement this
    find the pos perturbed from an original position to insert the box such that it is not colliding with other boxes and it does not contain any other background points except drivable surface

    - original_insert_xyz: original xyz-pos to insert the vehicle
    - current_viewing_angle: the viewing angle of the vehicle's bounding box
    - dataset: PolarDataset
    - insert_box: the bounding box of the inserted vehicle
    - other_boxes: a list of other bounding boxes
    '''
    other_boxes = [(copy.deepcopy(other_box).corners().T[[0,1,5,4], :2]).astype(np.float32) for other_box in other_boxes]
    ### pick a random allocentric angle offset with 45 degrees
    alphas = np.linspace(start=-np.pi/4, stop=np.pi/4, num=20)
    target_rand_alpha = 0
    # remember to shuffle
    np.random.shuffle(alphas)
    got_candidate = False
    perturbed_insert_xyz_copy = np.copy(original_insert_xyz)

    for num_try in range(10): # try perturb at most 10 times, if it does not succeed, return None
        # sample a perturbed insert_pos
        perturbed_insert_xyz_copy[:2] = perturb_2d(original_insert_xyz[:2], radius=2.5)

        for rand_alpha in alphas:
            insert_box_copy = copy.deepcopy(insert_box)
    
            # simulate moving the box there
            desired_viewing_angle = compute_viewing_angle(perturbed_insert_xyz_copy[:2])
            rotation_align = -(desired_viewing_angle - current_viewing_angle) # negative sign because gamma increases clockwise
            rotation_align-=rand_alpha
            insert_box_copy.translate(-insert_box_copy.center)
            insert_box_copy.rotate(pyquaternion_from_angle(rotation_align))
            insert_box_copy.translate(np.array([perturbed_insert_xyz_copy[0], perturbed_insert_xyz_copy[1], insert_box.center.reshape(-1)[2]]))

            insert_box_2d = (insert_box_copy.corners().T[[0,1,5,4], :2]).astype(np.float32)
            assert(isinstance(insert_box_2d, np.ndarray))

            if np.sum(points_in_box(insert_box_copy, dataset.point_cloud_dataset.other_background_points[:,:3].T, wlh_factor = 1.0))>0:
                continue
            
            if len(other_boxes)!=0:
                if not is_overlap_shapely(insert_box_2d, other_boxes):
                    got_candidate=True
                    target_rand_alpha = rand_alpha
                    break
            else:
                got_candidate=True
                target_rand_alpha = 0
                break
        # exit the loop if we have found a valid insert pos
        if got_candidate:
            break

    
    if not got_candidate:
        print("WARNING: no valid insert pos")
        return None
        

    perturbed_insert_xyz_pos = perturbed_insert_xyz_copy
    print(f"+++++ perturbed_insert_xyz_pos: {perturbed_insert_xyz_pos}")

    return perturbed_insert_xyz_pos, target_rand_alpha


def insertion_vehicles_driver_perturbed(inpainted_points, allocentric_dict, voxels_occupancy_has, names, dataset, voxelizer, token2sample_dict, args, save_lidar_path, mode="spherical"):
    '''
    Driver method of inserting vehicle with slightly perturbed position and allocentric angle


    - allocentric dict: contains all information of the completed point clouds
    - voxels_occupancy_has: (1, #r, #theta, #z), occupancy grid of the scene
    #- names: list of sampled vehicle names
    - dataset: PolarDataset
    - voxelizer: Voxelizer
    - token2sample_dict: modify this in place and save it, maps from lidar sample token to generated data we will use
    - voxels_occupancy_has: occupancy grid of shape (1,#r, #theta, #z), make a copy and modify out of place
    - save_lidar_path: the full path to the where the data is save
    - mode: "spherical"
    '''
    assert(mode=="spherical")
    assert(names is None)

    # make a copy of the voxel occupancy
    voxels_occupancy_has = torch.clone(voxels_occupancy_has)

    new_bboxes = []
    new_obj_ann_token_list = []
    new_ann_info_list=[]

    #(obj_point_cloud_list, obj_name_list, points_3D, obj_allocentric_list, obj_centers_list, obj_boxes_list, obj_gamma_list, kitti_boxes_list, lidar_sample_token, boxes, obj_ann_token_list, sample_annotation_tokens, sample_records, obj_ann_info_list, curr_sample_table_token)
    dataset_obj_boxes_list = dataset.obj_properties[5][:3] #5
    original_box_idxs = [i for i, box in enumerate(dataset_obj_boxes_list) if (box.name in vehicle_names and vehicle_names[box.name] in {"bus", "car", "truck"})]
    dataset_obj_ann_token_list = [dataset.obj_properties[10][i] for i in original_box_idxs] #10
    original_vehicle_boxes = [dataset_obj_boxes_list[i] for i in original_box_idxs]

    names = [vehicle_names[box.name] for box in original_vehicle_boxes]
    assert(len(dataset_obj_ann_token_list)==len(original_vehicle_boxes))
    print(f"##### insert vehicle name: {names}")


    kd_tree = KDTree(dataset.points_xyz[:,:3])

    # prevent error when not entering the loop (empty names)
    new_scene_points_xyz = dataset.points_xyz[:,:3]

    new_points_xyz_no_resampling_occlusion = inpainted_points
    
    
    print("length_names", len(names))

    # iterate over each object we want to insert
    for i, name in enumerate(names):
        
        obj_properties = allocentric_dict[name] # the properties of objects belonging to the specified name
        allocentric_angles = (obj_properties[0]) #np.array(obj_properties[0])
        pc_filenames = obj_properties[1]
        viewing_angles = obj_properties[2] #np.array(obj_properties[2])
        boxes = obj_properties[3]
        center3Ds = obj_properties[4] #np.array(obj_properties[4])
        obj_ann_tokens = obj_properties[7]
        is_trains = obj_properties[8]
        ann_info_list = obj_properties[10]
        obj_lidar_sample_token_list = obj_properties[6]
        # is_good_completion_indicator_list = obj_properties[11]
        box_volume_list = obj_properties[12]
        print(name)

        N = len(allocentric_angles)
        assert(len(allocentric_angles)==len(pc_filenames)==len(viewing_angles)==len(boxes)==len(center3Ds))

        ####### choose an object
        original_box = original_vehicle_boxes[i]
        original_volume = np.prod(original_box.wlh)
        chosen_idx = np.argmin(np.abs((box_volume_list)-original_volume))

        pc_filename = pc_filenames[chosen_idx]
        bbox = boxes[chosen_idx]
        center3D = center3Ds[chosen_idx]
        pc_path = os.path.join(args.pc_path, name)
        new_obj_ann_token = obj_ann_tokens[chosen_idx]
        new_ann_info = ann_info_list[chosen_idx]
        current_viewing_angle = viewing_angles[chosen_idx]


        use_dense = args.dense==1
        #if use_dense and is_good_completion_indicator_list[chosen_idx]:
        if use_dense:
            pc_full_path = os.path.join(args.pc_path, "dense_nusc", name, pc_filename)
        else:
            print("using incomplete point cloud......")
            pc_full_path = os.path.join(pc_path, pc_filename)
            #raise Exception("I prefer completed point cloud LOL")
        vehicle_pc = np.asarray(open3d.io.read_point_cloud(pc_full_path).points) #np.load(pc_full_path)

        ###### insert it right at the position of the original box in the scene before foreground removal
        original_alpha, original_gamma, original_center = angles_from_box(original_box)
        insert_xyz_pos = original_center.reshape(-1)
        insert_xyz_pos[2]-= float(original_box.wlh[2])/2.0 # bottom of the box center

        ############## IDK WHY THIS ALLOCENTRIC ANGLE ALIGN IS WRONG ###########
        # ####### align the allocentric angle with the original box first
        # desired_allocentric_angle = original_alpha
        # current_allocentric_angle = allocentric_angles[chosen_idx]
        # tmp_new_center = np.copy(bbox.center)
        # bbox.translate(-tmp_new_center)
        # bbox.rotate(pyquaternion_from_angle(-(desired_allocentric_angle - current_allocentric_angle)))
        # bbox.translate(tmp_new_center)
        
        # ##### also do this on the vehicle Pc
        # vehicle_pc = vehicle_pc-center3D
        # # ### We have to align the vehicle_pc orientation / edit the allocentric angle by rotation_angle
        # vehicle_pc = cart2polar(vehicle_pc, mode=mode)
        # theta = vehicle_pc[:,1]
        # theta = theta - (desired_allocentric_angle - current_allocentric_angle)
        # theta[theta<0] += 2*np.pi
        # theta = theta%(2*np.pi)
        # vehicle_pc[:,1] = theta
        # vehicle_pc = polar2cart(vehicle_pc, mode=mode)
        # vehicle_pc += center3D

        # print("############## visualizing insert bbox")
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(vehicle_pc)
        # pcd_colors = np.tile(np.array([[0,0,1]]), (len(vehicle_pc), 1))
        # pcd.colors = open3d.utility.Vector3dVector(pcd_colors)

        # lines = [[0, 1], [1, 2], [2, 3], [0, 3],
        #     [4, 5], [5, 6], [6, 7], [4, 7],
        #     [0, 4], [1, 5], [2, 6], [3, 7]]
        # visboxes = []
        # for box in [bbox]:
        #     line_set = open3d.geometry.LineSet()
        #     line_set.points = open3d.utility.Vector3dVector(box.corners().T)
        #     line_set.lines = open3d.utility.Vector2iVector(lines)
        #     colors = [[1, 0, 0] for _ in range(len(lines))]
        #     line_set.colors = open3d.utility.Vector3dVector(colors)
        #     visboxes.append(line_set)

        # open3d.visualization.draw_geometries([pcd]+visboxes)


        ###### sample a perturbed position and a small random alpha offset
        preturbed_result = sample_perturb_insert_pos(insert_xyz_pos, current_viewing_angle, dataset, bbox, new_bboxes)
        if preturbed_result is None:
            print("**** warning: skip current vehicle due to unsuccessful perturbation")
            continue
        insert_xyz_pos, rand_alpha_offset = preturbed_result

        ###### get the rotation needed to align the viewing angle and to add the random alpha offset
        desired_viewing_angle = compute_viewing_angle(insert_xyz_pos[:2])
        current_viewing_angle = viewing_angles[chosen_idx]
        rotation_align = -(desired_viewing_angle - current_viewing_angle) # negative sign because gamma increases clockwise
        rotation_align-= (rand_alpha_offset)

        new_scene_points_xyz, new_occupancy, new_bbox, insert_xyz_pos, vehicle_pc = insert_vehicle_pc(vehicle_pc, bbox, insert_xyz_pos, rotation_align, voxels_occupancy_has, voxelizer, dataset, mode=mode, use_ground_seg=True, center=center3D, kitti2nusc=False, use_dense=use_dense)
        
        ### visualize without occlusion nor resampling
        new_points_xyz_no_resampling_occlusion = np.concatenate((new_points_xyz_no_resampling_occlusion, vehicle_pc), axis=0)


        voxels_occupancy_has = torch.tensor(new_occupancy).unsqueeze(0)
        new_bboxes.append(new_bbox)
        new_obj_ann_token_list.append(new_obj_ann_token)
        new_ann_info_list.append(new_ann_info)

        

    ### remove new bboxes that contain no points after applying occlusion
    new_bboxes_copy = [copy.deepcopy(box) for box in new_bboxes]
    new_obj_ann_token_list_copy = [ann_token for ann_token in new_obj_ann_token_list]
    new_ann_info_list_copy = [copy.deepcopy(ann_info) for ann_info in new_ann_info_list]
    new_bboxes = []
    new_obj_ann_token_list = []
    new_ann_info_list = []
    for i, box in enumerate(new_bboxes_copy):
        mask = points_in_box(box, new_scene_points_xyz.T, wlh_factor = 1.0)
        if np.sum(mask)!=0: # and np.sum(mask)>=50:
            new_bboxes.append(box)
            new_obj_ann_token_list.append(new_obj_ann_token_list_copy[i])
            new_ann_info_list.append(new_ann_info_list_copy[i])
    assert(len(new_bboxes)==len(new_obj_ann_token_list)==len(new_ann_info_list))

    ### add nosiy points to duplicate
    # noises = np.random.normal(loc=0.0, scale=0.001, size=(len(new_scene_points_xyz),3))
    # new_scene_points_add = np.copy(new_scene_points_xyz) + noises
    # new_scene_points_xyz = np.concatenate((new_scene_points_xyz, new_scene_points_add), axis=0)

    ######## visualize without resampling and occlusion
    # print("############## visualizing inserted cars with no resampling nor occlusion")
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(np.array(new_points_xyz_no_resampling_occlusion))
    # pcd_colors = np.tile(np.array([[0,0,1]]), (len(new_points_xyz_no_resampling_occlusion), 1))
    # mask_vehicle = np.ones((len(new_points_xyz_no_resampling_occlusion),))==0
    # for i, box in enumerate(new_bboxes_copy):
    #     mask = points_in_box(box, new_points_xyz_no_resampling_occlusion.T, wlh_factor = 1.0)
    #     mask_vehicle = mask_vehicle | mask
    # pcd_colors[mask_vehicle==1, 0] = 1
    # pcd_colors[mask_vehicle==1, 2] = 0
    # pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
    # mat = open3d.visualization.rendering.MaterialRecord()
    # mat.shader = 'defaultUnlit'
    # mat.point_size = 3.0

    # ground_pcd = open3d.geometry.PointCloud()
    # ground_pcd.points = open3d.utility.Vector3dVector(dataset.point_cloud_dataset.ground_points[:,:3][:,:3])
    # ground_pcd_colors = np.tile(np.array([[0,1,0]]), (len(dataset.point_cloud_dataset.ground_points[:,:3][:,:3]), 1))
    # ground_pcd.colors = open3d.utility.Vector3dVector(ground_pcd_colors)

    # lines = [[0, 1], [1, 2], [2, 3], [0, 3],
    #      [4, 5], [5, 6], [6, 7], [4, 7],
    #      [0, 4], [1, 5], [2, 6], [3, 7]]
    # visboxes = []
    # for box in new_bboxes_copy:
    #     line_set = open3d.geometry.LineSet()
    #     line_set.points = open3d.utility.Vector3dVector(box.corners().T)
    #     line_set.lines = open3d.utility.Vector2iVector(lines)
    #     colors = [[1, 0, 0] for _ in range(len(lines))]
    #     line_set.colors = open3d.utility.Vector3dVector(colors)
    #     visboxes.append(line_set)

    # open3d.visualization.draw_geometries([pcd, ground_pcd]+visboxes)


    # print("############## visualizing inserted cars with POST-PROCESSING i.e. OCCLUSION")
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(np.array(new_scene_points_xyz))
    # pcd_colors = np.tile(np.array([[0,0,1]]), (len(new_scene_points_xyz), 1))
    # mask_vehicle = np.ones((len(new_scene_points_xyz),))==0
    # for i, box in enumerate(new_bboxes):
    #     mask = points_in_box(box, new_scene_points_xyz.T, wlh_factor = 1.0)
    #     mask_vehicle = mask_vehicle | mask
    # pcd_colors[mask_vehicle==1, 0] = 1
    # pcd_colors[mask_vehicle==1, 2] = 0
    # pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
    # mat = open3d.visualization.rendering.MaterialRecord()
    # mat.shader = 'defaultUnlit'
    # mat.point_size = 3.0
    # open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], show_skybox=False)

    
    ## get nearest neighbor intensity
    original_points = dataset.points_xyz #(N,5)
    _, nearest_idxs = kd_tree.query(new_scene_points_xyz[:,:3], k=1)
    # make the point dim be 5
    extras = np.zeros((len(new_scene_points_xyz), 2))
    new_points_xyz = np.concatenate((new_scene_points_xyz, extras), axis=1)
    assert(new_points_xyz.shape[-1]==5)

    new_points_xyz[:,3] = original_points[:,3][nearest_idxs].astype(np.float64)
    new_points_xyz[:,4] = 0.0 #original_points[0,4]

    points_xyz = new_scene_points_xyz
    bounding_boxes = new_bboxes
    lidar_sample_token = dataset.lidar_sample_token

    ########### nuscens get_sample_data method has converted the box from global coodinates to lidar coordinates as follows:
    ######### This may be useful: you can convert the box coordinates back to global coordinates instead of lidar coordinates
    sample_records = dataset.obj_properties[12]
    cs_record, sensor_record, pose_record = sample_records["cs_record"], sample_records["sensor_record"], sample_records["pose_record"]

    ############# save point cloud 
    pc_name = f'{args.split}_{lidar_sample_token}.bin'
    os.makedirs(os.path.join(save_lidar_path, "lidar_point_clouds"), exist_ok=True)
    lidar_full_path = os.path.join(save_lidar_path, "lidar_point_clouds", pc_name)
    #assert(not os.path.exists(lidar_full_path))
    new_points_xyz.astype(np.float32).tofile(lidar_full_path)

    ############## Save the data needed to build the new database
    token2sample_dict[lidar_sample_token] = (lidar_full_path, bounding_boxes, new_obj_ann_token_list, sample_records, new_ann_info_list)

    return new_scene_points_xyz, new_bboxes, token2sample_dict, voxels_occupancy_has, original_vehicle_boxes
