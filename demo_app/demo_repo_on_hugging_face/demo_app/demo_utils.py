
import copy
import os
import torch
import numpy as np
import argparse

from datasets.data_utils import *
from datasets.data_utils_nuscenes import get_obj_regions, pyquaternion_from_angle
from datasets.dataset import Voxelizer, PolarDataset, collate_fn_BEV, get_BEV_label
from datasets.dataset_nuscenes import Nuscenes, NuscenesForeground, vehicle_names

import configs.nuscenes_config as config

from models.vqvae_transformers import VQVAETrans, MaskGIT, voxels2points
import open3d
import pickle
import logging
import timeit

from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_classes import Box
from shapely.geometry import Polygon

############ app
import gradio as gr
import plotly.graph_objects as go


############################################################################
#############################  HELPER METHOD   #############################
############################################################################

def create_logger(logging_root, log_name):
    logger = logging.getLogger('my_logger')
    # Set the default logging level (this can be adjusted as needed)
    logger.setLevel(logging.DEBUG)
    # Create two handlers for logging to two different files
    file_handler1 = logging.FileHandler(os.path.join(logging_root, log_name))
    # Set the log level for each handler (optional)
    file_handler1.setLevel(logging.DEBUG)   # For detailed logging
    # Create a formatter to define the log message format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Assign the formatter to both handlers
    file_handler1.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(file_handler1)

    # Example logging
    # self.logger.debug("This is a debug message")
    # self.logger.info("This is an info message")
    # self.logger.warning("This is a warning message")
    # self.logger.error("This is an error message")
    # self.logger.critical("This is a critical message")
    logger.info("\n")
    logger.info(">>>>>>> START LOGGING <<<<<<<<")
    return logger


def pcd_ize(points, colors):
    '''
    convert np array point cloud to open3d pcd
    color: (N,3)
    points: (N,3)
    '''
    
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(points))
    if colors is not None:
        colors = np.array(colors)
        if len(colors.shape)==1:
            colors = colors[np.newaxis,:]
        pcd.colors = open3d.utility.Vector3dVector(colors)
    return pcd

def bbox2lineset(bbox):
    '''
    convert Nuscenes bounding box object to open3d lineset for visualization. Optionally, it can be a list of bounding boxes
    '''
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
    [4, 5], [5, 6], [6, 7], [4, 7],
    [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [[1, 0, 0] for _ in range(len(lines))]
    if isinstance(bbox, list):
        visboxes = []
        for box in bbox:
            line_set = open3d.geometry.LineSet()
            line_set.points = open3d.utility.Vector3dVector(box.corners().T)
            line_set.lines = open3d.utility.Vector2iVector(lines)
            # if box.name in vehicle_names:
            #     if vehicle_names[box.name]=="car":
            #         colors = [[1, 0, 0] for _ in range(len(lines))]
            #     elif vehicle_names[box.name]=="truck":
            #         colors = [[0, 1, 0] for _ in range(len(lines))]
            #     elif vehicle_names[box.name]=="bus":
            #         colors = [[0, 0, 1] for _ in range(len(lines))]
            #     else:
            #         colors = [[1, 0, 1] for _ in range(len(lines))]
            # else:
            #         colors = [[1, 0.647, 0] for _ in range(len(lines))]
            line_set.colors = open3d.utility.Vector3dVector(colors)
            visboxes.append(line_set)
        return visboxes
    else:
        line_set = open3d.geometry.LineSet()
        line_set.points = open3d.utility.Vector3dVector(box.corners().T)
        line_set.lines = open3d.utility.Vector2iVector(lines)
        # if box.name in vehicle_names:
        #     if vehicle_names[box.name]=="car":
        #         colors = [[1, 0, 0] for _ in range(len(lines))]
        #     elif vehicle_names[box.name]=="truck":
        #         colors = [[0, 1, 0] for _ in range(len(lines))]
        #     elif vehicle_names[box.name]=="bus":
        #         colors = [[0, 0, 1] for _ in range(len(lines))]
        #     else:
        #         colors = [[1, 0, 1] for _ in range(len(lines))]
        # else:
        #         colors = [[1, 0.647, 0] for _ in range(len(lines))]
        line_set.colors = open3d.utility.Vector3dVector(colors)
        return line_set

def get_points_in_box_mask(points_xyz, bbox_list):
    '''
    Args:
        - points_xyz: (N,3)
        - bbox_list: a list of nucenes bounding box objects
    Return:
        - points_in_box_mask: (N,), which points are in a bounding box
        - points_in_boxes: a list of point clouds, each point cloud is in the corresponding bounding box in bbox_list
    '''
    points_3D = points_xyz[:,:3]
    N = len(points_xyz)
    points_in_box_mask = np.zeros((N,)).astype(int)
    points_in_boxes = []
    nonempty_boxes = []
    for i, box in enumerate(bbox_list):
        mask = points_in_box(box, points_3D.T, wlh_factor = 1.0).astype(int)
        points_in_boxes.append(np.copy(points_3D[mask==1]))
        points_in_box_mask = points_in_box_mask | mask
    return points_in_box_mask, points_in_boxes

def angles_from_box(box):
    '''
    Get the viewing angle, allocentric angle and the center coordinates of the box
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

def get_voxel_occlusion_mask(obj_region_list, voxelizer, use_z=True):
    '''
    Get the mask over voxels occluded by object regions

    Args:
        - obj_region_list: a list of (2,3) array, each array is an obj region
        - voxelizer: Voxelizer

    Return:
        - voxels_labels: the voxels that are occluded by the obj_region are labeled 1, otherwise 0. Shape (#r,#theta,#z)
        - BEV_labels: Bird's eye view of voxels_labels, Shape (#r, #Theta)
        - obj_voxels_mask_list: a list of voxel mask, each voxel mask is a mask over voxels occluded by the corresponding object in obj_region_list
    '''
    obj_voxels_mask_list = []
    if len(obj_region_list)!=0:
        if True:
            voxels_labels = voxelizer.create_mask_by_occlusion(obj_region_list[0], use_z=use_z)
            for i in range(len(obj_region_list)):
                obj_voxel_mask = voxelizer.create_mask_by_occlusion(obj_region_list[i], use_z=use_z)
                voxels_labels += obj_voxel_mask
                obj_voxels_mask_list.append(obj_voxel_mask)
            voxels_labels = (voxels_labels>=1).astype(np.int64)
    BEV_labels = get_BEV_label(voxels_labels)

    return voxels_labels, BEV_labels, obj_voxels_mask_list

def filter_in_bound_points(points_xyz, voxelizer, mode="spherical"):
    points_within_bound_mask = voxelizer.filter_in_bound_points(cart2polar(points_xyz, mode=mode))
    in_bound_points_xyz = points_xyz[points_within_bound_mask]
    return in_bound_points_xyz, points_within_bound_mask


################################################################################
######################## INPAINTING ###########################################
################################################################################

def prepare_inpainting_input(scene_points_xyz, bbox_list, voxelizer, device, mode="spherical"):
    '''
    Args: 
        - points_xyz:3D point cloud (np array) of the current lidar_scene, assuming points_xyz are within bound
    Return:
        - occupancy grid and occlusion masks
    '''
    points_in_box_mask, points_in_boxes = get_points_in_box_mask(scene_points_xyz, bbox_list)
    obj_region_list = get_obj_regions(bbox_list, mode=mode, points_in_boxes=points_in_boxes)
    voxels_labels, BEV_labels, obj_voxels_mask_list = get_voxel_occlusion_mask(obj_region_list, voxelizer, use_z=True)

    voxels_labels = torch.tensor(voxels_labels).float().unsqueeze(0).to(device) # (1, H, W, in_chans)
    BEV_labels = torch.tensor(BEV_labels).float().unsqueeze(0).to(device) # (1, H, W)

    scene_points_polar = cart2polar(scene_points_xyz[:,:3], mode=mode)
    _, _, _, voxels_occupancy = voxelizer.voxelize(scene_points_polar, return_point_info=False) #(H, W, in_chans)
    voxels_occupancy = torch.tensor(voxels_occupancy).unsqueeze(0).permute(0,3,1,2).to(device).float() #(1, in_chans, H, W)

    return voxels_occupancy, voxels_labels, BEV_labels


def pcd_ize_inpainted_pointcloud(gen_voxels_occupancy, voxels_labels, voxelizer, mode="spherical"):
    '''
    Args:
        - gen_voxels_occupancy: (1, in_chans, H, W)
        - voxels_labels: (1, H, W, in_chans), the occlusion mask
    Return:
        - open 3d point cloud of the inpainted point cloud, and the np array point cloud
    '''

    ######## get points, occupancy, occlusion mask without batch dimension
    gen_points_xyz = voxels2points(voxelizer, gen_voxels_occupancy, mode=mode)[0]
    voxels_occupancy = gen_voxels_occupancy[0].permute(1,2,0) #(H, W, C)
    non_zero_indices = torch.nonzero(voxels_occupancy.detach().cpu(), as_tuple=True)
    voxels_mask = voxels_labels[0].detach().cpu() #(H, W, C)

    ###### color occluded points red
    N = len(gen_points_xyz)
    assert(N==len(voxels_mask[non_zero_indices[0], non_zero_indices[1], non_zero_indices[2]]))
    occluded_point_mask = (voxels_mask[non_zero_indices[0], non_zero_indices[1], non_zero_indices[2]] == 1).detach().cpu().numpy()
    
    pcd_colors = np.tile(np.array([[1,1,1]]), (N, 1))
    pcd_colors[occluded_point_mask, 0] = 0
    pcd_colors[occluded_point_mask, 2] = 0

    return pcd_ize(gen_points_xyz, colors=pcd_colors), gen_points_xyz


def inpainting_driver(mask_git, scene_points_xyz, bbox_list, voxelizer, device, mode="spherical"):
    '''
    remove objects from  scene_points_xyz specified by bbox_list and inpaint the gap left behind

    Args:
        - mask_git: mask git model
        etc.

    Return:
        voxels after object removal and inpainting: Shape (1, in_chans, H, W)
        gen_points_xyz: (N, 3)
        gen_pcd: open3d pcd object of the generated point cloud
    '''
    device = device
    voxels_occupancy, voxels_labels, BEV_labels = prepare_inpainting_input(scene_points_xyz, bbox_list, voxelizer, device, mode=mode)
    if torch.sum(voxels_labels)==0:
        return None
    gen_voxels = mask_git.iterative_generation_driver(voxels_labels, voxels_occupancy, BEV_labels, generation_iter=20, denoise_iter=0, mode=mode) #(1, in_chans, H, W)
    
    gen_pcd, gen_points_xyz = pcd_ize_inpainted_pointcloud(gen_voxels, voxels_labels, voxelizer, mode=mode)

    return gen_voxels, gen_points_xyz, gen_pcd

################################################################################
######################## INSERTION ###########################################
################################################################################
def rotate_vehicle_pc_allocentric(vehicle_pc, center, angle, mode="spherical"):
    '''
    rotate the vehicle_pc about z-axis at its center by angle

    Args:
        - vehicle_pc: (N,3), cartesian coordinates
        - center: (3,), 3D center of the point cloud. It comes from the point cloud's bounding box
        - angle (float): the angle to rotate
    '''
    vehicle_pc = np.copy(vehicle_pc)
    vehicle_pc = vehicle_pc-center
    # ### We have to align the vehicle_pc orientation / edit the allocentric angle by rotation_angle
    vehicle_pc = cart2polar(vehicle_pc, mode=mode)
    theta = vehicle_pc[:,1]
    theta = theta + angle
    theta[theta<0] += 2*np.pi
    theta = theta%(2*np.pi)
    vehicle_pc[:,1] = theta
    vehicle_pc = polar2cart(vehicle_pc, mode=mode)
    vehicle_pc += center

    return vehicle_pc

def rotate_bbox_allocentric(bbox, angle):
    '''
    rotate the bounding box about z-axis at its center by angle
    '''
    bbox = copy.deepcopy(bbox)
    tmp_new_center = np.copy(bbox.center)
    bbox.translate(-tmp_new_center)
    bbox.rotate(pyquaternion_from_angle(angle))
    bbox.translate(tmp_new_center)

    return bbox

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

def box_is_collision_free(insert_box, other_boxes, other_background_points):
    '''
    Check insert_box collision with other_boxes or background points other than drivable surface
    '''
    
    insert_box_copy = copy.deepcopy(insert_box)
    other_boxes = [(copy.deepcopy(other_box).corners().T[[0,1,5,4], :2]).astype(np.float32) for other_box in other_boxes]
    insert_box_2d = (insert_box_copy.corners().T[[0,1,5,4], :2]).astype(np.float32)
    assert(isinstance(insert_box_2d, np.ndarray))

    if np.sum(points_in_box(insert_box_copy, other_background_points[:,:3].T, wlh_factor = 1.0))>0:
        return False
    
    if len(other_boxes)==0:
        return True
    
    # if len other_boxes is not zero and insert_box does not collide with other background points
    return not is_overlap_shapely(insert_box_2d, other_boxes)

def insert_vehicle_pc(vehicle_pc, bbox, insert_xyz_pos, rotation_angle, voxels_occupancy_has, voxelizer, ground_points, other_background_points, curr_bbox_list, mode='spherical', center=None):
    '''
    insert a completeed vehicle_pc into the scene point cloud

    - vehicle_pc: (N,3)
    - Nuscenes bounding box of the vehicle_pc
    - insert_xyz_pos: (3,), first two elements are the x-y pos to place the vehicle, the third element can be any value because the z coordinate for insertion is going to be determined in this method.
    - rotation_angle: the angle in radian to rotate the point cloud to align orientations / insert the vehicle
    - voxels_occupancy_has: (1, #r, #theta, #z), occupancy grid of the scene
    - voxelizer: datasets.Voxelizer
    - ground_points: np array of shape (G,3) containing points on drivable surface
    - other_background_points: background points other than drivable surface
    - current_bbox_list: list of bounding boxes existing in the scene
    - mode: either "polar" or "spherical
    - center: center the point cloud at the center or not, either None or (3,) shape array


    return:
        new_scene_points_xyz, new_bbox, insert_xyz_pos, vehicle_pc. If insertion is unsuccessful, return None and the failure status
    '''
   
    # pass object variables by copy
    new_bbox = copy.deepcopy(bbox)
    vehicle_pc = np.copy(vehicle_pc)
    insert_xyz_pos = np.copy(insert_xyz_pos)
    voxels_occupancy_has = torch.clone(voxels_occupancy_has)

    assert(center is not None)

    # ### We have to align the vehicle_pc orientation by rotation_angle
    vehicle_pc = rotate_vehicle_pc_allocentric(vehicle_pc, center, angle=rotation_angle, mode=mode)

     #### shift vehicle to the desired position
    vehicle_pc = vehicle_pc - center
    vehicle_pc[:,:2] += insert_xyz_pos[:2]

    # print("############## inserted: visualizing insert bbox")
    # vpcd = open3d.geometry.PointCloud()
    # vpcd.points = open3d.utility.Vector3dVector(vehicle_pc)
    # vpcd_colors = np.tile(np.array([[1,0,0]]), (len(vehicle_pc), 1))
    # vpcd.colors = open3d.utility.Vector3dVector(vpcd_colors)
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
    # open3d.visualization.draw_geometries([vpcd]+visboxes)
    # print("############## visualizing inserted cars with no resampling nor occlusion")
    # new_scene_points_xyz = voxels2points(voxelizer, voxels=voxels_occupancy_has.permute(0,3,1,2), mode=mode)[0]
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(np.array(new_scene_points_xyz))
    # pcd_colors = np.tile(np.array([[0,0,1]]), (len(new_scene_points_xyz), 1))
    # mask_vehicle = np.ones((len(new_scene_points_xyz),))==0
    # for i, box in enumerate([new_bbox]):
    #     mask = points_in_box(box, new_scene_points_xyz.T, wlh_factor = 1.0)
    #     mask_vehicle = mask_vehicle | mask
    # pcd_colors[mask_vehicle==1, 0] = 1
    # pcd_colors[mask_vehicle==1, 2] = 0
    # pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
    # open3d.visualization.draw_geometries([pcd, vpcd]+visboxes)

    
    ##### find nearest drivable surface point
    nearest_polar_voxels_pos = voxelizer.get_nearest_ground_BEV_pos(ground_points, cart2polar(insert_xyz_pos[np.newaxis,:], mode=mode), mode)

    ##### determine z coordinate of the inserted vehicle
    nearest_cart_voxels_pos = polar2cart(nearest_polar_voxels_pos, mode=mode)
    nearest_min_z = np.min(nearest_cart_voxels_pos[:,2])
    vehicle_min_z = np.min(vehicle_pc[:,2])

    height_diff = nearest_min_z - vehicle_min_z
    vehicle_pc[:,2] += height_diff
    insert_xyz_pos[2]=height_diff

    ############ transform the bounding box according to rotation_angle and insert_xyz_pos
    new_bbox.translate(-bbox.center)
    new_bbox.rotate(pyquaternion_from_angle(rotation_angle))
    new_bbox.translate(insert_xyz_pos)

    #### project to spherical grid, apply occlusion and convert back to point cloud
    polar_vehicle = cart2polar(vehicle_pc, mode=mode)     
    
    #### ensure the vehicle is within bound
    if np.sum(voxelizer.filter_in_bound_points(polar_vehicle))<=len(polar_vehicle)/2:
        print(np.rad2deg(polar_vehicle[np.logical_not(voxelizer.filter_in_bound_points(polar_vehicle))][:,1:3]))
        return None, 0
    elif not box_is_collision_free(insert_box=new_bbox, other_boxes=curr_bbox_list, other_background_points=other_background_points):
        if np.sum(points_in_box(new_bbox, other_background_points[:,:3].T, wlh_factor = 1.0))>0:
            return None, 1
        return None, 2
    
    new_occupancy = voxelizer.voxelize_and_occlude(voxels_occupancy_has[0].cpu().detach().numpy(), polar_vehicle, insert_only=False) #(H, W, C)
   
    new_scene_points_xyz = voxels2points(voxelizer, voxels=torch.tensor(new_occupancy).permute(2,0,1).unsqueeze(0), mode=mode)[0]

    # print("############## inserted: visualizing insert bbox")
    # vpcd = open3d.geometry.PointCloud()
    # vpcd.points = open3d.utility.Vector3dVector(vehicle_pc)
    # vpcd_colors = np.tile(np.array([[1,0,0]]), (len(vehicle_pc), 1))
    # vpcd.colors = open3d.utility.Vector3dVector(vpcd_colors)
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
    # open3d.visualization.draw_geometries([vpcd]+visboxes)
    # print("############## visualizing inserted cars with no resampling nor occlusion")
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(np.array(new_scene_points_xyz))
    # pcd_colors = np.tile(np.array([[0,0,1]]), (len(new_scene_points_xyz), 1))
    # mask_vehicle = np.ones((len(new_scene_points_xyz),))==0
    # for i, box in enumerate([new_bbox]):
    #     mask = points_in_box(box, new_scene_points_xyz.T, wlh_factor = 1.0)
    #     mask_vehicle = mask_vehicle | mask
    # pcd_colors[mask_vehicle==1, 0] = 1
    # pcd_colors[mask_vehicle==1, 2] = 0
    # pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
    # open3d.visualization.draw_geometries([pcd, vpcd]+visboxes)

    return new_scene_points_xyz, new_occupancy, new_bbox, insert_xyz_pos, vehicle_pc


def pcd_ize_inserted_pointcloud(points_xyz, bbox_list):
    '''
    convert point cloud with object inserted to open3d pcd, paint inserted objects red
    '''
    assert(isinstance(points_xyz, np.ndarray))
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points_xyz)
    
    mask_vehicle = np.ones((len(points_xyz),))==0
    for i, box in enumerate(bbox_list):
        mask = points_in_box(box, points_xyz.T, wlh_factor = 1.0)
        mask_vehicle = mask_vehicle | mask

    # pcd_colors = np.tile(np.array([[0,0,1]]), (len(points_xyz), 1))
    # pcd_colors[mask_vehicle==1, 0] = 1
    # pcd_colors[mask_vehicle==1, 2] = 0
    pcd_colors = np.tile(np.array([[1,1,1]]), (len(points_xyz), 1))
    pcd_colors[mask_vehicle==1, 0] = 0
    pcd_colors[mask_vehicle==1, 2] = 0
    pcd.colors = open3d.utility.Vector3dVector(pcd_colors)

    return pcd

def print_log(msg, logger=None):
    print(msg)
    if logger is not None:
        logger.info(msg)


def insertion_vehicles_driver(insert_names, insert_xy_pos_list, insert_alpha_list, voxels_occupancy_has, curr_bbox_list, voxelizer, allocentric_dict, full_obj_pc_path, ground_points, other_background_points, mode="spherical", logger=None):
    '''
    Args: 
        - insert_names: list of N strings, each one is the class of the vehicle to be inserted
        - insert_xy_pos_list: a list of N 2D np array
        - desired_alpha_list: a list of N allocentric angles

        - voxels_occupancy_has: (1, #r, #theta, #z), occupancy grid of the scene
        - voxelizer: Voxelizer
        - allocentric_dict: object library
        etc.
    
    Return:
        - new_scene_points_xyz, new_bbox_list, voxels_occupancy_has (1, #r, #theta, #z), new_points_xyz_no_resampling_occlusion, failure_message_list, failure_indicator_list
    '''
    assert(mode=="spherical")

    voxels_occupancy_has = torch.clone(voxels_occupancy_has)
    new_points_xyz_no_resampling_occlusion = voxels2points(voxelizer, voxels=voxels_occupancy_has.permute(0,3,1,2), mode=mode)[0]
    new_scene_points_xyz = np.copy(new_points_xyz_no_resampling_occlusion)

    new_bbox_list = []
    failure_message_list = ["success" for _ in range(len(insert_names))]
    failure_indicator_list = [0 for _ in range(len(insert_names))]
    
    for insert_idx, name in enumerate(insert_names):
        print(name)

        ########### get object library ############
        obj_properties = allocentric_dict[name] # the properties of objects belonging to the specified name
        allocentric_angles = (obj_properties[0]) #np.array(obj_properties[0])
        pc_filenames = obj_properties[1]
        viewing_angles = obj_properties[2] #np.array(obj_properties[2])
        boxes = obj_properties[3]
        center3Ds = obj_properties[4] #np.array(obj_properties[4])
        
        obj_library_size = len(allocentric_angles)
        assert(len(allocentric_angles)==len(pc_filenames)==len(viewing_angles)==len(boxes)==len(center3Ds))

        ####### choose an object
        chosen_files = [f"sample_{i}.pcd" for i in range(10)]
        chosen_idxs = [pc_filenames.index(pcd_file) for pcd_file in chosen_files]
        chosen_idx = np.random.choice(np.array(chosen_idxs))
        #chosen_idx = np.random.randint(0, obj_library_size)
        ####################### get object from dense reconstructed object library
        pc_filename = pc_filenames[chosen_idx]
        current_bbox = copy.deepcopy(boxes[chosen_idx])
       
        pc_full_path = os.path.join(full_obj_pc_path, "dense_nusc", name, pc_filename)
        vehicle_pc = np.asarray(open3d.io.read_point_cloud(pc_full_path).points)

        current_alpha, current_gamma, current_center = angles_from_box(current_bbox)
        center3D = center3Ds[chosen_idx]
        insert_xy_pos = insert_xy_pos_list[insert_idx]
        insert_xyz_pos = np.copy(current_center.reshape(-1))
        insert_xyz_pos[:2] = insert_xy_pos # set x-y insertion position
        insert_xyz_pos[2]-= float(current_bbox.wlh[2])/2.0 # bottom of the box center

        if logger is not None:
            logger.info(f"---- inserting {name} : vehicle point cloud shape: {vehicle_pc.shape} || insert xyz pos (z temp): {insert_xyz_pos}")
            logger.info(f"current_center={current_center}, center3D={center3D}")


        desired_allocentric_angle = insert_alpha_list[insert_idx]
        desired_viewing_angle = compute_viewing_angle(insert_xyz_pos[:2])

        ############# need to rotate vehicle_pc and current_bbox this much to insert at the specified allocentric angle and x-y position
        rotation_align = 0
        rotation_align += -(desired_viewing_angle - current_gamma) # negative sign because gamma increases clockwise
        rotation_align += -(desired_allocentric_angle - current_alpha)

        # print("############## before insertion: visualizing insert bbox")
        # vpcd = open3d.geometry.PointCloud()
        # vpcd.points = open3d.utility.Vector3dVector(vehicle_pc)
        # vpcd_colors = np.tile(np.array([[1,0,0]]), (len(vehicle_pc), 1))
        # vpcd.colors = open3d.utility.Vector3dVector(vpcd_colors)
        # lines = [[0, 1], [1, 2], [2, 3], [0, 3],
        #     [4, 5], [5, 6], [6, 7], [4, 7],
        #     [0, 4], [1, 5], [2, 6], [3, 7]]
        # visboxes = []
        # for box in [current_bbox]:
        #     line_set = open3d.geometry.LineSet()
        #     line_set.points = open3d.utility.Vector3dVector(box.corners().T)
        #     line_set.lines = open3d.utility.Vector2iVector(lines)
        #     colors = [[1, 0, 0] for _ in range(len(lines))]
        #     line_set.colors = open3d.utility.Vector3dVector(colors)
        #     visboxes.append(line_set)
        # open3d.visualization.draw_geometries([vpcd]+visboxes)


        ###### try insertion and collision checking, if invalid insertion, skip it
        insert_result = insert_vehicle_pc(vehicle_pc, current_bbox, insert_xyz_pos, rotation_align, voxels_occupancy_has, voxelizer, ground_points, other_background_points, new_bbox_list+curr_bbox_list, mode=mode, center=current_center)
        if len(insert_result)==2 and insert_result[0] is None:
            # assert(1==0)
            if insert_result[1]==0:
                print_log("**** warning: skip current vehicle due to out of bound vehicle", logger)
                failure_message_list[insert_idx] = "out of bound insertion"
                failure_indicator_list[insert_idx] = 1
            elif insert_result[1]==1:
                print_log("**** warning: skip current vehicle due to collision with non-ground background points", logger)
                failure_message_list[insert_idx] = "colliding background"
                failure_indicator_list[insert_idx] = 1
            elif insert_result[1]==2:
                print_log("**** warning: skip current vehicle due to collision with other boxes", logger)
                failure_message_list[insert_idx] = "colliding objects"
                failure_indicator_list[insert_idx] = 1

            continue
        
        ###### get new point cloud, voxel grid, new box, new vehicle pc
        new_scene_points_xyz, new_occupancy, new_bbox, insert_xyz_pos, vehicle_pc = insert_result

         ### visualize without occlusion nor resampling
        new_points_xyz_no_resampling_occlusion = np.concatenate((new_points_xyz_no_resampling_occlusion, vehicle_pc), axis=0)
        if logger is not None:
            logger.info(f"after inserting vehicle point cloud shape: {vehicle_pc.shape} || new scene point cloud scene shape: {new_scene_points_xyz.shape}")
        
        voxels_occupancy_has = torch.tensor(new_occupancy).unsqueeze(0)
        new_bbox_list.append(new_bbox)

    ### remove new bboxes that contain no points after applying occlusion
    new_bboxes_copy = [copy.deepcopy(box) for box in new_bbox_list]
    new_bbox_list = []
    for i, box in enumerate(new_bboxes_copy):
        mask = points_in_box(box, new_scene_points_xyz.T, wlh_factor = 1.0)
        if np.sum(mask)!=0: # and np.sum(mask)>=50:
            new_bbox_list.append(box)

    old_bboxes_copy = [copy.deepcopy(box) for box in curr_bbox_list]
    old_bbox_list = []
    for i, box in enumerate(old_bboxes_copy):
        mask = points_in_box(box, new_scene_points_xyz.T, wlh_factor = 1.0)
        if np.sum(mask)!=0: # and np.sum(mask)>=50:
            old_bbox_list.append(box)


    ####### visualize without resampling and occlusion
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
    # open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], show_skybox=False)
    
    return new_scene_points_xyz, new_bbox_list, old_bbox_list, voxels_occupancy_has, new_points_xyz_no_resampling_occlusion, failure_message_list, failure_indicator_list
        

    





















