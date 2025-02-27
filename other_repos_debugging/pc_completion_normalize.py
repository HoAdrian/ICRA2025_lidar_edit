import numpy as np
import copy


from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
import open3d
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors

'''
utilities for normalizing object pose (both position, scale and orientation) using bounding box, to feed into point cloud completion
'''


class NormalizeObjectPose(object):
    def __init__(self, parameters):
        '''
        parameters = {'input_keys':{'ptcloud': key to index point cloud in data dict, 'bbox':key to index bbox in data dict}}
        '''
        input_keys = parameters['input_keys']
        self.ptcloud_key = input_keys['ptcloud']
        self.bbox_key = input_keys['bbox']

    def __call__(self, data):
        ptcloud = data[self.ptcloud_key]
        bbox = data[self.bbox_key]
        bbox = copy.deepcopy(bbox)

        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        #scale = bbox[3, 0] - bbox[0, 0]
        scale = np.max(np.max(bbox, axis=0) - np.min(bbox, axis=0))
        bbox /= scale
        ptcloud = np.dot(ptcloud - center, rotation) / scale
        # print(f"====scale:{scale}")
        # print(f"====norm:{np.max(np.sqrt(np.sum(ptcloud**2, axis=1)))}")
        ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        data[self.ptcloud_key] = ptcloud
        return data, center, scale

    def inverse(ptcloud, bbox):
        '''
        bbox is the original bounding box
        '''
        bbox = copy.deepcopy(bbox)
        center = (bbox.min(0) + bbox.max(0)) / 2
        
        bbox -= center
        
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        #scale = bbox[3, 0] - bbox[0, 0]
        scale = np.max(np.max(bbox, axis=0) - np.min(bbox, axis=0))
        bbox /= scale

        ptcloud = np.dot(ptcloud,np.array( [[1, 0, 0], [0, 0, 1], [0, 1, 0]]).T)
        ptcloud *= scale
        ptcloud = np.dot(ptcloud, rotation.T)
        ptcloud += center

        return ptcloud
    

    
def nusc_box_from_axis_aligned_pc(pc):
    '''
    1. find the nuscenes bounding box containing pc of shape (N,3), assuming pc is axis-aligned
    '''
    xyz_max = np.max(pc, axis=0)
    xyz_min = np.min(pc, axis=0)
    x_max, y_max, z_max = xyz_max
    x_min, y_min, z_min = xyz_min
    corner_0 = np.array([x_max, y_max, z_max])
    corner_1 = np.array([x_max, y_min, z_max])
    corner_2 = np.array([x_max, y_min, z_min])
    corner_3 = np.array([x_max, y_max, z_min])

    corner_4 = np.array([x_min, y_max, z_max])
    corner_5 = np.array([x_min, y_min, z_max])
    corner_6 = np.array([x_min, y_min, z_min])
    corner_7 = np.array([x_min, y_max, z_min])

    corners = np.stack((corner_0, corner_1, corner_2, corner_3, corner_4, corner_5, corner_6, corner_7), axis=0)
    return corners

def nusc2kitti_box_for_pc_completion_normalize(nusc_box):
    '''
    nusc_box is the Nuscenes' Box class
    '''
    if isinstance(nusc_box, np.ndarray):
        kitti_box = nusc_box
    else:
        kitti_box = copy.deepcopy(nusc_box).corners().T
    # convert nuscenes bounding box to kitti's bounding box
    #kitti_box[:,:] = kitti_box[[2,3,7,6,1,0,4,5],:]
    kitti_box[:,:] = kitti_box[[7,6,2,3,4,5,1,0],:]
    return kitti_box

def rotate(points, axis=(0, 0, 1), angle=np.pi/2):
    kitti_to_nu_lidar = Quaternion(axis=axis, angle=angle)
    kitti_to_nu_lidar_mat = (np.array(kitti_to_nu_lidar.rotation_matrix))
    points = np.matmul(kitti_to_nu_lidar_mat, points.T).T
    return points


########################## for visualization ##########################

def kitti_box_yaw_normalization(bbox):
    bbox_rot = copy.deepcopy(bbox)

    center = (bbox_rot.min(0) + bbox_rot.max(0)) / 2
    bbox_rot -= center
    yaw = np.arctan2(bbox_rot[3, 1] - bbox_rot[0, 1], bbox_rot[3, 0] - bbox_rot[0, 0])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    bbox_rot = np.dot(bbox_rot, rotation)

    bbox_scale = copy.deepcopy(bbox_rot)
    scale = bbox_rot[3, 0] - bbox_rot[0, 0]
    #scale = np.sqrt(np.sum((bbox[3,:2] - bbox[1,:2])**2))/2
    bbox_scale /= scale

    return bbox, bbox_rot, bbox_scale

def visualize_kitti_box(corners, title="kitti_box"):
    '''
    corners have shape (N,3)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    colors = np.linspace(0, 1, len(corners))
    ax.scatter(corners[0,0], corners[0,1], corners[0,2],marker='x', s=80)
    scatter = ax.scatter(corners[:,0], corners[:,1], corners[:,2], c=colors, cmap='jet', marker='o', s=50)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Color Scale (Mapped to Z-values)')

    # Labels
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)

    # Create a legend with color swatches
    legend_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])  # Side legend area
    legend_ax.set_xticks([])  # Hide ticks
    legend_ax.set_yticks([])

    norm = mcolors.Normalize(vmin=min(colors), vmax=max(colors))  
    cmap = cm.get_cmap('jet')  
    mapped_colors = [cmap(norm(value)) for value in colors]
    # Plot color boxes and text labels
    for i, (val, color) in enumerate(zip(colors, mapped_colors)):
        legend_ax.add_patch(plt.Rectangle((0, i), 1, 1, color=color))
        legend_ax.text(1.2, i + 0.3, f"{i}", va='center', fontsize=10)

    legend_ax.set_ylim(0, len(colors))  # Adjust limits


    plt.show()

def visualize_boxes(boxes_list, labels, kitti=False):
    '''
    assume each box in boxes list is either Box object from NuScenes or np array of shape (N,3)

    if kitti=True, we assume the corners are arranged in kitti's fashion
    '''
    plt.figure(figsize=(8, 6))
    plt.gca().set_aspect('equal')
    for i, box in enumerate(boxes_list):
        if isinstance(box, np.ndarray):
            assert(box.shape[-1]==3)
            assert(len(box.shape)==2)
            corners = box.T #(3,N)
        else:
            corners = box.corners() #(3,8)
        if not kitti:
            # corner_1 = corners[:,5][:2]
            # corner_2 = corners[:,4][:2]
            # corner_5 = corners[:,6][:2]
            # corner_6 = corners[:,7][:2]
            corner_1 = corners[:,7][:2]
            corner_2 = corners[:,6][:2]
            corner_5 = corners[:,4][:2]
            corner_6 = corners[:,5][:2]
        else:
            corner_1 = corners[:,0][:2]
            corner_2 = corners[:,1][:2]
            corner_5 = corners[:,4][:2]
            corner_6 = corners[:,5][:2]
        rect = patches.Polygon([corner_1, corner_2, corner_6,corner_5], linewidth=1, edgecolor='r', facecolor='none')

        # Get the current axes and plot the polygon patch
        center2D = (corner_1 + corner_6)/2
        center3D = np.array([center2D[0], center2D[1], (corners[2,2]+corners[2,1])/2])

        plt.text(center2D[0], center2D[1], f"{labels[i]}", color='blue', fontsize=10, ha='center', va='center')


        right_pointing_vector = (corner_2 + corner_6)/2.0 - center2D
        front_pointing_vector = (corner_1 + corner_2)/2.0 - center2D
        #obj2cam_vector = -center2D
        
        plt.gca().add_patch(rect)

        # plot box axes
        plt.quiver(center2D[0], center2D[1], 0+right_pointing_vector[0], 0+right_pointing_vector[1], color='r', scale_units='xy', scale=1)
        plt.quiver(center2D[0], center2D[1], 0+front_pointing_vector[0], 0+front_pointing_vector[1], color='g', scale_units='xy',scale=1)
        # plot obj2cam_vector scaled and centered at box center
        #plt.quiver(center2D[0], center2D[1], -center2D[0]/np.linalg.norm(center2D)*200, -center2D[1]/np.linalg.norm(center2D)*200)
        
    plt.show()