import numpy as np
import copy


from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
import open3d
import os
import pickle

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
        scale = bbox[3, 0] - bbox[0, 0]
        scale = np.sqrt(np.sum((bbox[3,:2] - bbox[1,:2])**2))/2
        bbox /= scale
        ptcloud = np.dot(ptcloud - center, rotation) / scale
        ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        data[self.ptcloud_key] = ptcloud
        return data, center, scale

    def inverse(ptcloud, bbox):
        bbox = copy.deepcopy(bbox)
        center = (bbox.min(0) + bbox.max(0)) / 2
        
        bbox -= center
        
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        scale = np.sqrt(np.sum((bbox[3,:2] - bbox[1,:2])**2))/2
        bbox /= scale

        ptcloud = np.dot(ptcloud,np.array( [[1, 0, 0], [0, 0, 1], [0, 1, 0]]).T)
        ptcloud *= scale
        ptcloud = np.dot(ptcloud, rotation.T)
        ptcloud += center

        return ptcloud

def nusc2kitti_box_for_pc_completion_normalize(nusc_box):
    '''
    nusc_box is the Nuscenes' Box class
    '''
    kitti_box = copy.deepcopy(nusc_box).corners().T
    # convert nuscenes bounding box to kitti's bounding box
    kitti_box[:,:] = kitti_box[[2,3,7,6,1,0,4,5],:]
    return kitti_box

def rotate(points, axis=(0, 0, 1), angle=np.pi/2):
    kitti_to_nu_lidar = Quaternion(axis=axis, angle=angle)
    kitti_to_nu_lidar_mat = (np.array(kitti_to_nu_lidar.rotation_matrix))
    points = np.matmul(kitti_to_nu_lidar_mat, points.T).T
    return points

