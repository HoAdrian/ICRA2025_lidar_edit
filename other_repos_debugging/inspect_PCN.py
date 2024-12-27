import json
import os
import numpy as np

import h5py
import numpy as np
import open3d
import os
import random

random.seed(2021)

############################# normalizing ###########################################
def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m

def get_file_list(partial_points_path, complete_points_path, dataset_categories, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in dataset_categories:
            print('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in samples:
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_path': [
                        partial_points_path % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gt_path':
                    complete_points_path % (subset, dc['taxonomy_id'], s),
                })

        print('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list



############################## inspecting shape net dataset #############################################
shapenet_bus_list = []
nuscenes_bus_list = []

cat = "car"
partial_points_path = "/home/shinghei/lidar_generation/data_anchor/PCN/%s/partial/%s/%s/%02d.pcd"
complete_points_path = "/home/shinghei/lidar_generation/data_anchor/PCN/%s/complete/%s/%s.pcd"
category_file = "/home/shinghei/lidar_generation/data_anchor/PCN/PCN.json"
cars = True
subset = "train" #"test"

# Load the dataset indexing file
dataset_categories = []
with open(category_file) as f:
    dataset_categories = json.loads(f.read())
    if cars:
        dataset_categories = [dc for dc in dataset_categories if dc['taxonomy_id'] == '02958343']

n_renderings = 8 if subset == 'train' else 1
file_list_pcn = get_file_list(partial_points_path, complete_points_path, dataset_categories, subset, n_renderings)


file_list = []
for file in file_list_pcn:
    pc_path = file['%s_path' % 'gt']
    rand_idx = random.randint(0, n_renderings - 1) if subset=='train' else 0

    if type(pc_path) == list:
        pc_path = pc_path[rand_idx]
    
    data = IO.get(pc_path).astype(np.float32)


    data_normalized, centroid, m = pc_norm(data)

    shapenet_bus = data
    #_, centroid, m = pc_norm(shapenet_bus)
    
    shapenet_bus_list.append((shapenet_bus, centroid, m))




#################################### inspecting Nuscenes data using bbox normalizer
import pickle
import copy
from pc_completion_normalize import NormalizeObjectPose, nusc2kitti_box_for_pc_completion_normalize, rotate

from pyquaternion import Quaternion

vehicle_names = {
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ambulance',
    'vehicle.emergency.police': 'police',
    'vehicle.trailer': 'trailer'
}



pc_root = "/home/shinghei/lidar_generation/Lidar_generation/foreground_object_pointclouds"
pc_folder_list = os.listdir(pc_root)
with open(os.path.join(pc_root, "sample_dict.pickle"), 'rb') as handle:
    sample_dict= pickle.load(handle)

for folder in pc_folder_list:
    #folder = "car"
    if folder!=cat:
        continue
    if folder not in vehicle_names.values():
        continue
    #print(f"FOLDER: {folder}")
    full_path = os.path.join(pc_root, folder)
    pc_file_list = os.listdir(full_path)
    for pc_file in pc_file_list:
        #print(f"pc file: {pc_file}")
        key = (folder, pc_file)
        assert(len(sample_dict[key][5])==1)
        kitti_cam_box, kitti_lidar_box = sample_dict[key][5][0]
        center3D = sample_dict[key][4][0]
        nusc_box = sample_dict[key][3][0]

        kitti_box = nusc2kitti_box_for_pc_completion_normalize(nusc_box)
        points = np.asarray(open3d.io.read_point_cloud(os.path.join(pc_root, folder, pc_file)).points)
        sample = {'bbox':kitti_box, 'partial_cloud':points}

        no_bbox_normalized_points, centroid, m = pc_norm(points)
       
        parameters = {'input_keys':{'ptcloud': 'partial_cloud', 'bbox':'bbox'}}
        normalizer = NormalizeObjectPose(parameters)
        data_normalized = normalizer(sample)
        nuscenes_bus = data_normalized['partial_cloud']
        # nuscenes_bus = rotate(nuscenes_bus, axis=(0, 1, 0), angle=np.pi/2)
        
        # nuscenes_bus = rotate(no_bbox_normalized_points, axis=(1, 0, 0), angle=np.pi/2)
        # nuscenes_bus = rotate(nuscenes_bus, axis=(0, 0, 1), angle=-np.pi)

        #nuscenes_bus = no_bbox_normalized_points
        
        #_, centroid, m = pc_norm(nuscenes_bus)
       
        nuscenes_bus_list.append((nuscenes_bus, centroid, m))




for i in range(min([len(shapenet_bus_list), len(nuscenes_bus_list)])):
    shapenet_bus = shapenet_bus_list[i][0]
    nuscenes_bus = nuscenes_bus_list[i][0]
    print("shapemet norm: ", shapenet_bus_list[i][2])
    print("nusc norm: ", nuscenes_bus_list[i][2])
    pcd1 = open3d.geometry.PointCloud()
    pcd1.points = open3d.utility.Vector3dVector(np.array(shapenet_bus))
    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(np.array(nuscenes_bus))
    open3d.visualization.draw_geometries([pcd1, pcd2.translate([1,0,0])]) 


