import json
import os
import numpy as np

import h5py
import numpy as np
import open3d
import os

############################# IO ##############################
class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd', '.ply']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
       
    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    @classmethod
    def _read_pcd(cls, file_path):
        pc = open3d.io.read_point_cloud(file_path)
        ptcloud = np.array(pc.points)
        return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]

############################# normalizing ###########################################
def pc_norm(pc, center=None, scale=None):
    """ pc: NxC, return NxC """
    if center is not None:
        centroid = center
    else:
        centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    if scale is not None:
        m = scale
    else:
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m


###################### get partial point cloud of shapenet
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
random.seed(2021)
def seprate_point_cloud(xyz, num_points, crop, fixed_points = None, padding_zeros = False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    _,n,c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None
        
    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop,list):
            num_crop = random.randint(crop[0],crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:       
            center = F.normalize(torch.randn(1,1,3),p=2,dim=-1).cuda()
        else:
            if isinstance(fixed_points,list):
                fixed_point = random.sample(fixed_points,1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1,1,3).cuda()

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p =2 ,dim = -1)  # 1 1 2048

        idx = torch.argsort(distance_matrix,dim=-1, descending=False)[0,0] # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] =  input_data[0,idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0) # 1 N 3

        crop_data =  points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop,list):
            INPUT.append(fps(input_data,2048))
            CROP.append(fps(crop_data,2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT,dim=0)# B N 3
    crop_data = torch.cat(CROP,dim=0)# B M 3

    return input_data.contiguous(), crop_data.contiguous()

from pointnet2_ops import pointnet2_utils
def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data


############################## inspecting shape net dataset #############################################
# centroid: [-0.00071701  0.06227426  0.00122877]
# max norm: 0.49752506613731384

shapenet_bus_list = []
nuscenes_bus_list = []

data_root = '/home/shinghei/lidar_generation/PoinTr/data/ShapeNet55-34/ShapeNet-55'
pc_path = '/home/shinghei/lidar_generation/PoinTr/data/ShapeNet55-34/shapenet_pc'
shapenet_dict = json.load(open('/home/shinghei/lidar_generation/PoinTr/data/shapenet_synset_dict.json', 'r'))
subset = "test"#"test" #"test"
data_list_file = os.path.join(data_root, f'{subset}.txt')

print(f'[DATASET] Open file {data_list_file}')
with open(data_list_file, 'r') as f:
    lines = f.readlines()

cat = "bus"
if subset=="train":
    cat = "car"
        
file_list = []
for line in lines:
    line = line.strip()
    taxonomy_id = line.split('-')[0]
    model_id = line.split('-')[1].split('.')[0]
    category = shapenet_dict[taxonomy_id]
    #print(f"category: {category}")
    if category != cat:
        continue

    
    sample = {
        'taxonomy_id': taxonomy_id,
        'model_id': model_id,
        'file_path': line
    }
    data = IO.get(os.path.join(pc_path, sample['file_path'])).astype(np.float32)


    # gt = torch.from_numpy(data).unsqueeze(0).cuda()
    # npoints =  8192
    # partial_pc, _ = seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
    # data = partial_pc.squeeze().cpu().numpy()

    data_normalized, centroid, m = pc_norm(data)

    shapenet_bus = data_normalized
    #_, centroid, m = pc_norm(shapenet_bus)

    # _, centroid, m = pc_norm(shapenet_bus)
    
    shapenet_bus_list.append((shapenet_bus, centroid, m))




#################################### inspecting Nuscenes data using bbox normalizer
# centroid: [11.79759405 -1.3358742  -1.02677062]
# max norm: 6.160364713868418
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
        data_normalized, center, scale = normalizer(sample)
        nuscenes_bus = data_normalized['partial_cloud']
        
        nuscenes_bus = rotate(nuscenes_bus, axis=(0, 1, 0), angle=np.pi/2)
        
        # nuscenes_bus = rotate(no_bbox_normalized_points, axis=(1, 0, 0), angle=np.pi/2)
        #nuscenes_bus = rotate(nuscenes_bus, axis=(0, 0, 1), angle=-np.pi)

        #nuscenes_bus = no_bbox_normalized_points
        
        #nuscenes_bus, centroid, m = pc_norm(nuscenes_bus)
        #_, centroid, m = pc_norm(nuscenes_bus, center=center, scale=scale)
       
        nuscenes_bus_list.append((nuscenes_bus, centroid, m))




for i in range(min([len(shapenet_bus_list), len(nuscenes_bus_list)])):
    shapenet_bus = shapenet_bus_list[i][0]
    nuscenes_bus = nuscenes_bus_list[i][0]
    print("shapemet norm: ", shapenet_bus_list[i][2])
    print("nusc norm: ", nuscenes_bus_list[i][2])
    shapenet_idx = np.argmax(np.sqrt(np.sum(shapenet_bus**2, axis=1)))
    nuscenes_idx = np.argmax(np.sqrt(np.sum(nuscenes_bus**2, axis=1)))
    max_shapenet = shapenet_bus[shapenet_idx][np.newaxis, :]
    max_nuscenes = nuscenes_bus[nuscenes_idx][np.newaxis, :]
    print("max norm point shapenet: ", max_shapenet)
    print("max norm point nuscenes: ", max_nuscenes)
    # ground_pcd_colors = np.tile(np.array([[0,1,0]]), (len(grd_points), 1))
    # ground_pcd.colors = open3d.utility.Vector3dVector(ground_pcd_colors)

    print("shapemet centroiud: ", shapenet_bus_list[i][1])
    print("nusc centroid: ", nuscenes_bus_list[i][1])
    pcd1 = open3d.geometry.PointCloud()
    pcd1.points = open3d.utility.Vector3dVector(np.array(shapenet_bus))
    pcd_colors = np.tile(np.array([[0,0,1]]), (len(shapenet_bus), 1))
    pcd_colors[shapenet_idx] = np.array([1,0,0])
    pcd1.colors = open3d.utility.Vector3dVector(pcd_colors)

    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(np.array(nuscenes_bus))
    pcd_colors = np.tile(np.array([[0,0,1]]), (len(nuscenes_bus), 1))
    pcd_colors[nuscenes_idx] = np.array([1,0,0])
    pcd2.colors = open3d.utility.Vector3dVector(pcd_colors)

    open3d.visualization.draw_geometries([pcd1, pcd2.translate([2,0,0])]) 


