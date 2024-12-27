import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *
import copy
import pickle

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

# @DATASETS.register_module()
# class Custom(data.Dataset):
#     def __init__(self, config):
#         self.partial_points_path = config.PARTIAL_POINTS_PATH
#         self.complete_points_path = config.COMPLETE_POINTS_PATH
#         self.category_file = config.CATEGORY_FILE_PATH
#         self.npoints = config.N_POINTS
#         self.subset = config.subset
#         self.cars = config.CARS

#         # Load the dataset indexing file
#         self.dataset_categories = []
#         with open(self.category_file) as f:
#             self.dataset_categories = json.loads(f.read())
#             if config.CARS:
#                 self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

#         self.n_renderings = 8 if self.subset == 'train' else 1
#         self.file_list = self._get_file_list(self.subset, self.n_renderings)
#         self.transforms = self._get_transforms(self.subset)

#     def _get_transforms(self, subset):
#         if subset == 'train':
#             return data_transforms.Compose([{
#                 'callback': 'RandomSamplePoints',
#                 'parameters': {
#                     'n_points': 2048
#                 },
#                 'objects': ['partial']
#             }, {
#                 'callback': 'RandomMirrorPoints',
#                 'objects': ['partial', 'gt']
#             },{
#                 'callback': 'ToTensor',
#                 'objects': ['partial', 'gt']
#             }])
#         else:
#             return data_transforms.Compose([{
#                 'callback': 'RandomSamplePoints',
#                 'parameters': {
#                     'n_points': 2048
#                 },
#                 'objects': ['partial']
#             }, {
#                 'callback': 'ToTensor',
#                 'objects': ['partial', 'gt']
#             }])

#     def _get_file_list(self, subset, n_renderings=1):
#         """Prepare file list for the dataset"""
#         file_list = []

#         # for dc in self.dataset_categories:
#         #     print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
#         #     samples = dc[subset]

#         #     for s in samples:
#         #         file_list.append({
#         #             'taxonomy_id':
#         #             dc['taxonomy_id'],
#         #             'model_id':
#         #             s,
#         #             'partial_path': [
#         #                 self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
#         #                 for i in range(n_renderings)
#         #             ],
#         #             'gt_path':
#         #             self.complete_points_path % (subset, dc['taxonomy_id'], s),
#         #         })
#         pc_root = "/home/shinghei/lidar_generation/our_ws/foreground_object_pointclouds"
#         pc_folder_list = os.listdir(pc_root)
#         #print(pc_folder_list)
#         for folder in pc_folder_list:
#             #folder = "car"
#             if folder not in vehicle_names.values():
#                 continue
#             #print(f"FOLDER: {folder}")
#             full_path = os.path.join(pc_root, folder)
#             pc_file_list = os.listdir(full_path)
#             for pc_file in pc_file_list:
#                 #print(f"pc file: {pc_file}")
#                 suffix = pc_file.split("_")[1]
#                 sample_num = suffix.split(".")[0]
#                 file_list.append({'taxonomy_id':(folder, sample_num), 'partial_path':[os.path.join(full_path, pc_file)], 'gt_path':os.path.join(full_path, pc_file), 'model_id':0})


#         print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
#         return file_list

#     def __getitem__(self, idx):
#         sample = self.file_list[idx]
#         data = {}
#         rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

#         for ri in ['partial', 'gt']:
#             file_path = sample['%s_path' % ri]
#             if type(file_path) == list:
#                 file_path = file_path[rand_idx]
#             data[ri] = IO.get(file_path).astype(np.float32)
            
#         old_data = copy.deepcopy(data)
#         #assert data['gt'].shape[0] == self.npoints

#         if self.transforms is not None:
#             data = self.transforms(data)

#         return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt']), old_data

#     def __len__(self):
#         return len(self.file_list)




from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
import open3d


@DATASETS.register_module()
class Custom(data.Dataset):
    def __init__(self, config):
        self.cloud_path = config.CLOUD_PATH
        self.bbox_path = config.BBOX_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        assert self.subset == 'test'

        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
        self.transforms = data_transforms.Compose([{
                'callback': 'NormalizeObjectPose',
                'parameters': {
                    'input_keys': {
                        'ptcloud': 'partial_cloud',
                        'bbox': 'bounding_box'
                    }
                },
                'objects': ['partial_cloud', 'bounding_box']
            }, {
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'bounding_box']
            }])
        self.file_list = self._get_file_list(self.subset)

        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
        imsize = (1600, 900)




    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        pc_root = "/home/shinghei/lidar_generation/Lidar_generation/foreground_object_pointclouds"
        pc_folder_list = os.listdir(pc_root)
        with open(os.path.join(pc_root, "sample_dict.pickle"), 'rb') as handle:
            sample_dict= pickle.load(handle)

        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse

        for folder in pc_folder_list:
            #folder = "car"
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
                
                #kitti_box = copy.deepcopy(kitti_lidar_box).corners().T
                kitti_box = copy.deepcopy(nusc_box).corners().T

                # convert nuscenes bounding box to kitti's bounding box
                kitti_box[:,:] = kitti_box[[2,3,7,6,1,0,4,5],:]
                #TYPO fixed on 3rd November, the fix is above: kitti_box[:,:] = kitti_box[[2,3,7,6,1,5,4,5],:]
                #kitti_box[:,:] = kitti_box[[7,6,2,3,4,5,1,0],:]

                suffix = pc_file.split("_")[1]
                sample_num = suffix.split(".")[0]
                file_list.append({'taxonomy_id':(folder, sample_num), 'partial_cloud':[os.path.join(full_path, pc_file)], 'bounding_box':kitti_box, 'model_id':0, 'center3D':center3D})


        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        for ri in ['partial_cloud']:
            file_path = sample[ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            data[ri] = np.asarray(open3d.io.read_point_cloud(file_path).points) #IO.get(file_path).astype(np.float32)

        # convert nusc pc to kitti pc
        vehicle_pc = data["partial_cloud"]
        # kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        # kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
        # kitti_to_nu_lidar_mat_inv = (np.array(kitti_to_nu_lidar_inv.rotation_matrix))
        # vehicle_pc = np.matmul(kitti_to_nu_lidar_mat_inv, data["partial_cloud"].T).T


        data['partial_cloud'] = vehicle_pc
        data['bounding_box'] = sample['bounding_box']

        old_data = copy.deepcopy(data)
        #assert data['gt'].shape[0] == self.npoints

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], data['partial_cloud'], old_data

    def __len__(self):
        return len(self.file_list)