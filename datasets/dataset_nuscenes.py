import os
import numpy as np
import yaml
from pathlib import Path
from torch.utils import data

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box

from data_utils import compute_allocentric_angle
from data_utils_nuscenes import *
import open3d

import sys
sys.path.append("./")
sys.path.append("./datasets")
from datasets.dataset import Voxelizer

class Nuscenes(data.Dataset):
    def __init__(self, data_path, version = 'v1.0-trainval', split = 'train', use_z=False, exhaustive=False, filter_valid_scene=True, get_stat=False, vis=False, voxelizer=None, is_test=False, mode="polar"):
        '''
        For loading and processing nuscenes data.

        - use_z: specify whether we should consider the z dimension of the object region when filtering out occluded points for creating the training point cloud scene
            if mode is "spherical, then it means whether to use the phi dimension
        -exhaustive: whether search through multiple possible rotations of object to object-free intervals
        -filter_valid_scene: whether to rotate an object to object-free intervals
        -get_stat: whether to compute some dataset statistics
        -vis: whether to save images of the scene we picked
        -voxelizer: use voxelizer to get only points within bounds
        -is_test: If True, choose object(s) no matter whether it is in an object-free interval. Must also set filter_valid scene to True
        -mode: either "polar" or "spherical"
        
        '''
        if mode!="spherical" and mode!="polar":
            raise Exception(f"the mode {mode} is invalid")
        
        if mode=="spherical":
            assert(use_z)

        self.version = version
        assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
        if version == 'v1.0-trainval':
            train_scenes = splits.train
            val_scenes = splits.val
        elif version == 'v1.0-test':
            train_scenes = splits.test
            val_scenes = []
        elif version == 'v1.0-mini':
            train_scenes = splits.mini_train
            val_scenes = splits.mini_val
        else:
            raise NotImplementedError
        self.split = split
        self.data_path = data_path
        
        self.nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
        
        available_scenes = get_available_scenes(self.data_path, self.nusc)
        available_scene_names = [s['name'] for s in available_scenes]
        train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
        val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
        train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
        val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

        self.train_token_list,self.val_token_list = get_path_infos(self.data_path, self.nusc, train_scenes,val_scenes)

        if self.split == 'train':
            num_examples = len(self.train_token_list)
        elif self.split == 'val':
            num_examples = len(self.val_token_list)
        elif self.split == 'test':
            num_examples = len(self.train_token_list)

        self.valid_scene_idxs = np.ones((num_examples,))
        self.exhaustive = exhaustive
        self.filter_valid_scene = filter_valid_scene
        self.use_z = use_z
        self.vis = vis
        self.voxelizer = voxelizer
        self.is_test = is_test
        self.mode = mode
        print("Nuscene dataset COORDINATE mode: ", self.mode)
        print("Nuscene dataset use_z: ", self.use_z)


        self.get_stat = get_stat #False
        self.minmax_r = np.array([1e10, 0])
        self.minmax_z = np.array([1e10, -1e10])
        self.obj_min_theta_diff = 1e10
        self.obj_min_r_diff = 1e10

        print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))
        
    def __len__(self):
        'Denotes the total number of samples'
        if self.split == 'train':
            return len(self.train_token_list)
        elif self.split == 'val':
            return len(self.val_token_list)
        elif self.split == 'test':
            return len(self.train_token_list)
    
    def __getitem__(self, index):
        '''
        return:
        -new_points_xyz_no_bckgrnd (input to neural network): point cloud in cartesian coordinates with points occluded by the rotated object removed (N,3)
        -points_xyz: the original points (K,4)
        -occlude_mask: binary mask that remove points in new_points_xyz_has_bckgrnd (i.e. points_xyz) occluded by the rotated object (False if occluded) (K,)
        -processed_obj_region: an object region for creating the point cloud without background and mask
        -points_in_box_mask: binary mask over points. A point has mask value 1 if it belongs to an object

        Note: occlude_mask is not used in PolarDataset class
        '''

        is_invalid_scene = True
        assert(self.valid_scene_idxs is not None)
        
        if self.filter_valid_scene: # create training scene by rotating object
            while is_invalid_scene:
                ###### get sample token from index
                if self.split == 'train':
                    sample_token = self.train_token_list[index]
                elif self.split == 'val':
                    sample_token = self.val_token_list[index]
                elif self.split == 'test':
                    sample_token = self.train_token_list[index]

                # sample_token =  "0c630e44d5b645a39cfdf15ae9a481aa"#"18616ac104ee4f92b6127dc67bb85f0b"
                # print(sample_token)
                
                # print(self.nusc.get('sample_data', sample_token)['filename'])
                # assert(1==0)
                #### get lidar points and bounding boxes
                lidar_path = os.path.join(self.data_path, self.nusc.get('sample_data', sample_token)['filename'])
                points_xyz = np.fromfile(lidar_path, dtype = np.float32).reshape((-1, 5))
                data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_token, selected_anntokens=None) #in sensor frame

                # put points within bound first before finding objects that can occlude points
                if self.voxelizer is not None:
                    points_within_bound_mask = self.voxelizer.filter_in_bound_points(cart2polar(points_xyz, mode=self.mode))
                    points_xyz = points_xyz[points_within_bound_mask]
                
                points_3D = points_xyz[:,:3]
                N = len(points_xyz)
                points_in_box_mask = np.zeros((N,)).astype(int)
                for i, box in enumerate(boxes):
                  mask = points_in_box(box, points_3D.T, wlh_factor = 1.0).astype(int)
                  points_in_box_mask = points_in_box_mask | mask

                #### get object regions and free intervals
                points_3D = points_xyz[:,:3] #(N,3)
                N,_ = points_xyz.shape
                points_polar = cart2polar(points_xyz, mode=self.mode)

                obj_regions = get_obj_regions(boxes, mode=self.mode)
                if len(obj_regions)==0:
                    self.valid_scene_idxs[index] = 0
                    print(f"---- index {index} cannot create training scene, no objects in the scene")
                    index = np.random.choice(np.nonzero(self.valid_scene_idxs)[0]) # sample a scene index that may be valid
                    print(f": the new index is {index}")
                    continue

                # get statistics about the dimensions of the scenes
                if self.get_stat:
                    self.minmax_r[0] = min([np.min(points_polar[:,0]), self.minmax_r[0]])
                    self.minmax_r[1] = max([np.max(points_polar[:,0]), self.minmax_r[1]])
                    self.minmax_z[0] = min([np.min(points_polar[:,2]), self.minmax_z[0]])
                    self.minmax_z[1] = max([np.max(points_polar[:,2]), self.minmax_z[1]])

                    obj_r_diffs = obj_regions[:,1,0] - obj_regions[:,0,0]
                    self.obj_min_r_diff = min([np.min(obj_r_diffs), self.obj_min_r_diff])

                    obj_min_theta, obj_max_theta = obj_regions[:,0,1], obj_regions[:,1,1] # min and max theta of object region
                    # whether the object region and free interval crosses the first and forth quadrants
                    obj_cross_bound = (obj_max_theta >=3*np.pi/2) & (obj_min_theta <= np.pi/2)
                    obj_theta_diff = np.where(obj_cross_bound,  obj_min_theta - (obj_max_theta - 2*np.pi), obj_max_theta-obj_min_theta)
                    self.obj_min_theta_diff = min([np.min(obj_theta_diff), self.obj_min_theta_diff])

               ############# create training scene
                intervals = find_exact_free_theta_intervals(obj_regions)

                
                obj_idxs = np.arange(len(boxes))
                np.random.shuffle(obj_idxs)
                
                if self.is_test:
                    ##### get the object the occlude many points
                    # good_obj_idxs = []
                    # num_occluded = []
                    # for i, obj_idx in enumerate(obj_idxs):
                    #     processed_obj_region = obj_regions[obj_idx]
                    #     processed_box = boxes[obj_idx]
                    #     occlude_mask,_ = get_obj_mask(processed_obj_region, points_polar, use_z=self.use_z)
                    #     new_points_xyz_no_bckgrnd = points_xyz[np.logical_not(occlude_mask)]
                    #     if np.sum(occlude_mask)!=0:
                    #         good_obj_idxs.append(obj_idx)
                    #         num_occluded.append(np.sum(occlude_mask))
                    # print("$$$$ good obj idxs: ", good_obj_idxs)
                    # arg_sort = np.argsort(np.array(num_occluded))
                    # good_obj_idxs = np.array(good_obj_idxs)[arg_sort]
                    # #### pick the object that occlude many points to create a mask
                    # obj_idx = good_obj_idxs[-3] #np.argmax(np.array(num_occluded))#np.random.choice(np.array(good_obj_idxs))
                    # print("$$$ chosen obj idx: ", obj_idx)
                    
                    # processed_obj_region = obj_regions[obj_idx]
                    # processed_box = boxes[obj_idx]
                    # processed_obj_region_list = [processed_obj_region]
                    # processed_box_list = [processed_box]
                    # occlude_mask,_ = get_obj_mask(processed_obj_region, points_polar, use_z=self.use_z)
                    # new_points_xyz_no_bckgrnd = points_xyz[np.logical_not(occlude_mask)]

                    ############# get all obj regions
                    processed_obj_region_list = obj_regions
                    processed_box_list = boxes
                    occlude_mask,_ = get_obj_mask(processed_obj_region_list[0], points_polar, use_z=self.use_z) #dummy
                    #points_xyz = points_xyz[np.logical_not(points_in_box_mask)] # remove points of objects
                    new_points_xyz_no_bckgrnd = points_xyz
                    print("++++++++ box 0 name: ... ", boxes[0].name)
                    #self.nusc.render_sample_data(self.nusc.get('sample_data',sample_token)["token"])
                    #####################################
                    ############## get ground points by segmentation ############
                    if self.version=='v1.0-mini':
                        name2idx =  self.nusc.lidarseg_name2idx_mapping
                        idx2name = self.nusc.lidarseg_idx2name_mapping
                        lidarseg_path = os.path.join(self.data_path, self.nusc.get('lidarseg', sample_token)['filename'])
                        point_labels = np.fromfile(lidarseg_path, dtype=np.uint8).reshape((-1,1))
                        if self.voxelizer is not None:
                            point_labels = point_labels[points_within_bound_mask]
                        assert(len(point_labels)==len(points_xyz))
                        point_labels = point_labels.reshape(-1)

                        ground_mask = (point_labels==name2idx["flat.driveable_surface"])#|(point_labels==name2idx["flat.sidewalk"])|(point_labels==name2idx["flat.other"])
                        ground_points = points_xyz[ground_mask]
                        ### for access from external code
                        self.ground_points = ground_points

                    break
                    ############################################
                
                 #### rotate an object region to free intervals to create training examples ####
                for i, obj_idx in enumerate(obj_idxs):
                    obj_region = obj_regions[obj_idx]
                    box = boxes[obj_idx]
                
                    if self.exhaustive:
                        training_scene = create_training_scene_exhaustive(points_polar, points_xyz, box, obj_region, intervals, use_z=self.use_z)
                    else:
                        training_scene = create_training_scene(points_polar, points_xyz, box, obj_region, intervals, use_z=self.use_z)
                    if training_scene != None:
                        processed_obj_region, processed_box, new_points_xyz_no_bckgrnd, occlude_mask = training_scene
                        processed_obj_region_list = [processed_obj_region]
                        processed_box_list = [processed_box]
                        if np.sum(np.logical_not(occlude_mask))!=0:
                            is_invalid_scene = False
                            break
                    if i==len(obj_idxs)-1:
                        #### no training scene can be created, select a new scene index 
                        self.valid_scene_idxs[index] = 0
                        print(f"++++ index {index} cannot create training scene")
                        index = np.random.choice(np.nonzero(self.valid_scene_idxs)[0]) # sample a scene index that may be valid
                        print(f" : the new index is {index}")
                        # print("visualizing invalid scene: ")
                        # max_radius = np.max(points_polar[:,0])/4
                        # plot_obj_regions(intervals, obj_regions, points_xyz, max_radius, boxes, xlim=[-80,80], ylim=[-80,80], title="invalid original")

                    
        else: 
            # don't edit the scene, for stage 1 (vqvae) training
            ###### get sample token from index
            if self.split == 'train':
                sample_token = self.train_token_list[index]
            elif self.split == 'val':
                sample_token = self.val_token_list[index]
            elif self.split == 'test':
                sample_token = self.train_token_list[index]


            #### get lidar points and bounding boxes
            lidar_path = os.path.join(self.data_path, self.nusc.get('sample_data', sample_token)['filename'])
            points_xyz = np.fromfile(lidar_path, dtype = np.float32).reshape((-1, 5))
            data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_token, selected_anntokens=None) #in sensor frame

            if self.voxelizer is not None:
                points_within_bound_mask = self.voxelizer.filter_in_bound_points(cart2polar(points_xyz, mode=self.mode))
                points_xyz = points_xyz[points_within_bound_mask]

            #### get object regions and free intervals
            points_3D = points_xyz[:,:3] #(N,3)
            N,_ = points_xyz.shape
            points_in_box_mask = np.zeros((N,)).astype(int)
            for i, box in enumerate(boxes):
                mask = points_in_box(box, points_3D.T, wlh_factor = 1.0).astype(int)
                points_in_box_mask = points_in_box_mask | mask

            points_polar = cart2polar(points_xyz, mode=self.mode)

            new_points_xyz_no_bckgrnd = points_3D #dummy, no use
            occlude_mask = np.ones((len(points_xyz),))==1 #dummy, no use
            processed_obj_region = np.ones((2,3))# dummy, no use

            processed_obj_region_list = [processed_obj_region]# dummy, no use
            processed_box_list = boxes



        #### for visualization ####
        if self.vis:
            max_radius = np.max(points_polar[:,0])/4
            # plot_obj_regions(intervals, [processed_obj_region], points_xyz, max_radius, [processed_box]+boxes, xlim=[-40,40], ylim=[-40,40], title="raw", path="./test_figures", name="raw", vis=False)
            if not self.is_test:
                plot_obj_regions(intervals, processed_obj_region_list, points_xyz, max_radius, processed_box_list+boxes, xlim=[-40,40], ylim=[-40,40], title="raw", path="./test_figures", name="raw", vis=False)
            else:
                plot_obj_regions(intervals, processed_obj_region_list, points_xyz, max_radius, processed_box_list, xlim=[-40,40], ylim=[-40,40], title="raw", path="./test_figures", name="raw", vis=False)

        #raw_points_xyz = np.fromfile(lidar_path, dtype = np.float32).reshape((-1, 5))
        return new_points_xyz_no_bckgrnd[:,:3], points_xyz, occlude_mask, processed_obj_region_list, points_in_box_mask, None
        

    def change_split(self, s):
        assert s in ['train', 'val']
        self.split = s

def get_available_scenes(data_path, nusc):
    # data_path: where the data root is
    available_scenes = []
    print('total scene num:', len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break

        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num:', len(available_scenes))
    return available_scenes

def get_path_infos(data_path, nusc,train_scenes,val_scenes):
    train_token_list = []
    val_token_list = []
    for sample in nusc.sample:
        scene_token = sample['scene_token']
        data_token = sample['data']['LIDAR_TOP']
        lidar_path = os.path.join(data_path, nusc.get('sample_data',data_token)['filename'])
        if not Path(lidar_path).exists():
            continue

        if scene_token in train_scenes:
            train_token_list.append(data_token)
        else:
            val_token_list.append(data_token)
    return train_token_list, val_token_list

def get_path_infos_sample(data_path, nusc,train_scenes,val_scenes):
    train_sample_list = []
    val_sample_list = []
    for sample in nusc.sample:
        scene_token = sample['scene_token']
        data_token = sample['data']['LIDAR_TOP']
        lidar_path = os.path.join(data_path, nusc.get('sample_data',data_token)['filename'])
        if not Path(lidar_path).exists():
            continue

        if scene_token in train_scenes:
            train_sample_list.append(sample)
            
        else:
            val_sample_list.append(sample)
    return train_sample_list, val_sample_list

def get_path_infos_sample_tokens(data_path, nusc,train_scenes, val_scenes):
    train_sample_list = []
    val_sample_list = []
    for sample in nusc.sample:
        scene_token = sample['scene_token']
        data_token = sample['data']['LIDAR_TOP']
        lidar_path = os.path.join(data_path, nusc.get('sample_data',data_token)['filename'])
        if not Path(lidar_path).exists():
            continue

        if scene_token in train_scenes:
            train_sample_list.append(sample["token"])
            
        else:
            val_sample_list.append(sample["token"])
    return train_sample_list, val_sample_list


'''
The following dataset is for obtaining foreground objects' point clouds
'''

map_name_from_general_to_segmentation_class = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
    'flat.driveable_surface': 'driveable_surface',
    'flat.other': 'other_flat',
    'flat.sidewalk': 'sidewalk',
    'flat.terrain': 'terrain',
    'static.manmade': 'manmade',
    'static.vegetation': 'vegetation',
    'noise': 'ignore',
    'static.other': 'ignore',
    'vehicle.ego': 'ignore'
}

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

def farthest_point_sample_batched(point, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, D] where B= num batches, N=num points, D=point dim (typically D=3)
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint, D]
    """
    B, N, D = point.shape
    xyz = point[:, :, :3]
    centroids = np.zeros((B, npoint))
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.uniform(low=0, high=N, size=(B,)).astype(np.int32) #np.random.randint(0, N)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[np.arange(0, B), farthest, :] # (B, D)
        centroid = np.expand_dims(centroid, axis=1) # (B, 1, D)
        dist = np.sum((xyz - centroid) ** 2, -1) # (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1) # (B,)
    point = point[np.arange(0, B).reshape(-1, 1), centroids.astype(np.int32), :]
    return point
    
import cv2 

class NuscenesForeground(data.Dataset):
    def __init__(self, data_path, version = 'v1.0-trainval', split = 'train', vis=False, mode="spherical", any_scene=False, ignore_collect=False, multisweep=False, get_raw=False, voxelizer=None):
        '''
        For getting foreground object points and their bounding boxes and other statistics
        - ignore_collect: if True, just return the bounding boxes and the lidar points without doing anything else
        - voxelizer: the voxelizer using spherical coordinates that voxelize the entire point cloud scene
        - any_scene: even go through scenes that have no objects
        '''
        if mode!="spherical" and mode!="polar":
            raise Exception(f"the mode {mode} is invalid")
        
        if mode=="spherical":
            self.use_z = True
        else:
            self.use_z = False
        
        assert(mode=="spherical")
        
        assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
        if version == 'v1.0-trainval':
            train_scenes = splits.train
            val_scenes = splits.val
        elif version == 'v1.0-test':
            train_scenes = splits.test
            val_scenes = []
        elif version == 'v1.0-mini':
            train_scenes = splits.mini_train
            val_scenes = splits.mini_val
        else:
            raise NotImplementedError
        self.split = split
        self.data_path = data_path
        
        self.nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
        
        available_scenes = get_available_scenes(self.data_path, self.nusc)
        available_scene_names = [s['name'] for s in available_scenes]
        train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
        val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
        train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
        val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

        self.train_token_list,self.val_token_list = get_path_infos(self.data_path, self.nusc, train_scenes,val_scenes)
        self.train_sample_list, self.val_sample_list = get_path_infos_sample(data_path, self.nusc,train_scenes,val_scenes)
        self.train_sample_table_token_list, self.val_sample_table_token_list = get_path_infos_sample_tokens(data_path, self.nusc, train_scenes, val_scenes)

        if self.split == 'train':
            num_examples = len(self.train_token_list)
        elif self.split == 'val':
            num_examples = len(self.val_token_list)
        elif self.split == 'test':
            num_examples = len(self.train_token_list)

        self.version = version
        self.valid_scene_idxs = np.ones((num_examples,))
        self.vis = vis
        self.mode = mode
        self.multisweep = multisweep
        self.box_length_stats = {"front":[],"side":[],"vertical":[], "z_coord":[]}
        self.kitti_box_converter = KittiBoxConverter()
        self.cart2D_voxelizer = Voxelizer(grid_size=[100,100,32],  max_bound=[50, 50, 3], min_bound=[-50, -50, -5])
        self.ignore_collect = ignore_collect
        self.any_scene = any_scene
        self.get_raw = get_raw
        self.voxelizer = voxelizer

        print("Nuscene dataset COORDINATE mode: ", self.mode)

        print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))
        
    def __len__(self):
        'Denotes the total number of samples'
        if self.split == 'train':
            return len(self.train_token_list)
        elif self.split == 'val':
            return len(self.val_token_list)
        elif self.split == 'test':
            return len(self.train_token_list)
    
    def __getitem__(self, index):
        '''
        return:
        - a list of point clouds each of which belongs to a vehicle
        '''

        is_invalid_scene = True
        assert(self.valid_scene_idxs is not None)
        
        while is_invalid_scene:
            ###### get sample token from index
            if self.split == 'train':
                sample_token = self.train_token_list[index]
                sample = self.train_sample_list[index]
                curr_sample_table_token = self.train_sample_table_token_list[index]
            elif self.split == 'val':
                sample_token = self.val_token_list[index]
                sample = self.val_sample_list[index]
                curr_sample_table_token = self.val_sample_table_token_list[index]
            elif self.split == 'test':
                sample_token = self.train_token_list[index]
                sample = self.train_sample_list[index]
                curr_sample_table_token = self.train_sample_table_token_list[index]

            
            
            #### get lidar points and bounding boxes
            lidar_path = os.path.join(self.data_path, self.nusc.get('sample_data', sample_token)['filename'])
            points_xyz = np.fromfile(lidar_path, dtype = np.float32).reshape((-1, 5))
            if self.voxelizer is not None:
                points_within_bound_mask = self.voxelizer.filter_in_bound_points(cart2polar(points_xyz, mode=self.mode))
                points_xyz = points_xyz[points_within_bound_mask]
            sample_annotation_tokens = sample['anns']
            data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_token, selected_anntokens=sample_annotation_tokens) #in sensor frame
            single_sweep_points_xyz = np.copy(points_xyz)

            # if index==52:
            #     self.nusc.render_sample_data(self.nusc.get('sample_data', sample['data']['CAM_FRONT'])['token'])

                      
            if len(boxes)==0 and not self.any_scene:
                self.valid_scene_idxs[index] = 0
                print(f"---- index {index} cannot create training scene, no objects in the scene")
                return None
                index = np.random.choice(np.nonzero(self.valid_scene_idxs)[0]) # sample a scene index that may be valid
                print(f": the new index is {index}")
                continue

            points_3D = points_xyz[:,:3]
            N = len(points_xyz)
            points_in_box_mask = np.zeros((N,)).astype(int)
            points_in_boxes = []
            nonempty_boxes = []
            for i, box in enumerate(boxes):
                mask = points_in_box(box, points_3D.T, wlh_factor = 1.0).astype(int)
                points_in_boxes.append(np.copy(points_3D[mask==1]))
                points_in_box_mask = points_in_box_mask | mask
            
            obj_regions = get_obj_regions(boxes, mode=self.mode, points_in_boxes=points_in_boxes)
            occlude_mask = np.ones((len(points_xyz),))==1 #dummy, no use
            
            if not self.ignore_collect:
                single_sweep_points_xyz = points_xyz
            
                if not self.multisweep:
                    #### get lidarseg labels (pointwise labels)
                    name2idx =  self.nusc.lidarseg_name2idx_mapping
                    idx2name = self.nusc.lidarseg_idx2name_mapping
                    lidarseg_path = os.path.join(self.data_path, self.nusc.get('lidarseg', sample_token)['filename'])
                    point_labels = np.fromfile(lidarseg_path, dtype=np.uint8).reshape((-1,1))
                    if self.voxelizer is not None:
                        point_labels = point_labels[points_within_bound_mask]
                    #annotated_data = np.vectorize(self.map_name_from_general_index_to_segmentation_index.__getitem__)(annotated_data)
                    assert(len(point_labels)==len(points_xyz))
                    point_labels = point_labels.reshape(-1)
                else:
                    #self.nusc.render_sample(self.nusc.scene[0]['first_sample_token'])
                    #self.nusc.render_sample(sample_token)
                    #sample = self.nusc.get('sample', sample_token)
                    #sample = self.nusc.get('sample',self.nusc.scene[0]['first_sample_token'])
                    pcl, _ = LidarPointCloud.from_file_multisweep(self.nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=10)
                    points_xyz = pcl.points.T

                    name2idx =  self.nusc.lidarseg_name2idx_mapping
                    idx2name = self.nusc.lidarseg_idx2name_mapping
                    lidarseg_path = os.path.join(self.data_path, self.nusc.get('lidarseg', sample_token)['filename'])
                    point_labels = np.fromfile(lidarseg_path, dtype=np.uint8).reshape((-1,1))
                    point_labels = point_labels.reshape(-1)

                ### get all point on the ground
                ground_mask = (point_labels==name2idx["flat.driveable_surface"])#|(point_labels==name2idx["flat.sidewalk"])|(point_labels==name2idx["flat.other"])
                ground_points = single_sweep_points_xyz[ground_mask]
                ### for access from external code
                self.ground_points = ground_points[:,:3]

                print(f"---initial ground points: ", len(ground_points))
                ground_points = np.copy(ground_points)
                if self.voxelizer is not None:
                    _, _, _, ground_occupancy = self.voxelizer.voxelize(cart2polar(ground_points[:,:3], mode=self.mode), return_point_info=False)
                    ground_points = self.voxelizer.voxels2points(torch.tensor(ground_occupancy).unsqueeze(0).permute(0,3,1,2), mode=self.mode)[0]
                    self.ground_points = ground_points
                ###### erosion
                # _, _, _, ground_occupancy = self.cart2D_voxelizer.voxelize(self.ground_points, return_point_info=False)
                # print(f"---initial ground points after voxelize: ", np.sum(ground_occupancy))
                # kernel = np.ones((2, 1), np.uint8) 
                # for k in range(ground_occupancy.shape[2]):
                #     ground_occupancy[:,:,k] = cv2.erode(ground_occupancy[:,:,k], kernel, iterations=1)
                # print(f"remaining ground points: ", np.sum(ground_occupancy))
                # ground_idxs = self.cart2D_voxelizer.occupancy2idx(ground_occupancy)
                # ground_points = self.cart2D_voxelizer.idx2point(ground_idxs)

                ###### avoid collision with ego vehicle
                # ground_points = ground_points[np.linalg.norm(ground_points, axis=1)>5]
                # down_sample_n = min([len(ground_points), 800])
                # ground_points = farthest_point_sample_batched(ground_points[np.newaxis,...], npoint=down_sample_n)[0]
                # print(f"+++ remaining ground points: ", len(ground_points))
                self.sparse_ground_points = ground_points

                #### other background points
                other_background_mask = (point_labels==name2idx["flat.sidewalk"])|(point_labels==name2idx["flat.other"])|(point_labels==name2idx["flat.terrain"])\
                    |(point_labels==name2idx["vehicle.ego"])|(point_labels==name2idx["static.vegetation"])|(point_labels==name2idx["flat.sidewalk"])|(point_labels==name2idx["static.manmade"])

                self.other_background_points = single_sweep_points_xyz[other_background_mask][:,:3]
                

            points_3D = points_xyz[:,:3]
            N = len(points_xyz)
            obj_point_cloud_list = []
            obj_name_list = []
            obj_allocentric_list = []
            obj_gamma_list = []
            obj_centers_list = []
            obj_boxes_list = []
            kitti_boxes_list = []
            obj_ann_token_list = []
            
            obj_ann_info_list = []
            count_visible  = 0
            
            
            sd_record = self.nusc.get('sample_data', sample_token)
            cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            sensor_record = self.nusc.get('sensor', cs_record['sensor_token'])
            pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])


            if not self.ignore_collect:
                for i, box in enumerate(boxes):
                    bb_mask = points_in_box(box, points_3D.T, wlh_factor = 1.0)
                    ### TODO: evaluate how this may improve the performance
                    if np.sum(bb_mask)==0:
                        continue
                    obj_points = points_3D[bb_mask]
                    category = boxes[i].name
                    ann_token = sample_annotation_tokens[i]
                    visibility_token = self.nusc.get('sample_annotation', ann_token)['visibility_token']
                    visibility_level = self.nusc.get('visibility', visibility_token)['level']
                    # #print(f"<o> A <o>   VISIBILITY LEVEL: {visibility_level}")
                    # if visibility_level != 'v80-100':
                    #     continue
                    count_visible+=1

                    if not self.multisweep:
                        # remove ground points
                        obj_point_labels = point_labels[bb_mask]
                        lidarseg_mask = obj_point_labels==name2idx[category]
                        obj_points = obj_points[lidarseg_mask]
                    else:
                        # remove ground points
                        bb_mask = points_in_box(box, single_sweep_points_xyz[:,:3].T, wlh_factor = 1.0)
                        single_sweep_points_in_box = single_sweep_points_xyz[:,:3][bb_mask]
                        obj_point_labels = point_labels[bb_mask]
                        background_lidarseg_mask = obj_point_labels!=name2idx[category]
                        background_points_in_box = single_sweep_points_in_box[background_lidarseg_mask]
                        mean_z = np.mean(background_points_in_box, axis=0)[2]
                        std_z = np.sqrt(np.var(background_points_in_box, axis=0))[2]
                        low_z = mean_z - std_z
                        high_z = mean_z + std_z
                        filter_floor_mask = (obj_points[:,2] > high_z)
                        obj_points = obj_points[filter_floor_mask]

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


                    # kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
                    # kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
                    # kitti_box = copy.deepcopy(box)
                    # kitti_box.rotate(kitti_to_nu_lidar_inv)
                    
                    box_cam_kitti, box_lidar_kitti = self.kitti_box_converter.nuscenes_gt_to_kitti(self.nusc, sample, nusc_lidar_box=box)
                    
                    box_not_empty = np.sum(points_in_box(box, points_3D.T, wlh_factor = 1.0))!=0
                    if (category in vehicle_names or self.get_raw) and (box_not_empty):
                        obj_point_cloud_list.append(obj_points)
                        if category in vehicle_names:
                            obj_name_list.append(vehicle_names[category])
                        else:
                            obj_name_list.append(category)
                        obj_allocentric_list.append(alpha)
                        obj_centers_list.append(center3D)
                        obj_boxes_list.append(box)
                        obj_gamma_list.append(gamma)
                        kitti_boxes_list.append((box_cam_kitti, box_lidar_kitti))
                        obj_ann_token_list.append(ann_token)
                        sample_annotation = self.nusc.get('sample_annotation', ann_token)
                        obj_ann_info_list.append({"category_name":sample_annotation['category_name'], 'attribute_tokens':sample_annotation['attribute_tokens'], 'instance_token':sample_annotation['instance_token'], 'num_lidar_pts':sample_annotation['num_lidar_pts'], 'num_radar)ots':sample_annotation['num_radar_pts']})
                        #obj_records_list.append({"cs_record":cs_record, "sensor_record":sensor_record, "pose_record":pose_record, "sample_token":sample_token, "version":self.version, "split":self.split})
                        

                        # box statistics
                        front_len = np.linalg.norm(corner1-corner2)
                        side_len = np.linalg.norm(corner2-corner6)
                        corner7_3D, corner6_3D = corners[:,6][:3], corners[:,5][:3]
                        vertical_len = np.linalg.norm(corner7_3D - corner6_3D)
                        self.box_length_stats["front"].append(front_len)
                        self.box_length_stats["side"].append(side_len)
                        self.box_length_stats["vertical"].append(vertical_len)
                        self.box_length_stats["z_coord"].append(center3D[2])

                    self.vis= False#(np.abs(alpha - np.pi/4)<0.2) and (len(obj_points)>=200)
                    if self.vis:
                        print(f"alpha in degree: {alpha/np.pi*180}")
                        #### visualize the box axes and its position
                        plt.figure(figsize=(8, 6))
                        plt.gca().set_aspect('equal')
                        corners = box.corners() #(3,8)
                        corner_1 = corners[:,0][:2]
                        corner_2 = corners[:,1][:2]
                        corner_5 = corners[:,4][:2]
                        corner_6 = corners[:,5][:2]
                        rect = patches.Polygon([corner_1, corner_2, corner_6,corner_5], linewidth=1, edgecolor='r', facecolor='none')
                        # Get the current axes and plot the polygon patch
                        
                        plt.gca().add_patch(rect)

                        # plot box axes
                        plt.scatter(obj_points[:,0], obj_points[:,1], s=1)
                        plt.quiver(center2D[0], center2D[1], 0+right_pointing_vector[0], 0+right_pointing_vector[1], color='r', scale_units='xy', scale=1)
                        plt.quiver(center2D[0], center2D[1], 0+front_pointing_vector[0], 0+front_pointing_vector[1], color='g', scale_units='xy',scale=1)
                        # plot obj2cam_vector scaled and centered at box center
                        plt.quiver(center2D[0], center2D[1], -center2D[0]/np.linalg.norm(center2D)*200, -center2D[1]/np.linalg.norm(center2D)*200)
                        
                        plt.show()
                        
            break
        
        if self.vis:
            plot_obj_regions([], [], points_xyz, 100, boxes, xlim=[-40,40], ylim=[-40,40], title="raw", path="./test_figures", name="raw", vis=False)


        lidar_sample_token = sample_token
        all_boxes = boxes
        sample_records = {"cs_record":cs_record, "sensor_record":sensor_record, "pose_record":pose_record, "sample_token":sample_token, "version":self.version, "split":self.split}
        obj_properties = (obj_point_cloud_list, obj_name_list, points_3D, obj_allocentric_list, obj_centers_list, obj_boxes_list, obj_gamma_list, kitti_boxes_list, lidar_sample_token, boxes, obj_ann_token_list, sample_annotation_tokens, sample_records, obj_ann_info_list, curr_sample_table_token)

        #print(f"<o> A <o>   VISIBILITY object counts: {count_visible}")
        #plot_obj_regions([], [], points_xyz, 40, boxes, xlim=[-40,40], ylim=[-40,40], title="raw", path=None, name=None, vis=True)
    
        return points_3D, points_xyz, occlude_mask, obj_regions, points_in_box_mask, obj_properties
    



    def change_split(self, s):
        assert s in ['train', 'val']
        self.split = s




from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.kitti import KittiDB

class KittiBoxConverter:
    '''
    We are not really using this class
    '''
    def __init__(self):
        print("kitti converter initialized")
        self.cam_name = 'CAM_FRONT'
        self.lidar_name = 'LIDAR_TOP'

    @staticmethod
    def box_nuscenes_to_kitti(box: Box, velo_to_cam_rot: Quaternion,
                              velo_to_cam_trans: np.ndarray,
                              r0_rect: Quaternion,
                              kitti_to_nu_lidar_inv: Quaternion = Quaternion(axis=(0, 0, 1), angle=np.pi / 2).inverse) \
            -> Box:
        """
        Transform from nuScenes lidar frame to KITTI reference frame.
        :param box: Instance in nuScenes lidar frame.
        :param velo_to_cam_rot: Quaternion to rotate from lidar to camera frame.
        :param velo_to_cam_trans: <np.float: 3>. Translate from lidar to camera frame.
        :param r0_rect: Quaternion to rectify camera frame.
        :param kitti_to_nu_lidar_inv: Quaternion to rotate nuScenes to KITTI LIDAR.
        :return: Box instance in KITTI reference frame.
        """
        # Copy box to avoid side-effects.
        box = box.copy()

        # Rotate to KITTI lidar.
        box.rotate(kitti_to_nu_lidar_inv)

        kitti_lidar_box = copy.deepcopy(box)

        # Transform to KITTI camera.
        box.rotate(velo_to_cam_rot)
        box.translate(velo_to_cam_trans)

        # Rotate to KITTI rectified camera.
        box.rotate(r0_rect)

        # KITTI defines the box center as the bottom center of the object.
        # We use the true center, so we need to adjust half height in y direction.
        box.translate(np.array([0, box.wlh[2] / 2, 0]))

        return box, kitti_lidar_box
    
    # @staticmethod
    # def to_kitti_pc(pc, velo_to_cam_rot: Quaternion,
    #                 velo_to_cam_trans: np.ndarray,
    #                 r0_rect: Quaternion,
    #                 kitti_to_nu_lidar_inv: Quaternion = Quaternion(axis=(0, 0, 1), angle=np.pi / 2).inverse):
        

    def nuscenes_gt_to_kitti(self, nusc, sample, nusc_lidar_box) -> None:
        """
        Converts nuScenes GT annotations to KITTI format.
        """
        self.nusc = nusc
        nusc_lidar_box = copy.deepcopy(nusc_lidar_box)

        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
        imsize = (1600, 900)

        # Get sample data.
        sample_annotation_tokens = sample['anns']
        cam_front_token = sample['data'][self.cam_name]
        lidar_token = sample['data'][self.lidar_name]

        # Retrieve sensor records.
        sd_record_cam = self.nusc.get('sample_data', cam_front_token)
        sd_record_lid = self.nusc.get('sample_data', lidar_token)
        cs_record_cam = self.nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
        cs_record_lid = self.nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])

        # Combine transformations and convert to KITTI format.
        # Note: cam uses same conventions in KITTI and nuScenes.
        lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']),
                                        inverse=False)
        ego_to_cam = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']),
                                        inverse=True)
        velo_to_cam = np.dot(ego_to_cam, lid_to_ego)

        # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
        velo_to_cam_kitti = np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix)

        # Currently not used.
        imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.
        r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.

        # Projection matrix.
        p_left_kitti = np.zeros((3, 4))
        p_left_kitti[:3, :3] = cs_record_cam['camera_intrinsic']  # Cameras are always rectified.

        # Create KITTI style transforms.
        velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
        velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

        # Check that the rotation has the same format as in KITTI.
        assert (velo_to_cam_rot.round(0) == np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])).all()
        assert (velo_to_cam_trans[1:3] < 0).all()

        # Retrieve the token from the lidar.
        # Note that this may be confusing as the filename of the camera will include the timestamp of the lidar,
        # not the camera.
        filename_cam_full = sd_record_cam['filename']
        filename_lid_full = sd_record_lid['filename']
        # token = '%06d' % token_idx # Alternative to use KITTI names.
        
        # Convert from nuScenes to KITTI box format.
        box_cam_kitti, kitti_lidar_box = KittiBoxConverter.box_nuscenes_to_kitti(
            nusc_lidar_box, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect)

        return box_cam_kitti, kitti_lidar_box
















def get_allocentric_box(box, radius, theta, z, mode, alpha):
    '''
    set the box to be at a fixed radius from the camera (origin) with a fixed z (if spherical, z is the elevation angle from cartesion z axis) and fixed theta at a fixed allocentric angle

    - box: a box object Box(center=[[0.0],[0.0],[0.0]], size=list(wlh), orientation=pyquaternion_from_angle(0))
    - radius, theta, z: the desired insertion position
    - mode: spherical (this method should also work for polar, but we do not use polar)
    - alpha: desired allocentric angle
    '''
    BEV_polar_pos = np.array([[radius, theta, z]]).astype(np.float64) #(1,3)
    BEV_cart_pos = polar2cart(BEV_polar_pos, mode=mode) #(1,3)
    print("....... BEV_cart_pos: ", BEV_cart_pos)
    # find box axes
    box.translate(BEV_cart_pos.T)
    corners = box.corners()
    center2D = box.center[:2]
    corner1, corner2 = corners[:,0][:2], corners[:,1][:2]  # top front corners (left and right)
    corner7, corner6 = corners[:,6][:2], corners[:,5][:2] # bottom back corner (right), top back corner (right)
    center2D = (corner1 + corner6)/2
    right_pointing_vector = (corner2 + corner6)/2.0 - center2D
    front_pointing_vector = (corner1 + corner2)/2.0 - center2D
    obj2cam_pos = -BEV_cart_pos[0,:2]
    curr_alpha = compute_allocentric_angle(obj2cam_pos, obj_right_axis=right_pointing_vector, obj_front_axis=front_pointing_vector)
    # rotate box to the desired alpha and then translate it to the desired pos
    box.translate(-BEV_cart_pos.T) # shift back to origin and then rotate
    alpha_diff = alpha - curr_alpha
    box.rotate(pyquaternion_from_angle(-alpha_diff)) # alpha_diff>0, rotate box clockwise, else counterclockwise
    box.translate(BEV_cart_pos.T)

    return box


class NuscenesEval(data.Dataset):
    def __init__(self, data_path, version = 'v1.0-trainval', split = 'train', use_z=False, exhaustive=False, vis=False, voxelizer=None, mode="polar", allocentric_angle=0):
        '''
        Use this dataset to create artificial bounding boxes with specific width, length, height and allocentric angle to create mask for evaluation.
        For allocentric angle between 0 and pi/2, the object occludes the most when it is 0, the least when it is pi/2.

        use_z: specify whether we should consider the z dimension of the object region when filtering out occluded points for creating the training point cloud scene
            if mode is "spherical, then it means whether to use the phi dimension
        exhaustive: whether search through multiple possible rotations of object to object-free intervals
        vis: whether to save images of the scene we picked
        voxelizer: use voxelizer to get only points within bounds
        mode: either "polar" or "spherical"
        allocentric_angle: allocentric angle of the artificial bounding boxes
        '''
        if mode!="spherical" and mode!="polar":
            raise Exception(f"the mode {mode} is invalid")
        
        if mode=="spherical":
            assert(use_z)

        assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
        if version == 'v1.0-trainval':
            train_scenes = splits.train
            val_scenes = splits.val
        elif version == 'v1.0-test':
            train_scenes = splits.test
            val_scenes = []
        elif version == 'v1.0-mini':
            train_scenes = splits.mini_train
            val_scenes = splits.mini_val
        else:
            raise NotImplementedError
        self.split = split
        self.data_path = data_path
        
        self.nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
        
        available_scenes = get_available_scenes(self.data_path, self.nusc)
        available_scene_names = [s['name'] for s in available_scenes]
        train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
        val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
        train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
        val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

        self.train_token_list,self.val_token_list = get_path_infos(self.data_path, self.nusc, train_scenes,val_scenes)

        if self.split == 'train':
            num_examples = len(self.train_token_list)
        elif self.split == 'val':
            num_examples = len(self.val_token_list)
        elif self.split == 'test':
            num_examples = len(self.train_token_list)

        
        self.valid_scene_idxs = np.ones((num_examples,))
        self.exhaustive = exhaustive
        self.use_z = use_z
        self.vis = vis
        self.voxelizer = voxelizer
        self.mode = mode
        self.alpha = allocentric_angle
        print("Nuscene dataset COORDINATE mode: ", self.mode)
        print("Nuscene dataset use_z: ", self.use_z)

        print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))
        
    def __len__(self):
        'Denotes the total number of samples'
        if self.split == 'train':
            return len(self.train_token_list)
        elif self.split == 'val':
            return len(self.val_token_list)
        elif self.split == 'test':
            return len(self.train_token_list)
    
    def __getitem__(self, index):
        '''
        return:
        -new_points_xyz_no_bckgrnd (input to neural network): point cloud in cartesian coordinates with points occluded by the rotated object removed (N,3)
        -points_xyz: the original points (K,4)
        -occlude_mask: binary mask that remove points in new_points_xyz_has_bckgrnd (i.e. points_xyz) occluded by the rotated object (False if occluded) (K,)
        -processed_obj_region: an object region for creating the point cloud without background and mask
        -points_in_box_mask: binary mask over points. A point has mask value 1 if it belongs to an object
        '''

        assert(self.valid_scene_idxs is not None)
        
        ###### get sample token from index
        if self.split == 'train':
            sample_token = self.train_token_list[index]
        elif self.split == 'val':
            sample_token = self.val_token_list[index]
        elif self.split == 'test':
            sample_token = self.train_token_list[index]


        #### get lidar points and bounding boxes
        lidar_path = os.path.join(self.data_path, self.nusc.get('sample_data', sample_token)['filename'])
        points_xyz = np.fromfile(lidar_path, dtype = np.float32).reshape((-1, 5))
        data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_token, selected_anntokens=None) #in sensor frame

        # put points within bound first before finding objects that can occlude points
        if self.voxelizer is not None:
            points_within_bound_mask = self.voxelizer.filter_in_bound_points(cart2polar(points_xyz, mode=self.mode))
            points_xyz = points_xyz[points_within_bound_mask]
        
        points_3D = points_xyz[:,:3]
        N = len(points_xyz)
        points_in_box_mask = np.zeros((N,)).astype(int)
        for i, box in enumerate(boxes):
            mask = points_in_box(box, points_3D.T, wlh_factor = 1.0).astype(int)
            points_in_box_mask = points_in_box_mask | mask

        #### get object regions and free intervals
        points_3D = points_xyz[:,:3] #(N,3)
        N,_ = points_xyz.shape
        points_polar = cart2polar(points_xyz, mode=self.mode)
        
        # only need vehicles, ignore pedestrians
        boxes = [box for box in boxes if box.name in vehicle_names.keys()]

        if len(boxes)==0:
            self.valid_scene_idxs[index] = 0
            print(f"---- index {index} cannot create training scene, no objects in the scene")
            print(f": continue {index}")

        obj_regions = get_obj_regions(boxes, mode=self.mode)
        ############# find intervals where there are no foreground objects
        if len(boxes)!=0:
            intervals = find_exact_free_theta_intervals(obj_regions)
            #print("??? instervals: ", np.rad2deg(intervals))
        else:
            intervals = np.array([[0,2*np.pi]])

        ### construct bounding boxes for evaluation: w=front side, l=left/right side
        ### make some number of boxes, each box is at a fixed distance(radius) from the lidar camera and is at different theta.
        ### rotate the box according to the specified allocentric angle
        wlh_mean = np.array([2.010759375860466, 5.078891212594277, 1.9137354633555423])
        wlh_std = np.array([0.45754378838278364, 2.0220000960643008, 0.6501887235757507])
        wlh = wlh_mean
        z_mean = -0.7033732284201967
        z_std = 1.7092362191257453
        z = z_mean

         # every box is at a fixed radius from the camera
        radius = 10

        scale = 1 #1/2 #1/2 #4/5
        wlh[:2]*=scale
        synthetic_boxes = []

        while True:
            base_box = Box(center=[[0.0],[0.0],[0.0]], size=list(wlh), orientation=pyquaternion_from_angle(0))
            # print("z in spherical: ", cart2polar(np.array([[z,z,z]]), mode=self.mode)[0][2])
            allocentric_box = get_allocentric_box(copy.deepcopy(base_box), radius, theta=0, z=cart2polar(np.array([[radius,0,z]]), mode=self.mode)[0][2], mode=self.mode, alpha=self.alpha)
            obj_region = get_obj_regions([allocentric_box], mode=self.mode)[0]
            # print("++++ allocentric obj region", obj_region)
            # print(f"min max theta obj: {np.rad2deg(obj_region[0,1])},{np.rad2deg(obj_region[1,1])}")
            obj_dist = obj_region_theta_dist(obj_region)

            # get the angle in the middle of each free interval
            intervals_temp = np.copy(intervals)
            cross_bound_mask = intervals[:,0]>intervals[:,1] # cross first and forth quadrants
            print(cross_bound_mask)
            if np.sum(cross_bound_mask)!=0:
                intervals_temp[cross_bound_mask,0] -= 2*np.pi
            #print("??? temp instervals: ", np.rad2deg(intervals_temp))
            rand_weight = np.random.uniform(low=0, high=1)
            thetas = np.array([rand_weight*(intervals_temp[i,0]+obj_dist/2)+(1-rand_weight)*(intervals_temp[i,1]-obj_dist/2) for i in range(len(intervals_temp))])
            thetas[thetas<0]+=2*np.pi

            #print("??? thetas: ", np.rad2deg(thetas))

            #### evaluate which free intervals can fit in the box ############
            
            fit_interval_idxs = obj_region_is_fit(obj_region, intervals)

            ### for debugging
            allocentric_box_2 = get_allocentric_box(copy.deepcopy(base_box), radius, theta=np.pi/2, z=cart2polar(np.array([[radius,np.pi/2,z]]), mode=self.mode)[0][2], mode=self.mode, alpha=self.alpha)
            obj_region_2 = get_obj_regions([allocentric_box_2], mode=self.mode)[0]

            if len(fit_interval_idxs)==0:
                print("################## !!!!!!!! object too big to generate synthetic mask again")
                return None
                # break
                # wlh[:2]*=(scale)
                # print("################## !!!!!!!! try smaller (1/2) object to generate synthetic mask again")
                # continue

            ### generate synthetic boxes at different theta
            
            np.random.shuffle(fit_interval_idxs)
            for i in (fit_interval_idxs):
                theta = thetas[i]
                box = copy.deepcopy(allocentric_box)
                box.rotate(pyquaternion_from_angle(theta))
                synthetic_boxes.append(box)
                break

            break

        synthetic_obj_regions = get_obj_regions(synthetic_boxes, mode=self.mode)
            
        processed_obj_region_list = synthetic_obj_regions
        processed_box_list = synthetic_boxes
        
        new_points_xyz_no_bckgrnd = points_xyz # dummy
        occlude_mask = np.ones((len(points_xyz),))==1 #dummy, no use

        points_in_box_mask = np.zeros((N,)).astype(int)
        for i, box in enumerate(synthetic_boxes):
            mask = points_in_box(box, points_3D.T, wlh_factor = 1.0).astype(int)
            points_in_box_mask = points_in_box_mask | mask


        #### for visualization ####
        if self.vis:
            max_radius = np.max(points_polar[:,0])/4
            # plot_obj_regions(intervals, [processed_obj_region], points_xyz, max_radius, [processed_box]+boxes, xlim=[-40,40], ylim=[-40,40], title="raw", path="./test_figures", name="raw", vis=False)
            plot_obj_regions(intervals, processed_obj_region_list, points_xyz, max_radius, processed_box_list+boxes, xlim=[-40,40], ylim=[-40,40], title="raw", path="./test_figures", name="raw", vis=False)
            plot_obj_regions(intervals, [obj_region, obj_region_2], points_xyz, max_radius, [allocentric_box, allocentric_box_2], xlim=[-20,20], ylim=[-20,20], title="raw", path="./test_figures", name="allocentric_base", vis=False)
            plot_obj_regions(intervals, [ get_obj_regions([base_box], mode=self.mode)[0]], points_xyz, max_radius, [base_box], xlim=[-40,40], ylim=[-40,40], title="raw", path="./test_figures", name="base_box", vis=False)

        obj_properties = (None, None, None, None, None, None, None, None, sample_token)
        #raw_points_xyz = np.fromfile(lidar_path, dtype = np.float32).reshape((-1, 5))
        return new_points_xyz_no_bckgrnd[:,:3], points_xyz, occlude_mask, processed_obj_region_list, points_in_box_mask, obj_properties
        

    def change_split(self, s):
        assert s in ['train', 'val']
        self.split = s






