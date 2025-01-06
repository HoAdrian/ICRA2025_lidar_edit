import os
import numpy as np
import yaml
from pathlib import Path
from torch.utils import data

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box

from data_utils import *
from data_utils_nuscenes import *
import open3d

import sys
# sys.path.append("./")
# sys.path.append("./datasets")


################################### Helper methods for filtering out of bound points #####################
class Voxelizer:
    def __init__(self, grid_size=[512, 512, 32], max_bound=[50.635955604688654, 2*np.pi, np.deg2rad(120+40/31/2)], min_bound=[0, 0, np.deg2rad(80-40/31/2)]):
        '''
        The shape of voxels is  (#r, #theta, #z). the first and third dimensions are the radius in 3D and the elevation angle from the cartesian z axis when we are using spherical coordinates

        voxel_position: position of each voxel, Shape: (3, #r, #theta, #z)
        intervals: the step at each of the three dimensions
        grid_size: number of grids per dimension (3,)
        max_bound: max bound of each dimension (3,)
        min_bound: min bound of each dimension (3,)
        '''
        self.grid_size = np.array(grid_size)
        self.max_bound = np.array(max_bound).astype(np.float64)
        self.min_bound = np.array(min_bound).astype(np.float64)

        # get grid index
        crop_range = self.max_bound - self.min_bound
        cur_grid_size = self.grid_size
        intervals = np.where(cur_grid_size!=1, crop_range/(cur_grid_size-1).astype(np.float64), crop_range+1)

        if (intervals==0).any(): print("Zero interval!")

        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1 
        voxel_position = np.indices(self.grid_size)*intervals.reshape(dim_array) + self.min_bound.reshape(dim_array) # Shape: (3, #r, #theta, #z)

        self.intervals = intervals
        self.voxel_position = voxel_position

    def compute_voxel_position(self, grid_idx):
        grid_idx = np.array(grid_idx)
        return grid_idx*self.intervals + self.min_bound
    
    def filter_in_bound_points(self, points_polar):
        '''
        Return a binary mask over points. A point has a mask value 1 if it is within the grid's bound of this voxelizer else 0.
        '''
        r = points_polar[:,0]
        theta = points_polar[:,1]
        z = points_polar[:,2]

        r_within = (self.min_bound[0] <= r) & (r <= self.max_bound[0])
        theta_within = (self.min_bound[1] <= theta) & (theta <= self.max_bound[1])
        z_within = (self.min_bound[2] <= z) & (z <= self.max_bound[2])

        return (r_within) & (theta_within) & (z_within)
    
    def get_grid_ind(self, points_polar):
        '''
        get voxel index for each point
        '''
        grid_ind = (np.floor((np.clip(points_polar[:],self.min_bound,self.max_bound)-self.min_bound)/self.intervals)).astype(np.int64) # shape: (num_points, 3), assuming 3 is len(grid_size)
        # orig = len(points_polar)
        # now = len(np.unique(grid_ind, axis=0))
        # print(f"orig vs now: {orig} {now}")
        return grid_ind
    
    def idx2point(self, grid_idxs):
        xyz_pol = (grid_idxs[:, :].astype(np.float32) * self.intervals) + self.min_bound
        return xyz_pol
    
    def occupancy2idx(self, occupancy):
        r_id, theta_id, z_id = np.nonzero(occupancy)
        r_id = r_id.reshape(-1,1)
        theta_id = theta_id.reshape(-1,1)
        z_id = z_id.reshape(-1,1)
        non_zero_grid_indxs = np.concatenate((r_id, theta_id, z_id), axis=1) #(N,3)
        return non_zero_grid_indxs

    
    def voxelize(self, points_polar, return_point_info=False):  
        '''
        -points_polar: (N,3) array, points in polar coordinates

        Return:
        - voxel_centers (num_points, 3)
        - return_points (num_points, 6)
        - grid_ind (num_points,3)
        - voxels_occupancy: the voxels that contain any points are labeled 1, otherwise 0. Shape (#r,#theta,#z)
        '''

        assert(points_polar.shape[1]==len(self.grid_size))

        grid_ind = self.get_grid_ind(points_polar)

        # process_voxel occupancy
        voxels_occupancy = np.zeros(self.grid_size)
        voxels_occupancy[grid_ind[:,0], grid_ind[:,1], grid_ind[:,2]] = 1

        if return_point_info:
            # center data on each voxel for PTnet
            voxel_centers = (grid_ind.astype(np.float32) + 0.5)*self.intervals + self.min_bound
            centered_points = points_polar - voxel_centers
            return_points = np.concatenate((centered_points,points_polar),axis = 1)

            return grid_ind, return_points, voxel_centers, voxels_occupancy
        
        else:
            return None, None, None, voxels_occupancy


################################### Helper methods for getting Nuscenes scans #####################
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
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck'
}

vehicle_idxs = {
    'car': 0,
    'bus': 1,
    'truck':2
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

class NuscenesDetr(data.Dataset):
    def __init__(self, data_path, version = 'v1.0-trainval', split = 'train', vis=False, mode="spherical", multisweep=False):
        '''
        '''
        if mode!="spherical" and mode!="polar":
            raise Exception(f"the mode {mode} is invalid")
        
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

        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        self.max_num_obj = 64


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

               

            points_3D = points_xyz[:,:3]
            N = len(points_xyz)
           
            # ------------------------------- LABELS ------------------------------
            boxes = [box for box in boxes if np.sum(points_in_box(box, points_3D.T, wlh_factor = 1.0))!=0 and boxes.name in vehicle_names]
            angle_classes = np.zeros((self.max_num_obj,), dtype=np.float32)
            angle_residuals = np.zeros((self.max_num_obj,), dtype=np.float32)
            raw_angles = np.zeros((self.max_num_obj,), dtype=np.float32)
            raw_sizes = np.zeros((self.max_num_obj, 3), dtype=np.float32)
            label_mask = np.zeros((self.max_num_obj))
            label_mask[0 : len(boxes)] = 1
            max_bboxes = np.zeros((self.max_num_obj, 8))
            max_bboxes[0 : len(boxes), :] = bboxes

            target_bboxes_mask = label_mask
            target_bboxes = np.zeros((self.max_num_obj, 6)) # center and size
            box_corners_list = []
            box_category_list = []
            box_semantic_idx_list = []
            box_heading_angles = []
            filtered_bboxes = []



            for i, box in enumerate(boxes):
                bb_mask = points_in_box(box, points_3D.T, wlh_factor = 1.0)
                if np.sum(bb_mask)==0 or boxes[i].name not in vehicle_names:
                    continue
                filtered_bboxes.append(box)
                obj_points = points_3D[bb_mask]
                category = boxes[i].name
                simplified_category = vehicle_names[category]

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

                corners_3d = box.corners()
                # compute axis aligned box
                xmin = np.min(corners_3d[:, 0])
                ymin = np.min(corners_3d[:, 1])
                zmin = np.min(corners_3d[:, 2])
                xmax = np.max(corners_3d[:, 0])
                ymax = np.max(corners_3d[:, 1])
                zmax = np.max(corners_3d[:, 2])
                target_bbox = np.array(
                    [
                        (xmin + xmax) / 2,
                        (ymin + ymax) / 2,
                        (zmin + zmax) / 2,
                        xmax - xmin,
                        ymax - ymin,
                        zmax - zmin,
                    ]
                )

                # save data
                target_bboxes[i, :] = target_bbox
                box_corners_list.append(corners_3d)
                box_category_list.append(simplified_category)
                box_semantic_idx_list.append(vehicle_idxs[simplified_category])
                
                #### compute heading angle
                center2D = box.center[:2]
                corner1, corner2 = corners_3d[:,0][:2], corners_3d[:,1][:2]  # top front corners (left and right)
                corner7, corner6 = corners_3d[:,6][:2], corners_3d[:,5][:2] # bottom back corner (right), top back corner (right)
                center2D = (corner1 + corner6)/2
                center3D = np.array([center2D[0], center2D[1], (corners_3d[2,2]+corners_3d[2,1])/2])
                right_pointing_vector = (corner2 + corner6)/2.0 - center2D
                front_pointing_vector = (corner1 + corner2)/2.0 - center2D
                obj2cam_vector = -center2D
                ### compute allocentric angle and viewing angle gamma
                alpha = compute_allocentric_angle(obj2cam_pos=obj2cam_vector, obj_right_axis=right_pointing_vector, obj_front_axis=front_pointing_vector)
                gamma = compute_viewing_angle(-obj2cam_vector)

                box_heading_angles.append(alpha)

                # box statistics
                front_len = np.linalg.norm(corner1-corner2)
                side_len = np.linalg.norm(corner2-corner6)
                corner7_3D, corner6_3D = corners_3d[:,6][:3], corners_3d[:,5][:3]
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
                    corners_3d = box.corners() #(3,8)
                    corner_1 = corners_3d[:,0][:2]
                    corner_2 = corners_3d[:,1][:2]
                    corner_5 = corners_3d[:,4][:2]
                    corner_6 = corners_3d[:,5][:2]
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


        
        ret_dict = {}
        ret_dict["point_clouds"] = points_xyz.astype(np.float32)
        ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
    



    def change_split(self, s):
        assert s in ['train', 'val']
        self.split = s
