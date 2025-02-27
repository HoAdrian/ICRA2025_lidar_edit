#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.utils import data
from data_utils import *
from data_utils_nuscenes import get_obj_mask, rotate_obj_region, flip_obj_region, plot_obj_regions, pyquaternion_from_angle, get_obj_mask_occupied
from nuscenes.utils.geometry_utils import points_in_box
import timeit

def get_BEV_label(voxel_label):
     '''
     voxel_label: (#r, #theta, #z)

     convert voxel label to BEV by summing over the last dimension and each resultant voxel in BEV is labeled 1 with it is >=1, else 0

     return:
     BEV_label: (#r, #theta)
     '''
     BEV_label = np.sum(voxel_label, axis=-1) #(#r, #theta)
     return (BEV_label>=1).astype(np.int64)

class Voxelizer:
    def __init__(self, grid_size, max_bound, min_bound):
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

        self.mask_schedule = self.gamma_func("cosine")

    def compute_voxel_position(self, grid_idx):
        grid_idx = np.array(grid_idx)
        return grid_idx*self.intervals + self.min_bound

    def gamma_func(self, mode="cosine"):
        '''
        schedule the mask ratio, r must be in [0,1)
        '''
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r**2
        elif mode == "cubic":
            return lambda r: 1 - r**3
        else:
            raise NotImplementedError
        
    def create_mask_by_occlusion(self, obj_region, use_z=False):
        '''
        -obj_region: (2,3) array

        Return:
        - voxels_labels: the voxels that are occluded by the obj_region are labeled 1, otherwise 0. Shape (#r,#theta,#z)
        - voxels_occupancy: the voxels that contain any points are labeled 1, otherwise 0. Shape (#r,#theta,#z)
        '''
        flatten_voxel_pos = np.transpose(self.voxel_position, (1,2,3,0)).reshape(-1,3)
        vox_size_r, vox_size_theta, vox_size_z = (self.max_bound - self.min_bound)/self.grid_size
        safety_margin = True
        if safety_margin:
            if use_z:
                obj_region[0,2]-=vox_size_z #spherical coordinates, adding safety margin
                obj_region[0,0]-=vox_size_r
            else:
                # using polar coordinates
                obj_region[1,2]+=vox_size_z
                obj_region[0,0]-=vox_size_r
        #use_z = True
        voxels_labels, _ = get_obj_mask(obj_region, flatten_voxel_pos, use_z=use_z)
        voxels_labels = voxels_labels.astype(np.int64)
        voxels_labels = voxels_labels.reshape(self.grid_size)
        return voxels_labels
    
    def create_mask_by_object_occupancy(self, obj_region, use_z=False):
        '''
        -obj_region: (2,3) array
        create the mask over the voxel space indicating voxels occupied (not occluded by the object)

        Return:
        - voxels_labels: the voxels that are occluded by the obj_region are labeled 1, otherwise 0. Shape (#r,#theta,#z)
        - voxels_occupancy: the voxels that contain any points are labeled 1, otherwise 0. Shape (#r,#theta,#z)
        '''
        flatten_voxel_pos = np.transpose(self.voxel_position, (1,2,3,0)).reshape(-1,3)
        vox_size_r, vox_size_theta, vox_size_z = (self.max_bound - self.min_bound)/self.grid_size
        safety_margin = True
        if safety_margin:
            if use_z:
                obj_region[0,2]-=vox_size_z #spherical coordinates, adding safety margin
                obj_region[0,0]-=vox_size_r
            else:
                # using polar coordinates
                obj_region[1,2]+=vox_size_z
                obj_region[0,0]-=vox_size_r
        #use_z = True
        voxels_labels, _ = get_obj_mask_occupied(obj_region, flatten_voxel_pos, use_z=use_z)
        voxels_labels = voxels_labels.astype(np.int64)
        voxels_labels = voxels_labels.reshape(self.grid_size)
        return voxels_labels
    
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
        
    def voxels2points(self, voxels, mode="spherical"):
        '''
        - voxelizer: Voxelizer object from dataset.py
        - voxels: binary, shape (B, in_chans, H, W), assume in_chans corresponds to z, H and W corresponds to r and theta. 

        return: 
        - list of numpy array of point cloud in cartesian coordinate (each may have different number of points)
        '''
        B, _, _, _ = voxels.shape
        point_clouds = []
        for b in range(B):
            voxels_b = voxels[b]
            voxels_b = voxels_b.permute(1,2,0) # (H, W, in_chans)
            non_zero_indices = torch.nonzero(voxels_b).float() #(num_non_zero_voxel, 3)
            ## convert non zero voxels to points
            intervals = torch.tensor(self.intervals).to(voxels.device).unsqueeze(0) #(1,3)
            min_bound = torch.tensor(self.min_bound).to(voxels.device).unsqueeze(0) #(1,3)
            xyz_pol = ((non_zero_indices[:, :]+0.5) * intervals) + min_bound # use voxel center coordinate
            xyz_pol = xyz_pol.cpu().detach().numpy()
            xyz = polar2cart(xyz_pol, mode=mode)

            point_clouds.append(xyz)

        return point_clouds #a list of (num_non_zero_voxel_of_the_batch, 3)
    
    def vis_BEV_binary_voxel(self, voxel, points_xyz=None, intensity=None, vis=True, path=None, name=None, vox_size=10, xlim=[-80,80], ylim=[-80,80], only_points=False, mode="polar"):
        '''
        voxel: binary voxels, (#r, #theta, #z)
        points_xyz: (N, 3)
        intensity: (N,), intensity value for each lidar point
        '''
        BEV_voxel = get_BEV_label(voxel.cpu().detach().numpy()) # (#r, #theta)
        BEV_voxel = BEV_voxel.reshape(-1)
        voxel_position = torch.from_numpy(self.voxel_position) #Shape: (3, #r, #theta, #z)
        voxel_position_flat = polar2cart(voxel_position.permute(1,2,3,0)[:,:,0,:].reshape(-1,3).numpy(), mode=mode) # r-theta position of voxels
        _, uni_idx = np.unique(voxel_position_flat[:,:2], axis=0, return_index=True)
        BEV_voxel = BEV_voxel[uni_idx]
        voxel_position_BEV = voxel_position_flat[uni_idx]
        voxel_position_BEV = voxel_position_BEV[BEV_voxel==1]
        BEV_voxel = BEV_voxel[BEV_voxel==1]
        if not only_points:
            # points_xyz and intensity are supposed to be none
            plot_points_and_voxels(points_xyz, intensity, voxel_position_BEV, BEV_voxel, xlim=xlim, ylim=ylim, vis=vis, title=None, path=path, name=name, vox_size=vox_size)
        else:
            plot_points_and_voxels(points_xyz, intensity, voxel_xyz=None, labels=None, xlim=xlim, ylim=ylim, vis=vis, title=None, path=path, name=name, vox_size=vox_size)
    
    def get_nearest_ground_BEV_pos(self, ground_xyz, vehicle_xyz_polar, mode):
        '''
        vehicle_xyz_polar: (1,3)
        ground_xyz: (N,3)
        
        return the polar position of nearest ground points in BEV, which is an ndarray of shape (1,3)
        '''
        vehicle_xyz = polar2cart(vehicle_xyz_polar, mode=mode) #polar2cart(vehicle_pts, mode="polar")
        grid_dists = np.linalg.norm(vehicle_xyz[:,:2] - ground_xyz[:,:2], axis=1)
        BEV_nearest_idx = np.argmin(grid_dists, axis=0) 

        return cart2polar(ground_xyz[BEV_nearest_idx, :][np.newaxis,:], mode=mode)
    
    def voxelize_and_occlude(self, scene_occupancy, vehicle_pc_polar, insert_only=False):
        '''
        voxelize the vehicle points, add the vehicle occupancy, and mark all occluded voxels as 0

        - scene_occupancy: np.ndarray (#r, #theta, #z)
        - vehicle_pc_polar: np.ndarray (N,3) in the same coordinate system as the voxelizer
        return the resultant occupancy grid of shape (#r, $theta, #z). The voxels occluded by the vehicle and
        the vehicle's voxels occluded by other voxels should be marked 0. 
        '''
        new_occupancy = np.copy(scene_occupancy)
        car_grid_idxs = self.get_grid_ind(vehicle_pc_polar) #(N,3)
        new_occupancy[car_grid_idxs[:,0], car_grid_idxs[:,1], car_grid_idxs[:,2]] = 1

        if insert_only:
            print("WARNING:  insert only not occlude")
            return new_occupancy

        ### should only be objects here
        # r_idx, theta_idx, z_idx = np.nonzero(new_occupancy) 
        # nonempty_grid_idxs = np.concatenate((r_idx.reshape(-1,1), theta_idx.reshape(-1,1), z_idx.reshape(-1,1)), axis=1) #(M,3)
        nonempty_grid_idxs = car_grid_idxs
        
        unique_theta_z, unique_row_idxs = np.unique(nonempty_grid_idxs[:,1:], axis=0, return_index=True) #(K,2), (K,)
        
        # apply occlusion by only keeping the lowest r idx for each theta, z
        #start_time = timeit.default_timer()
        for k, theta_z in enumerate(unique_theta_z):
            
            array1d = new_occupancy[:, theta_z[0], theta_z[1]]
            min_r = np.min(np.nonzero(array1d==1)[0])
            if min_r<self.grid_size[0]:
                new_occupancy[min_r+1:, theta_z[0], theta_z[1]] = 0
            
        # mid_time = timeit.default_timer()
        # print(f"!!!!! occlude: {mid_time - start_time} seconds")
        
        return new_occupancy
    
    def verify_occlusion(self, new_occupancy):
        r_idx, theta_idx, z_idx = np.nonzero(new_occupancy) 
        nonempty_grid_idxs = np.concatenate((r_idx.reshape(-1,1), theta_idx.reshape(-1,1), z_idx.reshape(-1,1)), axis=1) #(M,3)
        unique_theta_z, unique_row_idxs = np.unique(nonempty_grid_idxs[:,1:], axis=0, return_index=True) #(K,2), (K,)
        
        start_time = timeit.default_timer()
        for k, theta_z in enumerate(unique_theta_z):
            array1d = new_occupancy[:, theta_z[0], theta_z[1]]
            
            assert(np.sum(array1d==1)==1)
            if (np.sum(array1d==1)>1):
                print("... num occ in column: ", np.sum(array1d==1))
                print(theta_z)
                print(np.nonzero(array1d)[0])
        mid_time = timeit.default_timer()
        print(f"!!!!! verified occlude: {mid_time - start_time} seconds")
        
        return new_occupancy

    
    def voxelize_and_occlude_2(self, scene_occupancy, vehicle_pc_polar, add_vehicle=False, use_margin=False):
        '''
        Assuming the vehicles in vehicle_pc_polar are in the scene already, 
        mark all voxels in scene_occupancy that are occluded by vehicle_pc_polar and that belongs to the vehicle as 0, and also return the occlusion mask

        - scene_occupancy: np.ndarray (#r, #theta, #z)
        - add_vehicle: whether to add vehicle pc to the scene
        - vehicle_pc_polar: np.ndarray (N,3) in the same coordinate system as the voxelizer
        return the resultant occupancy grid of shape (#r, $theta, #z). and the occlusion mask
        '''
        new_occupancy = np.copy(scene_occupancy)
        car_grid_idxs = self.get_grid_ind(vehicle_pc_polar) #(N,3)
        voxels_labels = np.zeros(self.grid_size)
        
        #use_margin=use_margin
        # we add some safety margin when applying occlusion
        for i in range(len(car_grid_idxs)):
            car_grid_idx = car_grid_idxs[i]
            r, theta, z = car_grid_idx

            
            margin = 2 #theta
            theta_plus = theta+margin
            theta_minus = theta-margin
            if theta_minus<0:
                theta_minus = self.grid_size[1] - (margin-theta)
            if theta_plus>=self.grid_size[1]:
                theta_plus = 0 + margin-(self.grid_size[1]-1 - theta)

            margin = 2 #z
            z_plus = z+margin
            z_minus = z-margin
            if z_minus<0:
                z_minus = 0
            if z_plus>=self.grid_size[2]:
                z_plus = self.grid_size[2] - 1

            if use_margin:
                left_theta = self.idx2point(np.array([[0, theta_minus, 0]]))[0,1]
                right_theta = self.idx2point(np.array([[0, theta_plus, 0]]))[0,1]
                cross_bounds = right_theta<np.pi/2 and left_theta>3*np.pi/2# cross 1st and 4th quadrants

                if not cross_bounds:
                    new_occupancy[r+1:, (theta_minus):theta_plus+1, z_minus:z_plus] = 0

                    margin = 5 #r
                    voxels_labels[r-margin:, theta_minus:theta_plus+1, z_minus:z_plus] = 1
                else:
                    new_occupancy[r+1:, (theta_minus):self.grid_size[1], z_minus:z_plus] = 0
                    new_occupancy[r+1:, (0):theta_plus+1, z_minus:z_plus] = 0

                    margin = 5 #r
                    voxels_labels[r-margin:, (theta_minus):self.grid_size[1], z_minus:z_plus] = 1
                    voxels_labels[r-margin:, (0):theta_plus+1, z_minus:z_plus] = 1
            else:
                new_occupancy[r+1:, (theta), (z)] = 0

                voxels_labels[r:, theta, z] = 1
                # voxels_labels[r:, theta_plus, z] = 1
                # voxels_labels[r:, theta_minus, z] = 1
                # voxels_labels[r:, theta, z_plus] = 1
                # voxels_labels[r:, theta, z_minus] = 1

                # margin = 5 #r
                # voxels_labels[r-margin:, theta, z] = 1
                # voxels_labels[r-margin:, theta_plus, z] = 1
                # voxels_labels[r-margin:, theta_minus, z] = 1
                # voxels_labels[r-margin:, theta, z_plus] = 1
                # voxels_labels[r-margin:, theta, z_minus] = 1

        if add_vehicle:
            new_occupancy[car_grid_idxs[:,0], car_grid_idxs[:,1], car_grid_idxs[:,2]] = 1

            
        
        return new_occupancy, voxels_labels
    
    def copy_and_paste_neighborhood(self, scene_occupancy, voxels_mask):
        '''
        copy the adjacent voxels (clockwise and anticlockwise) of the masked voxels along the theta dimension, and paste them to the masked voxels.

        scene_occupancy: torch.Tensor (#r, #theta, #z)
        voxels_mask: torch.Tensor (#r, #theta, #z), assuming the voxels_mask is the occlusion mask for only a single foreground object
        '''
        new_occupancy = torch.clone(scene_occupancy)
        masked_grid_idxs = torch.nonzero(voxels_mask)
        # remove masked grids first
        new_occupancy[masked_grid_idxs[:,0], masked_grid_idxs[:,1], masked_grid_idxs[:,2]] = 0

        min_theta_idx = torch.min(masked_grid_idxs[:, 1])
        max_theta_idx = torch.max(masked_grid_idxs[:, 1])

        min_z_idx = torch.min(masked_grid_idxs[:, 2])
        max_z_idx = torch.max(masked_grid_idxs[:, 2])

        min_r_idx = torch.min(masked_grid_idxs[:, 0])
        #max_r_idx = torch.max(masked_grid_idxs[:, 0])

        min_theta = self.idx2point(np.array([[0, min_theta_idx, 0]]))[0,1]
        max_theta = self.idx2point(np.array([[0, max_theta_idx, 0]]))[0,1]
        cross_bounds = min_theta<np.pi/2 and max_theta>3*np.pi/2# cross 1st and 4th quadrants

        # find nearby voxels 
        margin =1
        if not cross_bounds:
            theta_plus = max_theta_idx+margin
            theta_minus = min_theta_idx-margin
        else:
            theta_plus = min_theta_idx+margin
            theta_minus = max_theta_idx-margin

        if theta_minus<0:
            theta_minus = self.grid_size[1] - (margin-min_theta_idx)
        if theta_plus>=self.grid_size[1]:
            theta_plus = 0 + margin-(self.grid_size[1] - max_theta_idx)

        # nearby voxels
        left_neighborhood = new_occupancy[min_r_idx:, theta_minus:theta_minus+1, min_z_idx:max_z_idx+1]
        right_neighborhood = new_occupancy[min_r_idx:, theta_plus:theta_plus+1, min_z_idx:max_z_idx+1]

        # and then paste to the masked voxels
        if not cross_bounds:
            mid_idx = (min_theta_idx + max_theta_idx)//2
            new_occupancy[min_r_idx:, min_theta_idx:mid_idx+1,  min_z_idx:max_z_idx+1] = left_neighborhood
            new_occupancy[min_r_idx:, mid_idx:max_theta_idx+1,  min_z_idx:max_z_idx+1] = right_neighborhood
        else:
            mid_idx = 0
            new_occupancy[min_r_idx:, max_theta_idx:,  min_z_idx:max_z_idx+1] = left_neighborhood
            new_occupancy[min_r_idx:, 0:min_theta_idx+1,  min_z_idx:max_z_idx+1] = right_neighborhood


        return new_occupancy
    
    def get_intensity_grid(self, points_with_intensity, mode):
        '''
        Get an occupancy grid of intensity, each occupied voxel has a value equal to the average intensity value of the points in that voxel. 

        points_with_intensity: (N,d) where d >= 4
        mode: polar or spherical

        return an intensity grid of shape (#r, #theta, #z)
        '''
        grid_idxs = torch.tensor(self.get_grid_ind(cart2polar(points_with_intensity[:,:3], mode=mode))) #(N,3)
        intensity = torch.tensor(points_with_intensity[:,3])#/255.0 # normalized intensity, (N,). INtensity values are between 0 and 255

        uniq_grid_idxs, inverse_idxs = torch.unique(grid_idxs, sorted=True, return_inverse=True, dim=0) #(K,3), (N,)
        intensity_sum = torch.zeros(len(uniq_grid_idxs)).float() #(K,)
        duplicate_count = torch.zeros(len(uniq_grid_idxs)).float() #(K,)
        intensity_sum.scatter_add_(0, inverse_idxs, intensity)
        duplicate_count.scatter_add_(0, inverse_idxs, torch.ones_like(intensity))
        intensity_sum/=duplicate_count

        intensity_occupancy = torch.zeros(*self.grid_size)
        intensity_occupancy[uniq_grid_idxs[:,0], uniq_grid_idxs[:,1], uniq_grid_idxs[:,2]] = intensity_sum
        
        return intensity_occupancy.numpy()

    ############### range image projection and de-projection ##################
    ############### assume that the voxelizer is using spherical coordinates
    def pc2range(self, point_cloud):
        '''
        Assuming the voxelizer is using spherical coordinates
        point_cloud: shape (N,3) or (N,4)
        return an intensity image and a range image
        '''
        assert(len(point_cloud.shape)==2)
        assert(point_cloud.shape[-1]==3 or point_cloud.shape[-1]==4)
        return_intensity = point_cloud.shape[-1]==4

        x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2] #cartesian
        r = np.sqrt(x**2 + y**2 + z**2) #range

        if return_intensity:
            intensity = point_cloud[:,3] #reflectance

        grid_inds = self.get_grid_ind(cart2polar(point_cloud[:,:3], mode="spherical")) #get grid index of each point as defined by voxelizer
        u, v = grid_inds[:,1], grid_inds[:,2]

        img_height, img_width = self.grid_size[2],self.grid_size[1]
        max_range = self.max_bound[0]

        range_image = np.full((img_height, img_width), max_range+100)
        
        # order in decreasing depth
        order = np.argsort(r)[::-1]
        r = r[order]
        u = u[order]
        v = v[order]
        range_image[v,u] = r
        assert(np.any(range_image<max_range))

        if return_intensity:
            intensity_image = np.full((img_height, img_width), 0.0)
            intensity_image[v,u] = intensity[order]
            return range_image, intensity_image

        return range_image

    def range2pc(self, range_image, intensity_image=None):
        """
        Assuming voxelizer is using spherical coordinates
        Converts a range image back to a point cloud and an intensity image back to the intensity for each point.
        intensity_image: shape (img_height, img_width)
        range_image: shape (img_height, img_width)
        """
        # Generate pixel grid for azimuth and elevation angles
        img_height, img_width = self.grid_size[2], self.grid_size[1]
        max_range = self.max_bound[0]

        x = np.arange(img_width) #corresponds to azimuth
        y = np.arange(img_height) #corresponds to elevation
        u, v = np.meshgrid(x, y)


        uv = np.stack((u.reshape(-1), v.reshape(-1)), axis=1)
        az_elev = ((uv[:, :].astype(np.float32)+0.5) * self.intervals[1:]) + self.min_bound[1:] #### 0.5 is very very important to avoid buggy discretization of the reconstructed point cloud (if no 0.5, some points will overlap during discretization)
        azimuth, elevation = az_elev[:,0], az_elev[:,1]

        
        # Get range values
        r = range_image.reshape(-1)

        # Filter out max_range values (no return)
        valid_mask = r < max_range + 100
        r = r[valid_mask]
        azimuth = azimuth[valid_mask]
        elevation = elevation[valid_mask]

        points_polar = np.stack((r, azimuth, elevation), axis=1)
        point_cloud = polar2cart(points_polar, mode="spherical")

        if intensity_image is not None:
            intensity = intensity_image.flatten()
            intensity = intensity[valid_mask]
            point_cloud = np.concatenate((point_cloud, intensity[:, np.newaxis]), axis=1)

        return point_cloud



        

class PolarDataset(data.Dataset):
  def __init__(self, in_dataset, voxelizer:Voxelizer, rotate_aug = False, flip_aug = False, is_test=False, use_voxel_random_mask=False, vis=False, insert=False, use_intensity_grid=False, use_range_proj=False):
        '''
        Our pipelines or models directly access this dataset to obtain data. 
        - In_dataset: the dataset that returns a sample of the processed nuscenes data
        - grid_size: how many grids per dimension
        - set rotate_aug and flip_aug to True if you want data augmentation
        - set is_test to True if you want some auxiliary information about the dataset
        - IMPORTANT: set use_voxel_random_mask to True if you want to randomly drop some mask over the background voxels
        - insert: am I inserting vehicle or not
        - use_intensity_grid: whether you are using this dataset to train a network that predicts an intensity grid
        - use_range_proj: whether you are using this dataset to train a network that predicts intensity on a range image
        '''
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.voxelizer = voxelizer
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.is_test = is_test
        self.current_iter = 0.0
        self.occupancy_ratio_total = 0.0
        self.use_random_mask = use_voxel_random_mask
        print("YOU use voxel random mask: ", self.use_random_mask)
        self.vis = vis
        if isinstance(in_dataset, torch.utils.data.Subset):
            self.mode = in_dataset.dataset.mode
            self.use_z = in_dataset.dataset.use_z
        else:
            self.mode = in_dataset.mode
            self.use_z = in_dataset.use_z
            
        self.insert = insert
        self.use_intensity_grid = use_intensity_grid
        self.use_range_proj = use_range_proj
        print("polar dataset COORDINATE mode: ", self.mode)
        print("polar dataset use_z: ", self.use_z)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

  def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if data==None:
            return None
        new_points_xyz_no_bckgrnd, points_xyz, occlude_mask, obj_region_list, points_in_box_mask, obj_properties = data

        points_within_bound_mask = self.voxelizer.filter_in_bound_points(cart2polar(points_xyz, mode=self.mode))
        points_xyz = points_xyz[points_within_bound_mask]
        points_in_box_mask = points_in_box_mask[points_within_bound_mask]

        new_points_xyz_has_bckgrnd = points_xyz#[:,:3]
        
        nonempty_boxes = []
        if self.insert:
            #nonempty_boxes = []
            for i, box in enumerate(obj_properties[9]):
                mask = points_in_box(box, new_points_xyz_has_bckgrnd[:,:3].T, wlh_factor = 1.0).astype(int)
                if np.sum(mask)!=0:
                    nonempty_boxes.append(box)

        # if self.is_test or not self.use_random_mask:
        #     # don't do datat augmentation during evaluation or actor insertion
        #     assert(self.rotate_aug==False)
        #     assert(self.flip_aug==False)
        
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random()*360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            #new_points_xyz_no_bckgrnd[:,:2] = np.dot(new_points_xyz_no_bckgrnd[:,:2],j)
            new_points_xyz_has_bckgrnd[:,:2] = np.dot(new_points_xyz_has_bckgrnd[:,:2],j)

            for i in range(len(obj_region_list)):
                obj_region_list[i] = rotate_obj_region(obj_region_list[i], rotate_rad)

        # random data augmentation by flipping either x or y
        if self.flip_aug:
            flip_type = np.random.choice(2,1)
            # for points in [new_points_xyz_no_bckgrnd, new_points_xyz_has_bckgrnd]:
            for points in [new_points_xyz_has_bckgrnd]:
                if flip_type==0:
                    points[:,0] = -points[:,0]
                elif flip_type==1:
                    points[:,1] = -points[:,1]
                else:
                    raise Exception("++++ flip aug has some problem")
                
            if flip_type==0:
                for i in range(len(obj_region_list)):
                    obj_region_list[i] = flip_obj_region(obj_region_list[i], axis="x", mode=self.mode)
            elif flip_type==1:
                for i in range(len(obj_region_list)):
                    obj_region_list[i] = flip_obj_region(obj_region_list[i], axis="y", mode=self.mode)
                
        max_bound = self.voxelizer.max_bound
        min_bound = self.voxelizer.min_bound

        
        new_points_polar_has_bckgrnd = cart2polar(new_points_xyz_has_bckgrnd[:,:3], mode=self.mode)
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = self.voxelizer.voxelize(new_points_polar_has_bckgrnd, return_point_info=False)

        # create voxel mask by occlusion: Set voxel labels of all voxels occluded by the obj_region to 1
        obj_voxels_mask_list = []
        if len(obj_region_list)!=0:
            if True:
                voxels_labels = self.voxelizer.create_mask_by_occlusion(obj_region_list[0], use_z=self.use_z)
                for i in range(len(obj_region_list)):
                    obj_voxel_mask = self.voxelizer.create_mask_by_occlusion(obj_region_list[i], use_z=self.use_z)
                    voxels_labels += obj_voxel_mask
                    obj_voxels_mask_list.append(obj_voxel_mask)
                voxels_labels = (voxels_labels>=1).astype(np.int64)
            else:
                occluded_occupancy, voxels_labels = self.voxelizer.voxelize_and_occlude_2(voxels_occupancy_has, new_points_polar_has_bckgrnd[points_in_box_mask==1])
        else:
            ### no obj
            voxels_labels = np.zeros(self.voxelizer.grid_size)

        BEV_labels = get_BEV_label(voxels_labels)

        # IMPORTANT: we don't use the following point information: grid_ind_no, return_points_no, voxel_centers_no 
        # these are the same as what we did for points with background, but we will modify the voxels_occupancy_no later
        grid_ind_no = np.copy(grid_ind_has)
        return_points_no = np.copy(return_points_has)
        voxel_centers_no = np.copy(voxel_centers_has)
        voxels_occupancy_no = np.copy(voxels_occupancy_has)
        
        if self.use_random_mask: ########### set this True for training 
            #### if a voxel of BEV_labels contains obj point, it is 1, for filtering out obj-containing mask in training
            voxels_labels = np.zeros(self.voxelizer.grid_size)
            obj_grid_ind = self.voxelizer.get_grid_ind(new_points_polar_has_bckgrnd[points_in_box_mask==1])
            voxels_labels[obj_grid_ind[:,0], obj_grid_ind[:,1], obj_grid_ind[:,2]] = 1
            BEV_labels = get_BEV_label(voxels_labels)

        has_bckgrnd_data = (grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has)
        no_bckgrnd_data = (grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no)
        voxel_label_data = (voxels_labels, BEV_labels) # voxel mask for mask git
        
        data_tuple = (has_bckgrnd_data, no_bckgrnd_data, voxel_label_data)

        if self.use_intensity_grid:
            intensity_grid = self.voxelizer.get_intensity_grid(new_points_xyz_has_bckgrnd, mode=self.mode)
            data_tuple = (has_bckgrnd_data, no_bckgrnd_data, voxel_label_data, intensity_grid)
        elif self.use_range_proj:
            range_image, intensity_image = self.voxelizer.pc2range(new_points_xyz_has_bckgrnd[:,:4])
            range_intensity_data = (range_image, intensity_image)
            data_tuple = (has_bckgrnd_data, no_bckgrnd_data, voxel_label_data, range_intensity_data)
            


        ## visualization
        if self.vis:
            max_radius = np.max(new_points_polar_has_bckgrnd[:,0])/4
            #obj_properties[5] are boxes of vehicles
            plot_obj_regions([], obj_region_list, points_xyz, max_radius, boxes=[], xlim=[-20,20], ylim=[-20,20], title="augmented", path="./test_figures", name="augmented", vis=False)

        if self.is_test:
            # for visualization or evaluation
            if obj_properties is not None:
                self.lidar_sample_token = obj_properties[8]
            self.nonempty_boxes = nonempty_boxes
            self.obj_properties = obj_properties
            self.obj_voxels_mask_list = obj_voxels_mask_list
            self.points_in_box_mask = points_in_box_mask
            self.points_xyz = points_xyz # original points that includes intensity
            num_r, num_theta, num_z = voxels_occupancy_has.shape
            self.occupancy_ratio_total += np.sum(voxels_occupancy_has.astype(np.float64))/(num_r*num_theta*num_z)
            self.current_iter += 1

        return data_tuple
  
  
  



     


def collate_fn_BEV(data):
    has_bckgrnd_datas = [d[0] for d in data]
    no_bckgrnd_datas = [d[1] for d in data]
    voxel_datas = [d[2] for d in data]

    voxel_label_list = [d[0] for d in voxel_datas]
    BEV_label_list = [d[1] for d in voxel_datas]
    voxel_label = torch.from_numpy(np.stack(voxel_label_list, axis=0)) #(B,#r,#theta,#z)
    BEV_label = torch.from_numpy(np.stack(BEV_label_list, axis=0)) #(B,#r,#theta)

    voxel_occupancy_has = np.stack([d[3] for d in has_bckgrnd_datas], axis=0) #(B,#r,#theta,#z)
    voxel_occupancy_no = np.stack([d[3] for d in no_bckgrnd_datas], axis=0) #(B,#r,#theta,#z)

    # grid_ind_has = [torch.from_numpy(d[0]) for d in has_bckgrnd_datas] #(B,num_points_i,3), list
    # grid_ind_no = [torch.from_numpy(d[0])for d in no_bckgrnd_datas] #(B,num_points_i,3), list
    # return_points_has = [torch.from_numpy(d[1]) for d in has_bckgrnd_datas] #(B,num_points_i,6), list
    # return_points_no = [torch.from_numpy(d[1]) for d in no_bckgrnd_datas] #(B,num_points_i,6), list
    # voxel_centers_has = [torch.from_numpy(d[2]) for d in has_bckgrnd_datas] #(B,num_points_i,3), list
    # voxel_centers_no = [torch.from_numpy(d[2]) for d in no_bckgrnd_datas] #(B,num_points_i,3), list

    grid_ind_has = None
    grid_ind_no = None
    return_points_has = None
    return_points_no = None
    voxel_centers_has = None
    voxel_centers_no = None

    has = (grid_ind_has, return_points_has, voxel_centers_has, torch.from_numpy(voxel_occupancy_has))
    no = (grid_ind_no, return_points_no, voxel_centers_no, torch.from_numpy(voxel_occupancy_no))

    data_batch = (has, no, voxel_label, BEV_label)

    return data_batch

def collate_fn_BEV_intensity(data):
    has_bckgrnd_datas = [d[0] for d in data]
    no_bckgrnd_datas = [d[1] for d in data]
    voxel_datas = [d[2] for d in data]

    voxel_label_list = [d[0] for d in voxel_datas]
    BEV_label_list = [d[1] for d in voxel_datas]
    voxel_label = torch.from_numpy(np.stack(voxel_label_list, axis=0)) #(B,#r,#theta,#z)
    BEV_label = torch.from_numpy(np.stack(BEV_label_list, axis=0)) #(B,#r,#theta)

    voxel_occupancy_has = np.stack([d[3] for d in has_bckgrnd_datas], axis=0) #(B,#r,#theta,#z)
    voxel_occupancy_no = np.stack([d[3] for d in no_bckgrnd_datas], axis=0) #(B,#r,#theta,#z)

    # grid_ind_has = [torch.from_numpy(d[0]) for d in has_bckgrnd_datas] #(B,num_points_i,3), list
    # grid_ind_no = [torch.from_numpy(d[0])for d in no_bckgrnd_datas] #(B,num_points_i,3), list
    # return_points_has = [torch.from_numpy(d[1]) for d in has_bckgrnd_datas] #(B,num_points_i,6), list
    # return_points_no = [torch.from_numpy(d[1]) for d in no_bckgrnd_datas] #(B,num_points_i,6), list
    # voxel_centers_has = [torch.from_numpy(d[2]) for d in has_bckgrnd_datas] #(B,num_points_i,3), list
    # voxel_centers_no = [torch.from_numpy(d[2]) for d in no_bckgrnd_datas] #(B,num_points_i,3), list

    grid_ind_has = None
    grid_ind_no = None
    return_points_has = None
    return_points_no = None
    voxel_centers_has = None
    voxel_centers_no = None

    has = (grid_ind_has, return_points_has, voxel_centers_has, torch.from_numpy(voxel_occupancy_has))
    no = (grid_ind_no, return_points_no, voxel_centers_no, torch.from_numpy(voxel_occupancy_no))

    intensity_datas = [d[3] for d in data]
    intensity_grid = np.stack(intensity_datas, axis=0) #(B,#r,#theta,#z)
    data_batch = (has, no, voxel_label, BEV_label, torch.from_numpy(intensity_grid))

    return data_batch

def collate_fn_range_intensity(data):
    has_bckgrnd_datas = [d[0] for d in data]
    no_bckgrnd_datas = [d[1] for d in data]
    voxel_datas = [d[2] for d in data]

    voxel_label_list = [d[0] for d in voxel_datas]
    BEV_label_list = [d[1] for d in voxel_datas]
    voxel_label = torch.from_numpy(np.stack(voxel_label_list, axis=0)) #(B,#r,#theta,#z)
    BEV_label = torch.from_numpy(np.stack(BEV_label_list, axis=0)) #(B,#r,#theta)

    voxel_occupancy_has = np.stack([d[3] for d in has_bckgrnd_datas], axis=0) #(B,#r,#theta,#z)
    voxel_occupancy_no = np.stack([d[3] for d in no_bckgrnd_datas], axis=0) #(B,#r,#theta,#z)

    grid_ind_has = None
    grid_ind_no = None
    return_points_has = None
    return_points_no = None
    voxel_centers_has = None
    voxel_centers_no = None

    has = (grid_ind_has, return_points_has, voxel_centers_has, torch.from_numpy(voxel_occupancy_has))
    no = (grid_ind_no, return_points_no, voxel_centers_no, torch.from_numpy(voxel_occupancy_no))

    range_image_datas = [d[3][0] for d in data]
    intensity_image_datas = [d[3][1] for d in data]
    range_images = np.stack(range_image_datas, axis=0)
    intensity_images = np.stack(intensity_image_datas, axis=0)
    range_intensity_datas = (torch.from_numpy(range_images), torch.from_numpy(intensity_images))
    data_batch = (has, no, voxel_label, BEV_label, range_intensity_datas)

    return data_batch