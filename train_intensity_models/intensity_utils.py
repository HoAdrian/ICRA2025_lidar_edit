import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os


def voxelizer_pc2range(voxelizer, point_cloud):
    '''
    using a voxelizer
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

    grid_inds = voxelizer.get_grid_ind(cart2polar(point_cloud[:,:3], mode="spherical")) #get grid index of each point as defined by voxelizer
    u, v = grid_inds[:,1], grid_inds[:,2]

    img_height, img_width = voxelizer.grid_size[2], voxelizer.grid_size[1]
    max_range = voxelizer.max_bound[0]

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

def voxelizer_range2pc(voxelizer, range_image, intensity_image=None):
    """
    using a voxelizer
    Converts a range image back to a point cloud and an intensity image back to the intensity for each point.
    intensity_image: shape (img_height, img_width)
    range_image: shape (img_height, img_width)
    """
    # Generate pixel grid for azimuth and elevation angles
    img_height, img_width = voxelizer.grid_size[2], voxelizer.grid_size[1]
    max_range = voxelizer.max_bound[0]

    x = np.arange(img_width) #corresponds to azimuth
    y = np.arange(img_height) #corresponds to elevation
    u, v = np.meshgrid(x, y)


    uv = np.stack((u.reshape(-1), v.reshape(-1)), axis=1)
    az_elev = ((uv[:, :].astype(np.float32)+0.5) * voxelizer.intervals[1:]) + voxelizer.min_bound[1:]
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


def point_cloud_to_range_image(point_cloud, fov_up=10, fov_down=-30, img_width=512, img_height=32, max_range=100.0):
    '''
    point_cloud: shape (N,3) or (N,4)
    fov_up: max angle of elevation in degrees
    fov_down: min angle of elevation in degrees

    return an intensity image and a range image
    '''
    assert(len(point_cloud.shape)==2)
    assert(point_cloud.shape[-1]==3 or point_cloud.shape[-1]==4)
    return_intensity = point_cloud.shape[-1]==4

    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]

    r = np.sqrt(x**2 + y**2 + z**2)

    if return_intensity:
        intensity = point_cloud[:,3]
    
    # azimuth (horizontal angle) and elevation (vertical angle)
    r[r==0] = 1e-6
    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z / r)
    assert(np.max(elevation)<fov_up)
    assert(np.min(elevation)>fov_down)

    az_elev = np.stack((azimuth, elevation), axis=1)
    uniq = np.unique(np.round(az_elev, decimals=6), axis=0)
    print("num uniq: ", len(uniq))
    print("orig num: ", len(az_elev))
    #print("elevation: ", np.sum(elevation>fov_up))
    # print(f"Azimuth range: {np.min(np.rad2deg(azimuth))}, {np.max(np.rad2deg(azimuth))}")
    # print(f"Elevation range: {np.min(np.rad2deg(elevation))}, {np.max(np.rad2deg(elevation))}")
    # assert(not np.any(r>max_range))
    #print(az_elev)

    fov_up_rad = np.radians(fov_up)
    fov_down_rad = np.radians(fov_down)
    assert(fov_down_rad<0)

    azimuth = np.clip(azimuth, -np.pi, np.pi)
    elevation = np.clip(elevation, fov_down_rad, fov_up_rad)
    
    # Normalize angles to image coordinates
    u = (azimuth + np.pi) / (2*np.pi) * float((img_width-1))  # Horizontal coordinates
    v = (1 - ((elevation - float(fov_down_rad)) / float(fov_up_rad - fov_down_rad))) * float((img_height-1))
    #v = ((elevation - fov_down_rad) / float(fov_up_rad - fov_down_rad)) * float(img_height)  # Vertical coordinates

    uv = np.stack((u, v), axis=1)
    uniq = np.unique(uv, axis=0)
    print("mid  uv num uniq: ", len(uniq))
    print("mid  uv orig num: ", len(uv))
    #print(uv)

    # u = np.clip(np.floor(u), 0, img_width - 1).astype(np.int64) #(N,)
    # v = np.clip(np.floor(v), 0, img_height - 1).astype(np.int64) #(N,)
    u = np.floor(u).astype(np.int64)
    v = np.floor(v).astype(np.int64)

    uv = np.stack((u, v), axis=1)
    uniq = np.unique(uv, axis=0)
    print("uv num uniq: ", len(uniq))
    print("uv orig num: ", len(uv))
    #print(uv)

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

def range_image_to_point_cloud(range_image, intensity_image=None, fov_up=10, fov_down=-30, img_width=512, img_height=32, max_range=100.0):
    """
    Converts a range image back to a point cloud and an intensity image back to the intensity for each point.
    
    intensity_image: shape (img_height, img_width)
    range_image: shape (img_height, img_width)
    fov_up: max angle of elevation in degrees
    fov_down: min angle of elevation in degrees
    """
    # Convert FOV to radians
    fov_up_rad = np.radians(fov_up)
    fov_down_rad = np.radians(fov_down)

    # Generate pixel grid for azimuth and elevation angles
    x = np.arange(img_width)
    y = np.arange(img_height)
    u, v = np.meshgrid(x, y)

    img_width = float(img_width)
    img_height = float(img_height)
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # Convert image coordinates to azimuth and elevation angles
    azimuth = (u / (img_width)) * 2 * np.pi - np.pi
    elevation = (1 - v / (img_height)) * (fov_up_rad - fov_down_rad) + fov_down_rad
    #elevation = (v / img_height) * (fov_up_rad - fov_down_rad) + fov_down_rad

    # Get range values
    r = range_image.flatten()
    azimuth = azimuth.flatten()
    elevation = elevation.flatten()

    # Filter out max_range values (no return)
    valid_mask = r < max_range + 100
    r = r[valid_mask]
    azimuth = azimuth[valid_mask]
    elevation = elevation[valid_mask]

    # Convert to Cartesian coordinates
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)

    # Stack coordinates into point cloud
    point_cloud = np.stack((x, y, z), axis=-1)

    if intensity_image is not None:
        intensity = intensity_image.flatten()
        intensity = intensity[valid_mask]
        point_cloud = np.concatenate((point_cloud, intensity[:, np.newaxis]), axis=1)

    return point_cloud



def plot_range_img(img, path, name, vis=False):
  # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 20))  # Adjusted to plot one image
    
    # Plot the range image
    img = np.copy(img)
    # img/=np.max(img)
    im = ax.imshow(img, cmap="viridis")
    ax.set_title('Range image')
    
    # Add a colorbar to indicate the range
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Layout adjustment
    plt.tight_layout()
    
    # Show the image if vis is set to True
    if vis:
        plt.show()

    # Save the figure if path and name are provided
    if path is not None and name is not None:
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/{name}.png")
        print(f"Range image Figure {name}.png saved to {path}")
    plt.close(fig)  # Close the figure after saving to free up memory


import copy
import os
import numpy as np
import argparse

import sys
os.system(f"pwd")

sys.path.append("./")
sys.path.append("./datasets")
sys.path.append("./models")


from datasets.data_utils import *
from datasets.dataset import Voxelizer, PolarDataset, collate_fn_BEV_intensity
from datasets.dataset_nuscenes import Nuscenes
import torch
import configs.nuscenes_config as config

from models.vqvae_transformers import VQVAETrans, voxels2points
import open3d

from scipy.spatial import KDTree

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--trainval_data_path', type=str, default="data/nuscenes/v1.0-mini", help="path to training and validation examples")
    parser.add_argument('--data_version', type=str, default="v1.0-mini", help="versioin of the dataset e.g. nuscenes")
    parser.add_argument('--figures_path', type=str, default="./train_intensity_models", help="path to save the figures")
    args = parser.parse_args()

    config.device = "cpu"
    device = torch.device(config.device)
    print("--- device: ", device)

    ######### use spherical or polar coordinates
    mode = config.mode
    use_z = False
    if mode=="spherical":
        use_z=True

    train_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'train', filter_valid_scene=False, use_z=use_z, mode=mode)
    val_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'val', filter_valid_scene=False, use_z=use_z, mode=mode)

    voxelizer = Voxelizer(grid_size=config.grid_size, max_bound=config.max_bound, min_bound=config.min_bound)
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, use_intensity_grid=True)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = False, rotate_aug = False, is_test=True, use_intensity_grid=True)
   
    print("+++ num train: ", len(train_dataset))
    print("+++ num val: ", len(val_dataset))

    num_vis = 81
    dataset = val_dataset
    samples = np.random.choice(len(dataset), num_vis) #np.arange(len(dataset))#np.random.choice(len(dataset), num_vis)
    l2_errors = []
    for k in samples:
        #k = 31 #56 #66 #31 #44 #56 #31 #56 #31 #66, 31
        print(f"++++++++++|||||| sample index: {k}")
        data_tuple = collate_fn_BEV_intensity([dataset.__getitem__(k)])
        has, no, voxel_label, BEV_label, intensity_grid = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label.to(device) #(1, #r, #theta, #z)
        BEV_mask = BEV_label.to(device) #(1, #r, #theta)

        # voxelizer.verify_occlusion(voxels_occupancy_has[0].numpy())

        points = voxels2points(voxelizer, voxels_occupancy_has.permute(0,3,1,2), mode=mode)[0]
        
        non_zero_indices = torch.nonzero(voxels_occupancy_has[0].detach().cpu(), as_tuple=True)
        point_intensity = intensity_grid[0][non_zero_indices[0].numpy(), non_zero_indices[1].numpy(), non_zero_indices[2].numpy()] # get the intensity of the occupied voxels
        points = np.concatenate((points, point_intensity[:, np.newaxis]), axis=1).astype(np.float64)

        #points = dataset.points_xyz[:,:4]
        #points = points[:6] #6 #50
        
        ###################### not using voxelizer 
        # const = 1 #5
        # range_image, intensity_image = point_cloud_to_range_image(points, fov_up=config.fov_up, fov_down=config.fov_down, img_width=config.grid_size[1]*const, img_height=config.grid_size[2]*const, max_range=config.max_range)
        # plot_range_img(range_image, args.figures_path, name="range image", vis=False)
        # rec_points = range_image_to_point_cloud(range_image, intensity_image, fov_up=config.fov_up, fov_down=config.fov_down, img_width=config.grid_size[1]*const, img_height=config.grid_size[2]*const, max_range=config.max_range)

        # grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has_rec = voxelizer.voxelize(cart2polar(rec_points[:,:3], mode=mode), return_point_info=False)
        # voxelizer.verify_occlusion(voxels_occupancy_has_rec)
        
        # range_image2, intensity_image2 = point_cloud_to_range_image(rec_points, fov_up=config.fov_up, fov_down=config.fov_down, img_width=config.grid_size[1]*const, img_height=config.grid_size[2]*const, max_range=config.max_range)
        # rec_points2 = range_image_to_point_cloud(range_image2, intensity_image=intensity_image2, fov_up=config.fov_up, fov_down=config.fov_down, img_width=config.grid_size[1]*const, img_height=config.grid_size[2]*const, max_range=config.max_range)
        
        ###################### using voxelizer
        const = 1 #5
        range_image, intensity_image = voxelizer_pc2range(voxelizer, points)
        plot_range_img(range_image, args.figures_path, name="range image", vis=False)
        rec_points = voxelizer_range2pc(voxelizer, range_image, intensity_image)
        
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has_rec = voxelizer.voxelize(cart2polar(rec_points[:,:3], mode=mode), return_point_info=False)
        voxelizer.verify_occlusion(voxels_occupancy_has_rec)
        
        range_image2, intensity_image2 = voxelizer_pc2range(voxelizer, rec_points)
        rec_points2 = voxelizer_range2pc(voxelizer, range_image2, intensity_image2)

        print("rec_points: ", rec_points)
        print("rec points 2: ", rec_points2)
        
        ##### points through 1 range projection and reprojection VS points through 2 range projection and reprojection
        print(f"###### rec vs rec2: {len(rec_points)} {len(rec_points2)}")
        assert(len(rec_points)==len(rec_points2))

        colors = np.array([0,1,0]).astype(np.float64)[np.newaxis,:]
        #visualize_pointcloud(rec_points[:, :3], colors*rec_points[:,3:4]/255.0)
        #visualize_pointcloud(rec_points2[:, :3], colors*rec_points2[:,3:4]/255.0)
        assert(len(rec_points)==len(rec_points2))

        

        ############ compare orig and rec points ############
        print("orig num points", len(points))
        print("rec num points", len(rec_points))

        
        colors = np.array([0,1,0]).astype(np.float64)[np.newaxis,:]
        visualize_pointcloud(points[:, :3], colors*points[:,3:4]/255.0)
        visualize_pointcloud(rec_points[:, :3], colors*rec_points[:,3:4]/255.0)

        ### get reconstruction error of range image representation
        kd_tree = KDTree(rec_points[:,:3])
        _, nearest_idxs = kd_tree.query(points[:,:3], k=1)
        nn_intensity = rec_points[:,3][nearest_idxs].astype(np.float64)
        error = np.sqrt(np.sum((nn_intensity - points[:,3])**2)/len(points))
        print(error)
        l2_errors.append(error)

    print(np.mean(np.array(l2_errors)))
