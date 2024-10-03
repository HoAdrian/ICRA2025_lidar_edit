import sys
sys.path.append("../")
from dataset import Voxelizer
from data_utils import cart2polar, polar2cart, plot_points_and_voxels
import numpy as np
import torch


def voxel2patch_label(binary_voxel_label, patch_size):
    """
    convert voxel-wise label to patch-wise label. If the patch is associated with a voxel labeled 1, then that patch is labeled 1.
    
    Inputs:
        binary_voxel_label (torch tensor): Binary label array of shape (batch, H, W); label on each "pixel" (bird eye view)
        patch_size (int): Size of each patch (p)
    
    Returns:
        patch_label (torch tensor): Array of shape (batch, H/p, W/p) with binary 1 or 0 label
    """
    B, H, W = binary_voxel_label.shape
    patched_height = H // patch_size
    patched_width = W // patch_size

    # Reshape binary_label to make it easier to iterate over patches
    binary_label_reshaped = binary_voxel_label.reshape(B, patched_height, patch_size, patched_width, patch_size)
    
    # Sum the values within each patch to check if any pixel within the patch is labeled as 1
    patch_label = binary_label_reshaped.sum(dim=(2, 4)) > 0
    
    return patch_label.long()
    
def patch2voxel_label(binary_patch_label, patch_size):
    """
    convert voxel-wise label to patch-wise label. If the patch is associated with a voxel labeled 1, then that patch is labeled 1.
    
    Inputs:
        binary_patch_label (torch tensor): Binary label array of shape (batch, H/p, W/p); label on each "pixel" (bird eye view)
        patch_size (int): Size of each patch (p)
    
    Returns:
        voxel_label (torch tensor): Array of shape (batch, H, W) with binary 1 or 0 label
    """
    B, H_p, W_p = binary_patch_label.shape
    H = H_p*patch_size
    W = W_p*patch_size

    # Reshape binary_label to make it easier to iterate over patches
    binary_label_reshaped = binary_patch_label.reshape(B, H_p, 1, W_p, 1)
    binary_label_reshaped = binary_label_reshaped.expand(-1,-1,patch_size,-1,patch_size)
    binary_label_reshaped = binary_label_reshaped.reshape(B, H, W)
    
    return binary_label_reshaped.long()


if __name__=="__main__":

    ###### test polar voxel
    # voxelizer = Voxelizer(grid_size=[10,9,1], max_bound=[10,2*np.pi,3], min_bound=[0,0,-5])

    # theta = np.pi
    # xyz_pol = np.array([[1,0,1],
    #                 [2,0,1],
    #                 [3,0,1],
    #                 [2,theta,1],
    #                 [3,theta,1],
    #                 [4,theta,1]]).astype(np.float64)
    
    # mask = np.array([0,0,0,1,1,1])
    # intensity = np.copy(mask)
    # intensity = intensity[:,np.newaxis]
    
    # xyz_pol = np.concatenate((xyz_pol, intensity), axis=1)

    # mask = mask==1
    
    # xyz = polar2cart(xyz_pol)
    
    # grid_ind, return_points, voxel_centers, voxels_labels, voxels_occupancy = voxelizer.voxelize(xyz_pol[:,:3], mask)

    # print("intervals (r,theta,z): \n", voxelizer.intervals)
    # print("grid_ind (r,theta,z): \n", grid_ind)

    # voxel_position = torch.from_numpy(voxelizer.voxel_position)
    # voxel_position_flat = polar2cart(voxel_position.permute(1,2,3,0)[:,:,0,:].reshape(-1,3).numpy())
    
    # voxel_label_flat = voxels_labels.reshape(-1)
    # plot_points_and_voxels(xyz, xyz[:,3], voxel_position_flat, voxel_label_flat, xlim=[-10,10], ylim=[-10,10], vis=True, title="voxel_label and lidar points")

    ###### test voxel label ==> patch label
    binary_label = np.array([[1,0,0,0],
                             [1,1,0,0],
                             [0,0,1,0],
                             [0,0,0,0]])
    
    # print(binary_label.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3))
    #binary_label_reshaped = binary_label.transpose(0, 1, 3, 2, 4)

    # Sum the values within each patch to check if any pixel within the patch is labeled as 1
    #patch_label = binary_label_reshaped.reshape(B, patched_height, patched_width, -1).max(axis=3) > 0
    

    binary_label = binary_label[np.newaxis,:]
    binary_label = torch.tensor(binary_label)
    
    patch_label=voxel2patch_label(binary_label, patch_size=2)
    print("converted patch label: \n", patch_label.squeeze(0))

    voxel_label = patch2voxel_label(patch_label, patch_size=2)
    print("converted voxel label:\n", voxel_label.squeeze(0))

    masked_idxs = torch.nonzero(voxel_label)
    print(masked_idxs)