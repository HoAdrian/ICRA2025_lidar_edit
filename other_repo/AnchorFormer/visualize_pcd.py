import os
import open3d
import numpy as np

taxonomy_id = "kitti-car"
sparse_dir = f"/home/shinghei/lidar_generation/AnchorFormer/test_results_KITTI/{taxonomy_id}/sparse" #"/home/shinghei/lidar_generation/AnchorFormer/test_results"
files = os.listdir(sparse_dir)
# prefix = "input"
# # List all files in the directory that start with the prefix
# input_files = [f for f in files if f.startswith(prefix)]
# prefix = "dense"
# # List all files in the directory that start with the prefix
# dense_files = [f for f in files if f.startswith(prefix)]

prefix = "sample"
# List all files in the directory that start with the prefix
input_files = [f for f in files if f.startswith(prefix)]
input_files.sort()

dense_dir = f"/home/shinghei/lidar_generation/AnchorFormer/test_results_KITTI/{taxonomy_id}/dense" #"/home/shinghei/lidar_generation/AnchorFormer/test_results"
files = os.listdir(dense_dir)
prefix = "sample"
# List all files in the directory that start with the prefix
dense_files = [f for f in files if f.startswith(prefix)]
dense_files.sort()


assert(len(input_files)==len(dense_files))

for idx in range(len(input_files)):
    #i = idx*50
    print("input_files: ", input_files[idx])
    print("dense_files: ", dense_files[idx])
    input_points = open3d.io.read_point_cloud(f"{sparse_dir}/{input_files[idx]}").points
    dense_points = open3d.io.read_point_cloud(f"{dense_dir}/{dense_files[idx]}").points
    dense_pcd = open3d.geometry.PointCloud()
    dense_pcd.points = open3d.utility.Vector3dVector(np.array(dense_points))
    pcd_colors = np.tile(np.array([[0,0,1]]), (len(dense_points), 1))
    dense_pcd.colors = open3d.utility.Vector3dVector(pcd_colors)

    sparse_pcd = open3d.geometry.PointCloud()
    sparse_pcd.points = open3d.utility.Vector3dVector(np.array(input_points))
    pcd_colors = np.tile(np.array([[1,0,0]]), (len(input_points), 1))
    sparse_pcd.colors = open3d.utility.Vector3dVector(pcd_colors)

    open3d.visualization.draw_geometries([sparse_pcd, dense_pcd.translate((1,0,0))]) 

