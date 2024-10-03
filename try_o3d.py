import open3d as o3d
import numpy as np

# Create a simple point cloud
pcd = o3d.geometry.PointCloud()

# Add some points to the point cloud
pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))

# Try to visualize the point cloud
o3d.visualization.draw_geometries([pcd])
