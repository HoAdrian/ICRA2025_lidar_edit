import open3d as o3d
import numpy as np
import argparse

def generate_random_point_cloud(num_points=1000):
    """Generate a random point cloud for testing."""
    points = np.random.rand(num_points, 3) * 10  # Scale points for better visualization
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

def visualize_point_cloud(file_path=None):
    """Load and visualize a point cloud from a file or generate a random one."""
    if file_path:
        pcd = o3d.io.read_point_cloud(file_path)
    else:
        pcd = generate_random_point_cloud()
    
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Viewer")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a point cloud in Open3D.")
    parser.add_argument("--file", type=str, help="Path to point cloud file (e.g., .ply, .pcd, .xyz)")
    args = parser.parse_args()
    
    visualize_point_cloud(args.file)