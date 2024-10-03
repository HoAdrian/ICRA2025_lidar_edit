import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import torch


from sklearn.metrics import confusion_matrix
import sklearn.metrics

import re
import csv
import open3d

'''
General utilities for processing, visualizing data or evaluating quality of data
'''

def cart2polar(input_xyz, mode="polar"):
    '''
    input_xyz: (N,d), 
        with the first three dimensions x,y,z, also d>2
    mode: "polar" or "spherical"

    return (N,d) array, 
        first column is radius, second column is theta, the rest of the columns are unchanged. The returned thetas are in the range [0, 2*pi)
        if mode is spherical, follow cart2spherical instead
    '''
    if mode=="polar":
        r = np.sqrt(input_xyz[:,0:1]**2 + input_xyz[:,1:2]**2)
        theta = np.arctan2(input_xyz[:,1:2],input_xyz[:,0:1])
        theta[theta<0] += 2*np.pi # map the range of theta back to [0, 2*pi)
        return np.concatenate((r,theta,input_xyz[:,2:]),axis=1)
    elif mode=="spherical":
        return cart2spherical(input_xyz)
    else:
        raise Exception(f"the mode {mode} is invalid")

def polar2cart(input_xyz_polar, mode="polar"):
    '''
    input_xyz_polar: (N,d), 
        with the first three dimensions r, theta, z,  also d>2
        if mode is spherical, then the input is in spherical coordinate
    mode: "polar" or "spherical"

    return (N,d) array, 
        first column is x, second column is y, the rest of the columns are unchanged. 
        if mode is spherical, follow spherical2cart instead
    '''
    if mode=="polar":
        x = input_xyz_polar[:,0:1]*np.cos(input_xyz_polar[:,1:2])
        y = input_xyz_polar[:,0:1]*np.sin(input_xyz_polar[:,1:2])
        return np.concatenate((x,y,input_xyz_polar[:,2:]),axis=1)
    elif mode=="spherical":
        return spherical2cart(input_xyz_polar)
    else:
        raise Exception(f"the mode {mode} is invalid")


def cart2spherical(input_xyz):
    '''
    input_xyz: (N,d), with the first three dimensions x,y,z, also d>2

    return (N,d) array, first column is radius, second column is theta (on x-y), third column is phi (wrt z), the rest of the columns are unchanged. The returned thetas are in the range [0, 2*pi), returned phi are in the range [0, pi]
    '''
    r = np.sqrt(input_xyz[:,0:1]**2 + input_xyz[:,1:2]**2 + input_xyz[:,2:3]**2)
    theta = np.arctan2(input_xyz[:,1:2],input_xyz[:,0:1])
    theta[theta<0] += 2*np.pi # map the range of theta back to [0, 2*pi)
    r = np.where(r==0, 1e-6, r)
    phi = np.arccos(input_xyz[:,2:3]/r)
    return np.concatenate((r,theta,phi,input_xyz[:,3:]),axis=1)

def spherical2cart(input_xyz_spherical):
    '''
    input_xyz_spherical: (N,d), with the first three dimensions r, theta, phi,  also d>2

    return (N,d) array, first column is x, second column is y, the thrid column is z, the rest of the columns are unchanged. 
    '''
    radius = input_xyz_spherical[:,0:1]
    theta = input_xyz_spherical[:,1:2]
    phi = input_xyz_spherical[:,2:3]
    x = radius*np.sin(phi)*np.cos(theta)
    y = radius*np.sin(phi)*np.sin(theta)
    z = radius*np.cos(phi)
    return np.concatenate((x,y,z,input_xyz_spherical[:,3:]),axis=1)



def plot_points_and_voxels(lidar_xyz, intensity, voxel_xyz, labels, xlim=[-20,20], ylim=[-20,20], vis=True, title="lidar_points", path=None, name=None, vox_size=10):
    '''
    Plot lidar points points and voxel positions

    lidar_xyz: shape (N, d'), where each row is (x, y, z, intensity, ...)
    intensity: shape (N,), intensity value for each lidar point
    voxel_xyz: voxel positions
    labels: label for each voxel position
    vox_size: size of the dots representing voxels
    '''
    plt.figure(figsize=(8, 6))
    if voxel_xyz is not None:
        x = voxel_xyz[:,0]
        y = voxel_xyz[:,1]

        
        scatter_voxel = plt.scatter(x, y, c=labels, cmap='viridis', alpha=0.3, s=vox_size)
        plt.colorbar(scatter_voxel, label='voxel label')
    
    if lidar_xyz is not None:
        x = lidar_xyz[:,0]
        y = lidar_xyz[:,1]

        scatter_lidar = plt.scatter(x, y, c=intensity, cmap='viridis', alpha=0.75, s=3)
        if labels is None or voxel_xyz is None:
            plt.colorbar(scatter_lidar, label='lidar label')


    plt.xlim(xlim)
    plt.ylim(ylim)

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    if title != None:
        plt.title(title)

    if vis:
      plt.show()

    if path is not None and name is not None:
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/{name}.png")
        print(f"Figure {name}.png saved to {path}")


def plot_xy(xs, ys_list, labels_list, title, x_label, y_label, name, path, vis=False):
    fig, ax = plt.subplots()
    for idx in range(len(ys_list)):
        ax.scatter(xs, ys_list[idx], s=8)
        ax.plot(xs, ys_list[idx], label=labels_list[idx])
    ax.legend()
    if title!=None:
        ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if path!=None and name!=None:
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/{name}.png")
    if vis:
        plt.show()
    plt.cla()
    plt.close(fig)

def confusion_matrix_wrapper(expected, predicted, labels):
    '''
    Confusion matrix is a matrix C with Cij = number of samples predicted to be class j but is actually i

    For each class i:

    TP = number of samples of class i that are predicted as class i
    FP = number of samples of class j that are predicted as class i, j!=i
    TN = number of samples of class j that are predicted as class j, j!=i
    FN = number of samples of class i that are predicted as class j, j!=i

    TP + FP + TN + FN = number of samples

    Accuracy: (TP+TN)/(TP+FP+TN+FN)
    Precision: (TP)/(TP+FP), how many TP out of positive prediction
    Recall: (TP)/(TP+FN), how many TP out of actually positive samples
    F1-score: (2*precision*recall)/(precision+recall), harmonic mean of precision and recall
    Specificity: (TN)/(FP+TN), how many TN out of actually negative samples
    TPR: (TP)/(TP+FN), how many TP out of actually positive samples
    FPR: (FP)/(FP+TN), how many FP out of actually negative samples
    '''
    C = confusion_matrix_2_numpy(expected, predicted, labels=labels).astype(np.float64)
    num_samples = len(expected)
    num_classes = len(labels)
    total_accuracy = np.trace(C)/num_samples
    TPs = []
    FPs = []
    FNs = []
    TNs = []
    classes = np.arange(num_classes)
    for i in range(num_classes):
        TP = C[i,i]
        negative_classes = classes[classes!=i]
        FP = np.sum(C[negative_classes,i])
        FN = np.sum(C[i,negative_classes])
        TN = num_samples - TP - FP - FN
        TPs.append(TP)
        FPs.append(FP)
        FNs.append(FN)
        TNs.append(TN)
    return C, total_accuracy, np.array(TPs), np.array(FPs), np.array(FNs), np.array(TNs)

def compute_perf_metrics(TP, FP, FN, TN):
    '''
    Assume the inputs are all np arrays. For each array, element i is the value (TP, FP, FN or TN) of the class i,
    compute accuracy, precision, recall, f1_score, specificity, TPR, FPR
    '''
    accuracy= (TP+TN)/(TP+FP+TN+FN)
    precision= (TP)/(TP+FP)
    recall= (TP)/(TP+FN)
    f1_score = (2*precision*recall)/(precision+recall)
    specificity =  (TN)/(FP+TN)
    TPR= (TP)/(TP+FN)
    FPR= (FP)/(FP+TN)

    return accuracy, precision, recall, f1_score, specificity, TPR, FPR

def compute_auprc(true_labels, pred_probs):
    '''
    true_labels: list of ground truth labels , each 0 or 1
    pred_probs: list of predicted probabilities for positive class
    compute area under precision recall curve
    '''
    auprc = sklearn.metrics.average_precision_score(true_labels, pred_probs)
    return auprc

def confusion_matrix_1(y_true, y_pred, labels):
    N = len(labels)
    y_true = torch.tensor(y_true, dtype=torch.long)
    y_pred = torch.tensor(y_pred, dtype=torch.long)
    return torch.sparse.LongTensor(
        torch.stack([y_true, y_pred]), 
        torch.ones_like(y_true, dtype=torch.long),
        torch.Size([N, N])).to_dense()

def confusion_matrix_2(y_true, y_pred, labels):
    N = len(labels)
    y_true = torch.tensor(y_true, dtype=torch.long)
    y_pred = torch.tensor(y_pred, dtype=torch.long)
    y = N * y_true + y_pred # element_idx_in_2d_array = (row_idx * width_of_matrix + col_idx)
    y = torch.bincount(y)
    if len(y) < N * N:
        y = torch.cat(y, torch.zeros(N * N - len(y), dtype=torch.long))
    y = y.reshape(N, N)
    return y
    
def confusion_matrix_2_numpy(y_true, y_pred, labels):
    y_true = y_true.reshape(-1).astype(np.int64)
    y_pred = y_pred.reshape(-1).astype(np.int64)
    N = len(labels)
    y = N * y_true + y_pred
    y = np.bincount(y, minlength=N*N)
    y = y.reshape(N, N)
    return y

def extract_epoch_number(s):
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    return None

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def write_csv_row(file_path, row, overwrite=False):
    '''
    append a row to the csv file, optionally erase existing content
    '''
    ensure_dir(file_path)
    if overwrite:
        mode = "w"
    else:
        mode = "a"
    with open(file_path, mode, newline="") as csvfile: 
    # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        # writing the fields 
        csvwriter.writerow(row)
        
def write_csv_rows(file_path, rows, overwrite=False):
    '''
    append multiple rows to the csv file, optionally erase existing content
    '''
    ensure_dir(file_path)
    if overwrite:
        mode = "w"
    else:
        mode = "a"
    with open(file_path, mode, newline="") as csvfile: 
    # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        # writing the fields 
        csvwriter.writerows(rows) 

def load_csv_data(data_path, preppend_one=False):
    '''
    load csv data into a np array of type float
    '''
    data = []
    with open(data_path, mode ='r') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            data.append(line)
    #data.pop(0)
    data = np.array(data)
    data = data.astype(float)
    # preppend 1 to the end of each feature vector (bias trick)
    if preppend_one:
        one = np.ones((len(data), 1), dtype=float)
        data = np.concatenate((one, data), axis=1)
    return data

def plot_xy_from_csv(file_path, title, x_label, y_label, name, plot_path, vis=True):
    data = load_csv_data(file_path, preppend_one=False)
    start = 0
    epochs = data[start:,0] #epoch
    train = data[start:,1] #train
    val = data[start:,2] #val
    plot_xy(xs=epochs, ys_list=[train, val], labels_list=["train", "val"], title=title, x_label=x_label, y_label=y_label, name=name, path=plot_path, vis=vis)



def compute_allocentric_angle(obj2cam_pos, obj_right_axis, obj_front_axis):
  '''
  Given I have an object, vector R points from a sensor to the object center. The sensor is like a lidar sensor and the object is like a vehicle, and we consider birds eye view (x-y). 
  Vector B points from the object center to the front of the object. Vector A points from the object center to the right of the object. 
  Now, I want to get the angle between the vector R and vector A. the angle has the range from 0 to 2 pi. This angle is the allocentric angle alpha.  
  
  Alpha together with the viewing angle gamma (angle between the cam2obj_vector and the camera front pointing axis) fully determines which part of the vehicle
  is occluded from the view of the senose. Define the global yaw angle theta as the angle between the object's front pointing axis and the right axis of the camera. In fact

  theta = (alpha + gamma) mod (2*pi)
  
  We treat A as the x axis and B as the y axis, the allocentric angle should be consistent with the quadrant it is in. 

  -alpha varies from 0 to 2pi from object right axis to obj2cam_vector counterclockwise
  -gamma varies from 0 to 2pi from camera front axis clockwise
  -theta varies from object front axis to camera right axis counterclockwise

  obj_right_axis: np.ndarray (2,), the vector from object center to its right side (A)
  obj_front_axis: np.ndarray (2,), the vector from object center to its front side (B)
  obj2cam_pos: np.ndarray (2,), the vector from object center to the sensor (R)

  return: the allocentric angle in radian
  '''

  r = obj2cam_pos/np.linalg.norm(obj2cam_pos) # the vector to be evaluated
  a = obj_right_axis/np.linalg.norm(obj_right_axis) # x-axis
  b = obj_front_axis / np.linalg.norm(obj_front_axis) # y-axis

  cos = np.dot(r,a)
  sin = np.dot(r,b)
  allocentric = np.arctan2(sin, cos)

  if allocentric < 0:
     allocentric += 2*np.pi

  return allocentric


def compute_viewing_angle(cam2obj_vector):
    '''
    angle gamma between the cam2obj vector and the front axis (y axis) of the camera. gamma varies from 0 to 2pi from camera front axis clockwise. [0,2pi)
    cam2obj_vector: (2,) ndarray
    '''
    x, y = cam2obj_vector
    a = np.arctan2(y,x)
    if x<=0 and y==0:
        # degree negative pi
        return np.pi/2 + np.pi
    elif x<0 and y>0:
        # second quadrant
        return np.pi/2 + np.pi + (np.pi - a)
    else:
        return np.pi/2 - a

    # if x>0 and y>0:
    #     # first quadrant
    #     return np.pi/2 - a
    # elif x<0 and y>0:
    #     return np.pi/2 - a
    # elif x<0 and y<0:
    #     return np.pi/2 - a
    # elif x>0 and y<0:
    #     return np.pi/2 - a
    # if x==0:
    #     if y>=0:
    #         return 0
    #     else:
    #         return np.pi/2
    # elif y==0:
    #     if x>=0:
    #         np.pi/2
    #     else:


def visualize_pointcloud(points, pcd_colors):
    '''
    pcd_colors, each row is a rgb vector (length 3) for the corresponding point
    '''
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(points))
    pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
    
    mat = open3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    mat.point_size = 2.0

    open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], show_skybox=False)

def visualize_generated_pointclouds(voxelizer, voxels_occupancy_list, points_list, names_list, voxels_mask, image_path=None, image_name=None):
    '''
    voxels_mask: (1,H,W,C)
    voxels_occupancy_list: list of occupancy grid each of shape (H,W,C)
    points_list: list of the corresponding point cloud
    names_list: the corresponding names
    image_path: default is None. it is the location to save the point cloud visualization image
    '''
    
    assert(len(points_list)==len(voxels_occupancy_list)==len(names_list))

    for j, points in enumerate(points_list):
        #points = points[0]
        print(f"++++ visualizing {names_list[j]}")

        # get the grid index of each point
        voxel_occupancy = voxels_occupancy_list[j]
        non_zero_indices = torch.nonzero(voxel_occupancy.detach().cpu(), as_tuple=True)
        voxel_mask_ = voxels_mask[0].detach().cpu()

        point_intensity = np.zeros((len(points),))
        assert(len(points)==len(voxel_mask_[non_zero_indices[0], non_zero_indices[1], non_zero_indices[2]]))
        if True:
            # color the masked regions if there are points there
            point_intensity_mask = (voxel_mask_[non_zero_indices[0], non_zero_indices[1], non_zero_indices[2]] == 1).detach().cpu().numpy()
            point_intensity[point_intensity_mask] = 1
            print("**************** any points in mask region? ", (np.sum(point_intensity)))

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(points))
        pcd_colors = np.tile(np.array([[0,0,1]]), (len(points), 1))
        pcd_colors[point_intensity==1, 0] = 1
        pcd_colors[point_intensity==1, 2] = 0
        pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
    
        mat = open3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.point_size = 3.0

        open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], show_skybox=False)
        ### rotating point cloud to get nice video, change j as you need ###
        # if j==0:
        #     cam_right_vec = np.array([1.0, 0.0, 0.0])
        #     cam_pos = np.array([0.0, 0.0, 5.0])
        #     cam_target = np.array([0.0, 20.0, 10.0])
        #     cam_front_vec = cam_target - cam_pos
        #     cam_up_vec = np.cross(cam_right_vec, cam_front_vec)#np.array([0.5, 0.5,  1.0])
        #     visualize_rotating_open3d_objects([pcd], offsets=[[0.1,0,0]], shift_to_centroid=False, 
        #                                     rotation_axis = np.array([0, 0, 1]), rotation_speed_deg=0.45,
        #                                     cam_position=cam_pos, cam_target=cam_target, 
        #                                     cam_up_vector=cam_up_vec, zoom=0.1)

        if image_path is None:
            open3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], show_skybox=False)
        else:
            os.makedirs(image_path, exist_ok=True)
            vox_size = 1
            xlim = [-30, 30]
            ylim = [-30, 30]
            intensity=np.zeros((len(points),))
            intensity[point_intensity==1] = 1
            voxelizer.vis_BEV_binary_voxel(voxel_occupancy, points_xyz=points, intensity=intensity, vis=False, path=f"{image_path}", name=f"{image_name}_{names_list[j]}", vox_size=vox_size, xlim=xlim, ylim=ylim, only_points=True)
            

            # # Create the visualizer
            # vis = open3d.visualization.Visualizer()
            # vis.create_window(visible=True)  # Offscreen rendering

            # # Add geometry and customize the view
            # vis.add_geometry(pcd)
            # vis.update_geometry(pcd)
            # vis.get_render_option().point_size = 2.0
            # vis.get_render_option().background_color = np.array([0, 0, 0])

            # # Update the visualizer and render the scene
            # vis.poll_events()
            # vis.update_renderer()
            # # Save the screenshot
            # vis.capture_screen_image(image_path)
            # # Destroy the visualizer window
            # vis.destroy_window()

            print(f"-- Saved point cloud visualization to {image_path}")


def visualize_rotating_open3d_objects(open3d_objects, offsets=[[0.1,0,0]], shift_to_centroid=False, 
                                      rotation_axis = np.array([0, 0, 1]), rotation_speed_deg=1,
                                      cam_position=None, cam_target=None, 
                                      cam_up_vector=None, zoom=None):
                                      
    """ 
    rotation_axis: The rotation axis (for example, [0, 0, 1] for Z-axis).
    num_frames: Number of frames for a full rotation.
    rotation_speed_deg: Rotation speed in degrees per frame.
    offset: how much each object is shifted.
    """

    for i, obj in enumerate(open3d_objects):
        if isinstance(obj, open3d.geometry.TriangleMesh):
            obj.compute_vertex_normals()
            if shift_to_centroid:
                obj.translate(-np.mean(np.asarray(obj.vertices), axis=0))
                    
        elif isinstance(obj, open3d.geometry.PointCloud):    
            print("Great, point cloud ( <0> A <0> )")
            #obj.translate(cam_position)
            # if shift_to_centroid:
            #     obj.translate(-np.mean(np.asarray(obj.points), axis=0))
        else:
            raise ValueError("Invalid open3d object. Can only be either open3d mesh, or open3d point cloud.")            
            
        # if len(offsets) == 1:
        #     obj.translate(tuple(i*np.array(offsets[0])))
        # elif len(offsets) == len(open3d_objects):
        #     obj.translate(tuple(offsets[i]))
        # else:
        #     raise ValueError("Invalid offsets array. len(offsets) must either = 1, or = len(open3d_objects)")  
        
    # Convert the rotation speed to radians per frame
    rotation_speed_rad = np.radians(rotation_speed_deg)

    # Create a visualization window
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    mat = open3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    mat.point_size = 3.0

    # Add the object to the visualization
    for open3d_object in open3d_objects:
        vis.add_geometry(open3d_object)

    vis.get_render_option().point_size = mat.point_size

    ### Change viewing angle and zoom
    if cam_position is not None and cam_target is not None and cam_up_vector is not None:
        front = cam_target-cam_position  # Camera direction 
        lookat = cam_target  # Camera target point 
        view_control = vis.get_view_control()
        view_control.set_front(front)
        view_control.set_lookat(lookat)
        view_control.set_up(cam_up_vector)  # Set the camera up vector

        # ctr = vis.get_view_control()
        # camera_params = ctr.convert_to_pinhole_camera_parameters()
        # # width, height = vis.get_window_size()
        # # camera_params.intrinsic.width = width
        # # camera_params.intrinsic.height = height
        # K = np.copy(camera_params.extrinsic)#np.eye(4) 
        # # K[:3, :3] = np.array([[0,1,0],
        # #                     [1,0,0],
        # #                     [0,0,-1]])
        # K[:3, -1] = cam_position#np.array([0,0,0])
        # camera_params.extrinsic = K
        # ctr.convert_from_pinhole_camera_parameters(camera_params)
    if zoom is not None:
        view_control.set_zoom(zoom)
        
            
    # Continuous rotation animation loop
    while True:  # Infinite loop for continuous rotation
        for obj in open3d_objects:
            # Calculate the rotation center as the local center of the current object
            rotation_center = np.array([0,0,0])#obj.get_center()

            # Create a rotation matrix for the current angle
            rotation_matrix = np.eye(4)
            rotation_matrix[:3, :3] = np.array([
                [np.cos(rotation_speed_rad), -np.sin(rotation_speed_rad), 0],
                [np.sin(rotation_speed_rad), np.cos(rotation_speed_rad), 0],
                [0, 0, 1]
            ])
            
            # Apply the rotation to the current object around its local center
            obj.translate(-rotation_center)  # Move object to origin
            obj.transform(rotation_matrix)    # Rotate
            obj.translate(rotation_center)    # Move back to center
        
        # Update the visualization
        for open3d_object in open3d_objects:
            vis.update_geometry(open3d_object)
        vis.poll_events()
        vis.update_renderer()

        



# def get_obj_mask(obj_region, points_polar, use_z=False):
#   '''
#   get the mask that masks out points enclosed and occluded by the object, defined by obj_region. By default, we ignore the z bounds of obj_region
#   obj_region: shape (2, 3), [[min_r, min_theta, min_z], [max_r, max_theta, max_z]]
#   points_polar: shape (N,d), d>=3, polar coordinates
#   use_z: whether to compute the mask with the z dimension of the obj rejion as well

#   *** points_polar is in spherical coordinates, then use_z indicates whether to compute the mask with the phi dimension of the object
#   '''
#   min_dim, max_dim = obj_region[0], obj_region[1]
#   r = points_polar[:,0]
#   theta = points_polar[:,1]
#   z = points_polar[:,2]
#   obj_regions = []

#   if (max_dim[1]>=3*np.pi/2 and min_dim[1]<=np.pi):
#    # # split region if it crosses the first and forth quadrants
#    obj_regions.append(np.array([np.array([min_dim[0], 0, min_dim[2]]), np.array([max_dim[0], min_dim[1], max_dim[2]])])) # 0 to min_dim[1]
#    obj_regions.append(np.array([np.array([min_dim[0], max_dim[1], min_dim[2]]), np.array([max_dim[0], 2*np.pi, max_dim[2]])])) # max_dim[1] to 2pi
#    mask = np.array([False for i in range(len(points_polar))])
#    # do not use strict inequality when obj_region is the upper bound
#    for obj_region in obj_regions:
#     r_mask = (obj_region[0,0]<=r)
#     theta_mask = (obj_region[0,1]<=theta)&(obj_region[1,1]>theta)
#     if use_z:
#       z_mask = (obj_region[0,2]<=z)&(obj_region[1,2]>z)
#       mask = mask|((r_mask)&(theta_mask)&(z_mask))
#     else:
#       mask = mask|((r_mask)&(theta_mask))
#   else:
#     ## no need to split 
#     r_mask = (obj_region[0,0]<=r)
#     theta_mask = (obj_region[0,1]<=theta)&(obj_region[1,1]>theta)
#     if use_z:
#       z_mask = (obj_region[0,2]<=z)&(obj_region[1,2]>z)
#       mask = (r_mask)&(theta_mask)&(z_mask)
#     else:
#       mask = ((r_mask)&(theta_mask))

#   return mask, points_polar[mask]



if __name__=="__main__":
    # y_true = [2, 0, 2, 2, 0, 1]
    # y_pred = [0, 0, 2, 2, 0, 2]

    # C = confusion_matrix_1(y_true, y_pred, labels=[0,1,2])
    # print(C)

    # C = confusion_matrix_2(y_true, y_pred, labels=[0,1,2])
    # print(C)

    # filenames = ["epoch_100", "epoch_50", "epoch_2000"]
    # l = [extract_epoch_number(s) for s in filenames]
    # l = sorted(l)

    # print(l)

    # vecs = np.array([[0,1], [1,0], [0,-1], [-1,0], [-1,1]])
    # expected = np.deg2rad(np.array([0,90,180,270,315]))
    # for i, vec in enumerate(vecs):
    #     print(i)
    #     pred = compute_viewing_angle(vec)
    #     print(np.rad2deg(pred))
    #     assert(pred==expected[i])
    xyz = np.array([[598,100,0.3]])
    #xyz = np.array([[0.1,0.2,0.3]])
    xyz_polar = cart2spherical(xyz)
    xyz_rec = spherical2cart(xyz_polar)
    print(xyz_rec)
