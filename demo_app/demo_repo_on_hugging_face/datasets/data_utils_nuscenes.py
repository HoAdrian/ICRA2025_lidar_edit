import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
from pyquaternion import Quaternion
from data_utils import *

'''
Utilities for processing Nuscenes lidar data

Polar coordinates: 
Invariants / Protocols:
1. obj regions (num_boxes, 2, 3) array, each element is [min_dim, max_dim], where min_dim=[min_r, min_theta, min_z] is the lower bounds for coordinates of points in the bounding box, similar for max_dim.
2. Angles (thetas) are within the range [0, 2pi)
3. for each free interval, it is defined by interval[0] to interval[1] in counterclockwise direction
4. An interval crosses quadrant 1 and 4 if (max_theta >=3*np.pi/2) & (min_theta <= np.pi/2)

There are some methods that you can use spherical coordinates instead of polar coordinates. you can set the mode argument as "spherical" instead of "polar". Also read the comments of the methods
'''  

def get_obj_regions(boxes, mode="polar", points_in_boxes=None):
  '''
  boxes: list of Box objects from nuscenes

  return obj regions (num_boxes, 2, 3) array, each element is [min_dim, max_dim], where min_dim=[min_r, min_theta, min_z] is the lower bounds for coordinates of points in the bounding box, similar for max_dim.

  Angles are within the range [0, 2pi)

  Note: If the obj region crosses the positive x-axis, then its max_theta is in the forth quadrant, and its min_theta is in the first quadrant
  '''
  if mode == "polar":
    obj_regions = []
    for i, box in enumerate(boxes):
      corners = box.corners() #(3,8)
      corner_1 = corners[:,0].reshape(1,-1)
      corner_2 = corners[:,1].reshape(1,-1)
      # corner_5 = corners[:,4].reshape(1,-1)
      # corner_6 = corners[:,5].reshape(1,-1)
      corner_7 = corners[:,6].reshape(1,-1)
      corner_8 = corners[:,7].reshape(1,-1)

      corners = np.concatenate((corner_1, corner_2, corner_7, corner_8), axis=0)
      corners_polar = cart2polar(corners)
      max_dim = np.max(corners_polar, axis=0)
      min_dim = np.min(corners_polar, axis=0)

      obj_min, obj_max = min_dim[1], max_dim[1] # min and max theta of object region
      # whether the object region crosses the first and forth quadrants
      obj_cross_bound = obj_max >=3*np.pi/2 and obj_min <= np.pi/2
      if obj_cross_bound:
         thetas = corners_polar[:,1]
         quad_1 = thetas[(thetas<=np.pi/2)]
         quad_4 = thetas[(thetas>=3*np.pi/2)]

         max_dim[1] = np.min(quad_4)
         min_dim[1] = np.max(quad_1)
         
      obj_regions.append([min_dim, max_dim])

    obj_regions = np.array(obj_regions) #(num_boxes, 2, 3)
    return obj_regions
  elif mode == "spherical":
    obj_regions = []
    for i, box in enumerate(boxes):
      corners = box.corners() #(3,8)
      corner_1 = corners[:,0].reshape(1,-1)
      corner_2 = corners[:,1].reshape(1,-1)
      corner_3 = corners[:,2].reshape(1,-1)
      corner_4 = corners[:,3].reshape(1,-1)
      corner_5 = corners[:,4].reshape(1,-1)
      corner_6 = corners[:,5].reshape(1,-1)
      corner_7 = corners[:,6].reshape(1,-1)
      corner_8 = corners[:,7].reshape(1,-1)

      
      # n = 25
      # points_12 = np.linspace(start=corners[:,0], stop=corners[:,1], num=n) #(N,3)
      # points_14 = np.linspace(start=corners[:,0], stop=corners[:,3], num=n)
      # points_15 = np.linspace(start=corners[:,0], stop=corners[:,4], num=n)
      # points_23 = np.linspace(start=corners[:,1], stop=corners[:,2], num=n)
      # points_26 = np.linspace(start=corners[:,1], stop=corners[:,5], num=n)
      # points_56 = np.linspace(start=corners[:,4], stop=corners[:,5], num=n)
      # points_58 = np.linspace(start=corners[:,4], stop=corners[:,7], num=n)
      # points_67 = np.linspace(start=corners[:,5], stop=corners[:,6], num=n)
      # points_78 = np.linspace(start=corners[:,6], stop=corners[:,7], num=n)
      # points_37 = np.linspace(start=corners[:,2], stop=corners[:,6], num=n)
      # points_34 = np.linspace(start=corners[:,2], stop=corners[:,3], num=n)
      # points_48 = np.linspace(start=corners[:,3], stop=corners[:,7], num=n)

      # compare = (corner_1, corner_2, corner_3, corner_4,corner_5, corner_6, corner_7, corner_8, points_12, points_14, points_15, points_23, points_26, points_56, points_58, points_67, points_78, points_37, points_34, points_48)
      compare = (corner_1, corner_2, corner_3, corner_4,corner_5, corner_6, corner_7, corner_8)
      if points_in_boxes is not None:
         point_in_box = points_in_boxes[i]
         compare = (corner_1, corner_2, corner_3, corner_4,corner_5, corner_6, corner_7, corner_8, point_in_box)

      corners = np.concatenate(compare, axis=0)
      corners_polar = cart2polar(corners, mode=mode)
      max_dim = np.max(corners_polar, axis=0)
      min_dim = np.min(corners_polar, axis=0)

      obj_min, obj_max = min_dim[1], max_dim[1] # min and max theta of object region
      # whether the object region crosses the first and forth quadrants
      obj_cross_bound = obj_max >=3*np.pi/2 and obj_min <= np.pi/2
      if obj_cross_bound:
         thetas = corners_polar[:,1]
         quad_1 = thetas[(thetas<=np.pi/2)]
         quad_4 = thetas[(thetas>=3*np.pi/2)]

         max_dim[1] = np.min(quad_4)
         min_dim[1] = np.max(quad_1)

      obj_regions.append([min_dim, max_dim])

    obj_regions = np.array(obj_regions) #(num_boxes, 2, 3)
    return obj_regions
  else:
    raise Exception(f"the mode {mode} is invalid")

     

def find_exact_free_theta_intervals(obj_regions, obj_intervals_raw=None):
  '''
  obj_regions: shape (num_boxes, 2, 3)
    each element is [min_dim, max_dim], where min_dim=[min_r, min_theta, min_z] is the lower bounds 
    for polar coordinates of points in the bounding box, similar for max_dim. Each object region is a rectangular region in polar coordinate space that bounds the object

  Find all intervals along the theta axis in polar coordinates containing no objects (free intervals). Each interval [a,b] covers the region from theta=a to theta=b in counter-clockwise direction. 
  '''
  
  if obj_intervals_raw is None:
    obj_intervals_raw = (obj_regions[:, :, 1]) #(num_boxes, 2)
  else:
     print("^^^^ WARNING: obj_intervals_raw not NONE !!!!!!!!!!!!!!!!!!")

  min_thetas = obj_intervals_raw[:,0]
  max_thetas = obj_intervals_raw[:,1]

  # handle the case when obj region crosses first and forth quadrants
  mask = (max_thetas >=3*np.pi/2) & (min_thetas <= np.pi/2)
  obj_intervals = np.copy(obj_intervals_raw)
  # convert the theta in quad 4 to be negative
  obj_intervals[:,0] = np.where(mask, obj_intervals_raw[:,1]-2*np.pi, obj_intervals_raw[:,0])
  obj_intervals[:,1] = np.where(mask, obj_intervals_raw[:,0], obj_intervals_raw[:,1])

  # merge overlapping intervals
  sorted_intervals = np.array(merge_rad_intervals_array(obj_intervals)) #(something, 2)

  #print(f"!!!!!!!!!!!!!!!! sorted object sintervals some negative: {np.rad2deg(sorted_intervals)}")

  
  # convert all free intervals back to positive
  sorted_intervals[sorted_intervals<0] += 2*np.pi
  #print(f"!!!!!!!!!!!!!!!! sorted object sintervals: {np.rad2deg(sorted_intervals)}")

  # the intervals between obj_intervals are free_intervals
  if len(sorted_intervals)>1:
    # [:-1, max], [1:, min] 
    free_intervals = np.concatenate((sorted_intervals[:-1,1:2], sorted_intervals[1:,0:1]), axis=1)
    #print(f"!!!!!!!!!!!!!!!! free intervals line 139: {np.rad2deg(free_intervals)}")

    last_free_int = np.array([[sorted_intervals[-1,1], sorted_intervals[0,0]]])
    free_intervals = np.concatenate((free_intervals, last_free_int), axis=0)
  else:
    # only one object interval => a large free interval
    free_intervals = np.concatenate((sorted_intervals[0:1,1:2], sorted_intervals[0:1,0:1]), axis=1)

  # convert all free intervals back to positive
  #free_intervals[free_intervals<0] += 2*np.pi

  return free_intervals

def rad_intervals_overlap(interval_1, interval_2, verbose=False):
    '''
    1. Assume interval[0]<interval[1] for each interval. 
    i.e.
    - if the interval crosses quadrant 4 and 1, interval[0]<0 and is in quadrant 4 and interval[1] is in first quadrant and >0
    - otherwise, both interval[0]>0 and interval[1]>0

    2. [interval_1, interval_2] are listed following the order that interval_1[0] <= interval_2[0]

    There are 4 cases of overlapping given the constraints above:
    1. Both interval_1 and interval_2 do not cross quad 1 and 4
    2. Both interval_1 and interval_2 cross quad 1 and 4
    3. interval_1 crosses quad 1 and 4, but interval_2 does not and is not in quad 4
    4. interval_1 crosses quad 1 and 4, but interval_2 does not and is in quad 4
    '''

    min1, max1 = interval_1
    min2, max2 = interval_2
    assert(min1<max1)
    assert(min2<max2)
    if verbose:
       print(f"case 4: ", (min2>=0 and max2>=3*np.pi/2) and (min1<0 and max1>=0))
    if (min2>=0 and max2>=3*np.pi/2) and (min1<0 and max1>=0):
       # case 4
       if verbose:
          print(f"max2: {max2}")
          print(f"min1: {min1+2*np.pi}")
          print(f"min2: {min2}")
       return min1+2*np.pi <=max2 #(min2 <= min1+2*np.pi <=max2)
    else:
      # cases 1,2,3
      return (min1 <= min2 <=max1)


def merge_rad_intervals_array(intervals):
    '''
    Merge overlapping intervals in radian. intervals is np array of shape (num_intervals, 2). 

    1. Assume interval[0]<interval[1] for each interval. 
    i.e.
    - if the interval crosses quadrant 4 and 1, interval[0]<0 and is in quadrant 4 and interval[1] is in first quadrant and >0
    - otherwise, both interval[0]>0 and interval[1]>0

    2. [ interval_1, interval_2, ... ] are listed following the order that interval_{i}[0] <= interval_{i+1}[0]
    
    There are 4 cases of overlapping given the constraints above:
    1. Both interval_1 and interval_2 do not cross quad 1 and 4
    2. Both interval_1 and interval_2 cross quad 1 and 4
    3. interval_1 crosses quad 1 and 4, but interval_2 does not and is not in quad 4
    4. interval_1 crosses quad 1 and 4, but interval_2 does not and is in quad 4
    '''
    # Sort the array on the basis of start values of intervals.
    idxs = np.argsort(intervals[:,0])
    intervals = intervals[idxs]
    # print("sorted: ", intervals)
    
    stack = []
    # insert first interval into stack
    stack.append(intervals[0])
    for i in intervals[1:]:
        # Check for overlapping interval,
        # if interval overlap
        #if (stack[-1][0]) <= i[0] <= stack[-1][-1]:

        if (i[0]>=0 and i[1]>=3*np.pi/2) and (stack[-1][0]<0 and stack[-1][1]>=0):
           # handle case 4 (i in quad 4 and does not cross quand 1 and 4, and top of stack crosses quad 1 and 4) later
           stack.append(i)
        else:
          if rad_intervals_overlap(interval_1=stack[-1], interval_2=i):
              stack[-1][-1] = max(stack[-1][-1], i[-1])
          else:
              stack.append(i)
    #print(f"STACK: {np.rad2deg(np.array(stack))}")
    if len(stack) > 1:
      #print(f"++++++ first interval: {stack[0]}, last interval: {stack[-1]}")
      if stack[0][0] < 0 and stack[0][1] >= 0: # if first interval crosses first and forth quadrants
        #print("STACK CROSSING QUADRANTS")
        if rad_intervals_overlap(stack[0], stack[-1], verbose=False):
          #print("++++ merge last: {}")
          stack[0][0] = min(stack[0][0], stack[-1][0]-2*np.pi)
          stack = stack[:-1]
    #print(f"+++STACK after merge: {np.rad2deg(np.array(stack))}")
    return stack

def plot_lidar_points(points_xyz, boxes, xlim=[-20,20], ylim=[-20,20], vis=True, title="lidar_points"):
    '''
    Plot lidar points and bounding boxes

    points_xyz: shape (N, d'), where each row is (x, y, z, intensity, ...)
    boxes: list of Box objects from nuscenes
    '''
    x = points_xyz[:,0]
    y = points_xyz[:,1]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=points_xyz[:,3], cmap='viridis', alpha=0.75, s=1)

    #plot boxes
    for i, box in enumerate(boxes):
        corners = box.corners() #(3,8)
        corner_1 = corners[:,0][:2]
        corner_2 = corners[:,1][:2]
        corner_5 = corners[:,4][:2]
        corner_6 = corners[:,5][:2]
        rect = patches.Polygon([corner_1, corner_2, corner_6,corner_5], linewidth=1, edgecolor='r', facecolor='none')

        # Get the current axes and plot the polygon patch
        plt.gca().add_patch(rect)

    # Add color bar
    plt.colorbar(scatter, label='label')
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)

    if vis:
      plt.show()

def plot_intervals(intervals, points_xyz, max_radius, boxes, xlim=[-20,20], ylim=[-20,20], vis=True, title="lidar_points"):
    '''
    plot free intervals, bounding boxes and points

    intervals: an array of intervals along the theta axis
    points_xyz: shape (N, d'), where each row is (x, y, z, intensity, ...)
    boxes: list of Box objects from nuscenes
    max_radius: maximum radius to apply the shade (for visualization) for the intervals 
    '''
    x = points_xyz[:,0]
    y = points_xyz[:,1]
    intensity = points_xyz[:,3]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=intensity, cmap='viridis', alpha=0.75, s=1)

    #plot boxes
    for i, box in enumerate(boxes):
        corners = box.corners() #(3,8)
        corner_1 = corners[:,0][:2]
        corner_2 = corners[:,1][:2]
        corner_5 = corners[:,4][:2]
        corner_6 = corners[:,5][:2]
        rect = patches.Polygon([corner_1, corner_2, corner_6,corner_5], linewidth=1, edgecolor='r', facecolor='none')
        # Get the current axes and plot the polygon patch
        plt.gca().add_patch(rect)

    # plot free intervals
    for i, interval in enumerate(intervals):
        theta_1, theta_2 = interval

        # Generate points along the arc
        if theta_1<theta_2:
          theta_range = np.linspace(theta_1, theta_2, 100)
        else:
           # the free interval crosses the positive x axis since theta_1 to theta_2 in counterclockwise
           theta_range = np.linspace(theta_1 - 2*np.pi, theta_2, 100)

        arc_x = max_radius * np.cos(theta_range)
        arc_y = max_radius * np.sin(theta_range)

        # Combine the points to form a closed polygon
        polygon_x = np.concatenate(([0], arc_x, [0]))
        polygon_y = np.concatenate(([0], arc_y, [0]))

        # Fill the area
        plt.fill(polygon_x, polygon_y, color='blue', alpha=0.2)

    # Add color bar
    plt.colorbar(scatter, label='Intensity')
    # you can relax the limit to see the entire lidar map
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)

    if vis:
      plt.show()


def plot_obj_regions(intervals, obj_regions, points_xyz, max_radius, boxes, xlim=[-80,80], ylim=[-80,80], vis=True, title="lidar_points", path=None, name=None, colors=None):
  '''
  Plot free intervals, object regions (rectangles in polar coordinate containing bounding box), points, and bounding boxes

  intervals: an array of intervals along the theta axis
  obj_regions: shape (num_boxes, 2, 3)
    each element is [min_dim, max_dim], where min_dim=[min_r, min_theta, min_z] is the lower bounds 
    for polar coordinates of points in the bounding box, similar for max_dim. Each object region is a rectangular region in polar coordinate space that bounds the object
  points_xyz: shape (N, d'), where each row is (x, y, z, intensity, ...)
  boxes: list of Box objects from nuscenes
  max_radius: maximum radius to apply the shade (for visualization) for the intervals 
  '''
  x = points_xyz[:,0]
  y = points_xyz[:,1]
  if points_xyz.shape[1]>3:
    intensity = points_xyz[:,3]
  else:
    intensity = np.ones((len(points_xyz), ))

  plt.figure(figsize=(8, 6))
  plt.gca().set_aspect('equal')
  scatter = plt.scatter(x, y, c=intensity, cmap='viridis', alpha=0.75, s=1)

  #plot boxes
  for i, box in enumerate(boxes):
    corners = box.corners() #(3,8)
    corner_1 = corners[:,0][:2]
    corner_2 = corners[:,1][:2]
    corner_5 = corners[:,4][:2]
    corner_6 = corners[:,5][:2]
    
    if colors is None:
      rect = patches.Polygon([corner_1, corner_2, corner_6,corner_5], linewidth=1, edgecolor='r', facecolor='none')
    else:
      rect = patches.Polygon([corner_1, corner_2, corner_6,corner_5], linewidth=1, edgecolor=colors[i], facecolor='none')


    # Get the current axes and plot the polygon patch
    plt.gca().add_patch(rect)

  # plot free intervals
  for i, interval in enumerate(intervals):
    theta_1, theta_2 = interval

    # Generate points along the arc
    if theta_1<theta_2:
      theta_range = np.linspace(theta_1, theta_2, 100)
    else:
        # the free interval crosses the positive x axis since theta_1 to theta_2 in counterclockwise
        theta_range = np.linspace(theta_1 - 2*np.pi, theta_2, 100)

    arc_x = max_radius * np.cos(theta_range)
    arc_y = max_radius * np.sin(theta_range)

    # Combine the points to form a closed polygon
    polygon_x = np.concatenate(([0], arc_x, [0]))
    polygon_y = np.concatenate(([0], arc_y, [0]))

    # Fill the area
    plt.fill(polygon_x, polygon_y, color='blue', alpha=0.2)

  # plot obj regions
  for i, obj_region in enumerate(obj_regions):
    min_dim, max_dim = obj_region
    min_r, min_theta,_ = min_dim
    max_r, max_theta,_ = max_dim

    # Generate points for the arc
    if max_theta >=3*np.pi/2 and min_theta <= np.pi/2:
       # if object region crosses the forth and first quadrants
       theta_range = np.linspace(max_theta-2*np.pi, min_theta, 100)
    else:
      theta_range = np.linspace(min_theta, max_theta, 100)

    arc_x = np.concatenate([max_r * np.cos(theta_range), min_r * np.cos(theta_range[::-1])])
    arc_y = np.concatenate([max_r * np.sin(theta_range), min_r * np.sin(theta_range[::-1])])

    # Generate points for the straight sides
    side_x = np.array([min_r, max_r, max_r, min_r]) * np.cos([min_theta, min_theta, max_theta, max_theta])
    side_y = np.array([min_r, max_r, max_r, min_r]) * np.sin([min_theta, min_theta, max_theta, max_theta])

    # Concatenate all boundary points
    boundary_x = np.concatenate([arc_x, side_x])
    boundary_y = np.concatenate([arc_y, side_y])

    # Fill the region
    plt.fill(boundary_x, boundary_y, color='green', alpha=0.2)

  # Add color bar
  plt.colorbar(scatter, label='Intensity')
  # you can relax the limit to see the entire lidar map
  plt.xlim(xlim)
  plt.ylim(ylim)

  # Add labels and title
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.title(title)

  if vis:
    plt.show()

  if path is not None and name is not None:
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/{name}.png")
    print(f"Figure {name}.png saved to {path}")

def get_obj_points_and_masks(boxes, points_xyz, points_polar):
  '''
  points_xyz: (N,d), first three columns are x, y, z
  points_polar: (N,d), first three columns are r, theta, z
  boxes: list of Box object from nuscenes

  return:
  obj_points_polar_list: list of points (polar) that belong to an object for each object
  obj_points_mask_list: list of boolean mask that mask out points that belong to an object for each object
  '''
  obj_points_polar_list = [] # len = num_boxes
  obj_points_mask_list = [] # len = num_boxes
  for i, box in enumerate(boxes):
      mask = points_in_box(box, points_xyz[:,:3].T, wlh_factor = 1.0)
      obj_points = points_polar[mask]
      obj_points_polar_list.append(obj_points)
      obj_points_mask_list.append(mask)
  return obj_points_polar_list, obj_points_mask_list


def rotation_method(theta, rotation_angle):
    '''
    theta: np array containing angles in randian in the range [0,2pi)
    rotation angle: angle to rotate the theta

    Return theta after rotation and mapped back to the range [0,2pi)
    '''
    theta = theta + rotation_angle
    theta[theta<0] += 2*np.pi
    theta = theta%(2*np.pi)
    return theta

def rotate_obj_region(obj_region, rotate_rad):
    '''
    obj_region[:,1] contains the min and max thetas of obj region, 
    rotate object region by an angle in the range [0,2pi)
    '''
    # apply rotation to the min and max thetas of obj region too
    obj_region[:,1] = rotation_method(obj_region[:,1], rotate_rad)
    # assign min and max theta for processed obj region again
    new1, new2 = obj_region[:,1]
    obj_region[0,1] = min([new1, new2])
    obj_region[1,1] = max([new1, new2])
    return obj_region

def flip_obj_region(obj_region, axis, mode="polar"):
    '''
    flip object region about x axis or y axis
    '''
    cart_obj_region = polar2cart(obj_region, mode=mode)
    if axis=="x":
      cart_obj_region[:,0]*=-1
    elif axis=="y":
      cart_obj_region[:,1]*=-1
    else:
       raise Exception("please only choose x, y axis when flipping obj region")
    obj_region = cart2polar(cart_obj_region, mode=mode)
    new1, new2 = obj_region[:,1]
    obj_region[0,1] = min([new1, new2])
    obj_region[1,1] = max([new1, new2])
    return obj_region


def pyquaternion_from_angle(angle):
    '''
    angle in radian, rotation only about z axis (on xy plane)
    '''

    return Quaternion(axis=(0, 0, 1), angle=angle)

def get_obj_mask(obj_region, points_polar, use_z=False):
  '''
  get the mask that masks out points enclosed and occluded by the object, defined by obj_region. By default, we ignore the z bounds of obj_region
  obj_region: shape (2, 3), [[min_r, min_theta, min_z], [max_r, max_theta, max_z]]
  points_polar: shape (N,d), d>=3, polar coordinates
  use_z: whether to compute the mask with the z dimension of the obj rejion as well

  *** points_polar is in spherical coordinates, then use_z indicates whether to compute the mask with the phi dimension of the object
  '''
  min_dim, max_dim = obj_region[0], obj_region[1]
  r = points_polar[:,0]
  theta = points_polar[:,1]
  z = points_polar[:,2]
  obj_regions = []

  if (max_dim[1]>=3*np.pi/2 and min_dim[1]<=np.pi):
   # # split region if it crosses the first and forth quadrants
   obj_regions.append(np.array([np.array([min_dim[0], 0, min_dim[2]]), np.array([max_dim[0], min_dim[1], max_dim[2]])])) # 0 to min_dim[1]
   obj_regions.append(np.array([np.array([min_dim[0], max_dim[1], min_dim[2]]), np.array([max_dim[0], 2*np.pi, max_dim[2]])])) # max_dim[1] to 2pi
   mask = np.array([False for i in range(len(points_polar))])
   # do not use strict inequality when obj_region is the upper bound
   for obj_region in obj_regions:
    r_mask = (obj_region[0,0]<=r)
    theta_mask = (obj_region[0,1]<=theta)&(obj_region[1,1]>theta)
    if use_z:
      z_mask = (obj_region[0,2]<=z)&(obj_region[1,2]>z)
      mask = mask|((r_mask)&(theta_mask)&(z_mask))
    else:
      mask = mask|((r_mask)&(theta_mask))
  else:
    ## no need to split 
    r_mask = (obj_region[0,0]<=r)
    theta_mask = (obj_region[0,1]<=theta)&(obj_region[1,1]>theta)
    if use_z:
      z_mask = (obj_region[0,2]<=z)&(obj_region[1,2]>z)
      mask = (r_mask)&(theta_mask)&(z_mask)
    else:
      mask = ((r_mask)&(theta_mask))

  return mask, points_polar[mask]

def get_obj_mask_occupied(obj_region, points_polar, use_z=False):
  '''
  get the mask that masks out points enclosed (not occluded) by the object, defined by obj_region. By default, we ignore the z bounds of obj_region
  obj_region: shape (2, 3), [[min_r, min_theta, min_z], [max_r, max_theta, max_z]]
  points_polar: shape (N,d), d>=3, polar coordinates
  use_z: whether to compute the mask with the z dimension of the obj rejion as well

  *** points_polar is in spherical coordinates, then use_z indicates whether to compute the mask with the phi dimension of the object
  '''
  min_dim, max_dim = obj_region[0], obj_region[1]
  r = points_polar[:,0]
  theta = points_polar[:,1]
  z = points_polar[:,2]
  obj_regions = []

  if (max_dim[1]>=3*np.pi/2 and min_dim[1]<=np.pi):
   # # split region if it crosses the first and forth quadrants
   obj_regions.append(np.array([np.array([min_dim[0], 0, min_dim[2]]), np.array([max_dim[0], min_dim[1], max_dim[2]])])) # 0 to min_dim[1]
   obj_regions.append(np.array([np.array([min_dim[0], max_dim[1], min_dim[2]]), np.array([max_dim[0], 2*np.pi, max_dim[2]])])) # max_dim[1] to 2pi
   mask = np.array([False for i in range(len(points_polar))])
   # do not use strict inequality when obj_region is the upper bound
   for obj_region in obj_regions:
    r_mask = (obj_region[0,0]<=r<=obj_region[1,0])
    theta_mask = (obj_region[0,1]<=theta)&(obj_region[1,1]>theta)
    if use_z:
      z_mask = (obj_region[0,2]<=z)&(obj_region[1,2]>z)
      mask = mask|((r_mask)&(theta_mask)&(z_mask))
    else:
      mask = mask|((r_mask)&(theta_mask))
  else:
    ## no need to split 
    r_mask = (obj_region[0,0]<=r<=obj_region[1,0])
    theta_mask = (obj_region[0,1]<=theta)&(obj_region[1,1]>theta)
    if use_z:
      z_mask = (obj_region[0,2]<=z)&(obj_region[1,2]>z)
      mask = (r_mask)&(theta_mask)&(z_mask)
    else:
      mask = ((r_mask)&(theta_mask))

  return mask, points_polar[mask]

def create_training_scene(points_polar, points_xyz, box, obj_region, intervals, use_z=False):
  '''

  # for a specified object:
        #     for a random free interval that can fit in the object:
          #     1. Find a position the object should rotate to
          #     2. Rotate its region and bounding box
  
  
  Input:
  -points_polar: shape (N, d'), where each row is (r, theta, z, intensity, ...)
  -points_xyz: shape (N, d'), where each row is (x, y, z, intensity, ...)
  -box: bounding box (Box object from nuscenes) of a specified object
  -obj_region: shape (2, 3)
    [min_dim, max_dim], where min_dim=[min_r, min_theta, min_z] is the lower bounds 
    for polar coordinates of points in the bounding box, similar for max_dim. Each object region is a rectangular region in polar coordinate space that bounds the object
  -intervals: an array of intervals along the theta axis

  *** If points_polar is in spherical coordinates, then use_z indicates whether to compute the mask with the phi dimension of the object. You should set use_z to True in this case. 
          
  return:
  -processed_obj_region: the rotated obj region
  -processed_box: the rotated box
  -new_points_xyz_no_bckgrnd (input to neural network): point cloud in cartesian coordinates with points occluded by the rotated object removed
  -occlude_mask: binary mask that remove points in new_points_xyz_has_bckgrnd occluded by the rotated object (False if occluded)
  '''

  candidate_interval_idxs = []

  min_dim, max_dim = obj_region
  obj_min, obj_max = min_dim[1], max_dim[1] # min and max theta of object region
   # whether the object region crosses the first and forth quadrants
  obj_cross_bound = obj_max >=3*np.pi/2 and obj_min <= np.pi/2
  if obj_cross_bound:
      obj_dist =  obj_min - (obj_max - 2*np.pi)
  else:
      obj_dist = obj_max - obj_min

  for i, interval in enumerate(intervals):
    free_1, free_2 = interval
    
    free_cross_bound = free_1 > free_2

    if free_cross_bound:
        free_dist = free_2 - (free_1 - 2*np.pi)
    else:
        free_dist = free_2 - free_1

    if (obj_dist <= free_dist):
        # if the obj region can fit in the free interval
       candidate_interval_idxs.append(i)

  if len(candidate_interval_idxs)==0:
     #print("NOTE: no candidate interval can fit in the current object region, trying again")
     return None
  
  interval_idx = np.random.choice(np.array(candidate_interval_idxs))
  interval = intervals[interval_idx]
  
  # for interval_idx in candidate_interval_idxs
  free_1, free_2 = interval
  free_cross_bound = free_1 > free_2

  # get the theta that bisects the object region
  if obj_cross_bound:
      mid_theta_obj = ((obj_max - 2*np.pi + obj_min)/2 + 2*np.pi)%(2*np.pi)
  else:
      mid_theta_obj = (obj_max + obj_min)/2

  off = (obj_dist)/2

  # get a random theta in the free interval that the object region should rotate to
  if free_cross_bound:
    low_theta_free = (free_1 - 2*np.pi) + off
    hi_theta_free = free_2 - off
    rand_theta_free = np.random.uniform(low=low_theta_free, high=hi_theta_free)
    if rand_theta_free < 0:
        rand_theta_free += 2*np.pi
  else:
    low_theta_free = (free_1) + off
    hi_theta_free = free_2 - off
    rand_theta_free = np.random.uniform(low=low_theta_free, high=hi_theta_free)
    if rand_theta_free < 0:
        rand_theta_free += 2*np.pi

  theta_diff = mid_theta_obj - rand_theta_free
  rotation = -theta_diff
  #print("rotation: ", rotation)

  # apply rotation to object points
  # mask = points_in_box(box, points_xyz[:,:3].T, wlh_factor = 1.0)
  # obj_points_polar = points_polar[mask]
  # obj_points_polar[:,1] = rotation_method(obj_points_polar[:,1], rotation)

  # apply rotation to the min and max thetas of obj region too
  processed_obj_region = np.copy(np.array(obj_region))
  processed_obj_region[:,1] = rotation_method(processed_obj_region[:,1], rotation)

  # assign min and max theta for processed obj region again
  new1, new2 = processed_obj_region[:,1]
  processed_obj_region[0,1] = min([new1, new2])
  processed_obj_region[1,1] = max([new1, new2])

  # apply rotation to the box
  box = copy.deepcopy(box)
  box.rotate(pyquaternion_from_angle(rotation))

  processed_box = box

  # remove_mask = points_in_box(processed_box, points_xyz[:,:3].T, wlh_factor = 1.0)
  # remove_mask = np.logical_not(remove_mask)
  # # Remove existing points that are in the rotated bounding box
  # new_points_xyz_has_bckgrnd = points_xyz[remove_mask]
  # new_points_polar_has_bckgrnd = points_polar[remove_mask]

  # Remove points in the new point cloud that are occluded by the object 
  occlude_mask, _ = get_obj_mask(processed_obj_region, points_polar, use_z)
  occlude_mask = np.logical_not(occlude_mask)
  new_points_xyz_no_bckgrnd = points_xyz[occlude_mask]
  #new_points_polar_no_bckgrnd = new_points_polar_has_bckgrnd[occlude_mask]

  
  # Get another point cloud with the rotated object and with the background (points occluded by the rotated object) (optional)

  return processed_obj_region, processed_box, new_points_xyz_no_bckgrnd, occlude_mask


def create_training_scene_exhaustive(points_polar, points_xyz, box, obj_region, intervals, use_z=False):
  '''
  (exhaustively): 

  # for a specified object:
        #     for a random free interval that can fit in the object :
          #     1. Find a position the object should rotate to (we loop through all possible position in random order and pick one that is valid)
          #     2. Rotate its region and bounding box
  

  
  Input:
  -points_polar: shape (N, d'), where each row is (r, theta, z, intensity, ...)
  -points_xyz: shape (N, d'), where each row is (x, y, z, intensity, ...)
  -box: bounding box (Box object from nuscenes) of a specified object
  -obj_region: shape (2, 3)
    [min_dim, max_dim], where min_dim=[min_r, min_theta, min_z] is the lower bounds 
    for polar coordinates of points in the bounding box, similar for max_dim. Each object region is a rectangular region in polar coordinate space that bounds the object
  -intervals: an array of intervals along the theta axis

  *** If points_polar is in spherical coordinates, then use_z indicates whether to compute the mask with the phi dimension of the object. You should set use_z to True in this case. 
          
  return:
  -processed_obj_region: the rotated obj region
  -processed_box: the rotated box
  -new_points_xyz_no_bckgrnd (input to neural network): point cloud in cartesian coordinates with points occluded by the rotated object removed
  -occlude_mask: binary mask that remove points in new_points_xyz_has_bckgrnd occluded by the rotated object (False if occluded)
  '''

  candidate_interval_idxs = []

  min_dim, max_dim = obj_region
  obj_min, obj_max = min_dim[1], max_dim[1] # min and max theta of object region
   # whether the object region crosses the first and forth quadrants
  obj_cross_bound = obj_max >=3*np.pi/2 and obj_min <= np.pi/2
  if obj_cross_bound:
      obj_dist =  obj_min - (obj_max - 2*np.pi)
  else:
      obj_dist = obj_max - obj_min


  for i, interval in enumerate(intervals):
    free_1, free_2 = interval
    
     # whether the free interval crosses the first and forth quadrants
    free_cross_bound = free_1 > free_2

    if free_cross_bound:
        free_dist = free_2 - (free_1 - 2*np.pi)
    else:
        free_dist = free_2 - free_1

    if (obj_dist <= free_dist):
        # if the obj region can fit in the free interval
       candidate_interval_idxs.append(i)

  if len(candidate_interval_idxs)==0:
     #print("NOTE: no candidate interval can fit in the current object region, trying again")
     return None
  
  #interval_idx = np.random.choice(candidate_interval_idxs)
  
  candidate_interval_idxs = np.array(candidate_interval_idxs)
  np.random.shuffle(candidate_interval_idxs)

  for interval_idx in candidate_interval_idxs:
    interval = intervals[interval_idx]
    
    # for interval_idx in candidate_interval_idxs
    free_1, free_2 = interval
    free_cross_bound = free_1 > free_2

    # get the theta that bisects the object region
    if obj_cross_bound:
        mid_theta_obj = ((obj_max - 2*np.pi + obj_min)/2 + 2*np.pi)%(2*np.pi)
    else:
        mid_theta_obj = (obj_max + obj_min)/2

    off = (obj_dist)/2

    # get a random theta in the free interval that the object region should rotate to
    if free_cross_bound:
      low_theta_free = (free_1 - 2*np.pi) + off
      hi_theta_free = free_2 - off
      rand_theta_free_list = np.linspace(start=low_theta_free, stop=hi_theta_free, num=100)
      rand_theta_free_list[rand_theta_free_list<0] += 2*np.pi
    else:
      low_theta_free = (free_1) + off
      hi_theta_free = free_2 - off
      rand_theta_free_list = np.linspace(start=low_theta_free, stop=hi_theta_free, num=100)
      rand_theta_free_list[rand_theta_free_list<0] += 2*np.pi

    np.random.shuffle(rand_theta_free_list)
    # loop through sufficiently many rand theta free
    for rand_theta_free in rand_theta_free_list:
      theta_diff = mid_theta_obj - rand_theta_free
      rotation = -theta_diff

      # apply rotation to object points
      # mask = points_in_box(box, points_xyz[:,:3].T, wlh_factor = 1.0)
      # obj_points_polar = points_polar[mask]
      # obj_points_polar[:,1] = rotation_method(obj_points_polar[:,1], rotation)

      # apply rotation to the min and max thetas of obj region too
      processed_obj_region = np.copy(np.array(obj_region))
      processed_obj_region[:,1] = rotation_method(processed_obj_region[:,1], rotation)

      # assign min and max theta for processed obj region again
      new1, new2 = processed_obj_region[:,1]
      processed_obj_region[0,1] = min([new1, new2])
      processed_obj_region[1,1] = max([new1, new2])

      # apply rotation to the box
      box = copy.deepcopy(box)
      box.rotate(pyquaternion_from_angle(rotation))

      processed_box = box

      # Remove points in the new point cloud that are occluded by the object 
      occlude_mask, _ = get_obj_mask(processed_obj_region, points_polar, use_z)
      occlude_mask = np.logical_not(occlude_mask)
      new_points_xyz_no_bckgrnd = points_xyz[occlude_mask]

      if np.sum(np.logical_not(occlude_mask))!=0:
        # there is something occluded
        return processed_obj_region, processed_box, new_points_xyz_no_bckgrnd, occlude_mask


  return processed_obj_region, processed_box, new_points_xyz_no_bckgrnd, occlude_mask

def obj_region_theta_dist(obj_region):
  min_dim, max_dim = obj_region
  obj_min, obj_max = min_dim[1], max_dim[1] # min and max theta of object region
  # print(f"obj_min: ", np.rad2deg(obj_min))
  # print(f"obj_max: ", np.rad2deg(obj_max))
   # whether the object region crosses the first and forth quadrants
  obj_cross_bound = obj_max >=3*np.pi/2 and obj_min <= np.pi/2
  if obj_cross_bound:
      obj_dist =  obj_min - (obj_max - 2*np.pi)
  else:
      obj_dist = obj_max - obj_min
  return obj_dist

def obj_region_is_fit(obj_region, intervals):
  '''
  determine what intervals can the obj_region fit in

  return the indices of the interval that can fit the obj_region
  '''
  
  min_dim, max_dim = obj_region
  obj_min, obj_max = min_dim[1], max_dim[1] # min and max theta of object region
  # print(f"obj_min: ", np.rad2deg(obj_min))
  # print(f"obj_max: ", np.rad2deg(obj_max))
   # whether the object region crosses the first and forth quadrants
  obj_cross_bound = obj_max >=3*np.pi/2 and obj_min <= np.pi/2
  if obj_cross_bound:
      obj_dist =  obj_min - (obj_max - 2*np.pi)
  else:
      obj_dist = obj_max - obj_min

  #print(f"obj dist: {obj_dist}")

  candidate_interval_idxs = []
  for i, interval in enumerate(intervals):
    free_1, free_2 = interval
    
    
     # whether the free interval crosses the first and forth quadrants
    free_cross_bound = free_1 > free_2

    if free_cross_bound:
        free_dist = free_2 - (free_1 - 2*np.pi)
    else:
        free_dist = free_2 - free_1

    if (obj_dist <= free_dist):
        # if the obj region can fit in the free interval
       candidate_interval_idxs.append(i)
      #  print(f"{i}:free_1: ", np.rad2deg(free_1))
      #  print(f"{i}:free_2: ", np.rad2deg(free_2))

  return np.array(candidate_interval_idxs)


if __name__=="__main__":
  ##########################################
  ######### UNIT TEST #####################
  ##########################################
  ###########################################
  print("+++ test 1 merge:")
  obj_intervals = np.array([[0,45], [343, 10], [200, 344]])
  obj_intervals = np.deg2rad(obj_intervals)
  obj_intervals[1][0] -= 2*np.pi

  assert(rad_intervals_overlap(obj_intervals[1], obj_intervals[0]))
  assert(rad_intervals_overlap(obj_intervals[1], obj_intervals[2]))
  assert(not rad_intervals_overlap(obj_intervals[0], obj_intervals[2]))

  merged = np.array(merge_rad_intervals_array(obj_intervals))
  merged[merged<0] += 2*np.pi
  merged = merged/np.pi*180
  assert(np.all(merged==np.array([[200,45]])))

  # free intervals
  obj_intervals = np.array([[0,45], [10, 343], [200, 344]])
  obj_intervals = np.deg2rad(obj_intervals)
  free_intervals = find_exact_free_theta_intervals(obj_regions=None, obj_intervals_raw=obj_intervals)
  free_intervals = free_intervals/np.pi*180
  assert(np.all(free_intervals==np.array([[45,200]])))

  #############################################
  print("+++ test 2 merge:")
  obj_intervals = np.array([[0,45], [343, 10], [90, 120], [110, 260]])
  obj_intervals = np.deg2rad(obj_intervals)
  obj_intervals[1][0] -= 2*np.pi

  assert(rad_intervals_overlap(obj_intervals[1], obj_intervals[0]))
  assert(rad_intervals_overlap(obj_intervals[2], obj_intervals[3]))
  assert(not rad_intervals_overlap(obj_intervals[0], obj_intervals[2]))
  assert(not rad_intervals_overlap(obj_intervals[0], obj_intervals[3]))
  assert(not rad_intervals_overlap(obj_intervals[1], obj_intervals[2]))
  assert(not rad_intervals_overlap(obj_intervals[1], obj_intervals[3]))

  merged = np.array(merge_rad_intervals_array(obj_intervals))
  merged[merged<0] += 2*np.pi
  merged = merged/np.pi*180
  assert(np.all(merged==np.array([[343,45], [90,260]])))

  # free intervals
  obj_intervals = np.array([[0,45], [10, 343], [90, 120], [110, 260]])
  obj_intervals = np.deg2rad(obj_intervals)
  free_intervals = find_exact_free_theta_intervals(obj_regions=None, obj_intervals_raw=obj_intervals)
  free_intervals = free_intervals/np.pi*180
  assert(np.all(free_intervals==np.array([[45, 90], [260, 343]])))

  ######################################################
  print("+++ test 3 merge:")
  obj_intervals = np.array([[0,45], [343, 10], [120, 200], [200, 290]])
  obj_intervals = np.deg2rad(obj_intervals)
  obj_intervals[1][0] -= 2*np.pi

  assert(rad_intervals_overlap(obj_intervals[1], obj_intervals[0]))
  assert(rad_intervals_overlap(obj_intervals[2], obj_intervals[3]))
  assert(not rad_intervals_overlap(obj_intervals[0], obj_intervals[2]))
  assert(not rad_intervals_overlap(obj_intervals[0], obj_intervals[3]))
  assert(not rad_intervals_overlap(obj_intervals[1], obj_intervals[2]))
  assert(not rad_intervals_overlap(obj_intervals[1], obj_intervals[3]))

  merged = np.array(merge_rad_intervals_array(obj_intervals))
  merged[merged<0] += 2*np.pi
  merged = merged/np.pi*180
  assert(np.all(merged==np.array([[343,45], [120,290]])))

  # free intervals
  obj_intervals = np.array([[0,45], [10, 343], [120, 200], [200, 290]])
  obj_intervals = np.deg2rad(obj_intervals)
  free_intervals = find_exact_free_theta_intervals(obj_regions=None, obj_intervals_raw=obj_intervals)
  free_intervals = free_intervals/np.pi*180
  assert(np.all(free_intervals==np.array([[45, 120], [290, 343]])))

  ######################################################
  print("+++ test 4 merge:")
  obj_intervals = np.array([[351, 45], [343, 10], [120, 200], [200, 290]])
  obj_intervals = np.deg2rad(obj_intervals)
  obj_intervals[1][0] -= 2*np.pi
  obj_intervals[0][0] -= 2*np.pi

  assert(rad_intervals_overlap(obj_intervals[1], obj_intervals[0]))
  assert(rad_intervals_overlap(obj_intervals[2], obj_intervals[3]))
  assert(not rad_intervals_overlap(obj_intervals[0], obj_intervals[2]))
  assert(not rad_intervals_overlap(obj_intervals[0], obj_intervals[3]))
  assert(not rad_intervals_overlap(obj_intervals[1], obj_intervals[2]))
  assert(not rad_intervals_overlap(obj_intervals[1], obj_intervals[3]))

  merged = np.array(merge_rad_intervals_array(obj_intervals))
  merged[merged<0] += 2*np.pi
  merged = merged/np.pi*180
  assert(np.all(merged==np.array([[343,45], [120,290]])))

  # free intervals
  obj_intervals = np.array([[45, 351], [10, 343], [120, 200], [200, 290]])
  obj_intervals = np.deg2rad(obj_intervals)
  free_intervals = find_exact_free_theta_intervals(obj_regions=None, obj_intervals_raw=obj_intervals)
  free_intervals = free_intervals/np.pi*180
  assert(np.all(free_intervals==np.array([[45, 120], [290, 343]])))
