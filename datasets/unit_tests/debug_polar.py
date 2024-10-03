import sys
sys.path.append("../")

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
from data_utils_nuscenes import *

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