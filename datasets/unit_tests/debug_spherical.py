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
  print("+++ test 1 spherical conversion:")
  xyz = np.array([0,-1,-1])[np.newaxis, :]
  xyz_spherical = cart2spherical(xyz)
  print(f"xyz spherical: ", xyz_spherical)
  print(f"xyz spherical theta, phi: ", np.rad2deg(xyz_spherical[:,1:3]))
  rec_xyz = spherical2cart(xyz_spherical)
  print(f"xyz reconstruct: ", rec_xyz)

  #############################################
  print("+++ test 2 merge:")
  
  ######################################################
  print("+++ test 3 merge:")
  
  ######################################################
  print("+++ test 4 merge:")
  