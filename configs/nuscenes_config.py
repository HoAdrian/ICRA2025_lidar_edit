import numpy as np
import torch
import os

device = "cuda"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(2021)
np.random.seed(2021)


grid_size = [512, 512, 32]

mode = "spherical" #"polar"
#mode = "polar"

max_bound=[50, 2*np.pi, 3]
min_bound=[0, 0, -5]
if mode=="polar":
    print("config choosing polar mode")
    max_bound=[50, 2*np.pi, 3]
    min_bound=[0, 0, -5]
elif mode=="spherical":
    print("config choosing spherical mode")
    # max_bound=[70, 2*np.pi, 2.45817184] # 140 degree
    # min_bound=[0, 0, 1.37937312] # 79 degree
    # max_bound=[50.635955604688654, 2*np.pi, 2.45817184] # 140 degree
    # min_bound=[0, 0, 1.37937312] # 79 degree
    max_bound=[50.635955604688654, 2*np.pi, np.deg2rad(120+40/31/2)] # 120 degree
    min_bound=[0, 0, np.deg2rad(80-40/31/2)] # 80 degree
else:
    raise Exception("INVALID MODE")


###### VALID SCENE IDXS FOR MASK GIT TRAINING #####
# full dataset
train_valid_scene_idxs_path = os.path.join(".", "train_valid_scene_idxs.pickle")
val_valid_scene_idxs_path = os.path.join(".", "val_valid_scene_idxs.pickle")

# mini
# train_valid_scene_idxs_path = os.path.join(".", "train_valid_scene_idxs_mini.pickle")
# val_valid_scene_idxs_path = os.path.join(".", "val_valid_scene_idxs_mini.pickle")

vqvae_trans_config = {
    "codebook_dim":1024,
    "num_code":1024,
    "dead_limit":256,
    "window_size":8,
    "patch_size":8,
    "patch_embed_dim":512,
    "num_heads": 16,
    "depth":12,
    "beta":0.25
}

maskgit_trans_config = {
    "hidden_dim":512, 
    "depth":24, 
    "num_heads":8
}

######## for range image (these specifications about LiDAR are modified from NuScenes official websites)
half_grid = ((120+40/31/2)-(80-40/31/2))/grid_size[2]/2
fov_up= 90 - (80-40/31/2)# + half_grid#10 # in degree
fov_down= -(120+40/31/2 - 90)# - half_grid #-30 #in degree
max_range=50.635955604688654 #100.0

############### try 1524 codes for spherical
# vqvae_trans_config = {
#     "codebook_dim":1024,
#     "num_code":1524,
#     "dead_limit":256,
#     "window_size":8,
#     "patch_size":8,
#     "patch_embed_dim":512,
#     "num_heads": 16,
#     "depth":12,
#     "beta":0.25
# }

# maskgit_trans_config = {
#     "hidden_dim":512, 
#     "depth":24, 
#     "num_heads":8
# }




# *************** train stats (on mini dataset) ***************
# #r =- 349, #theta = 1481.7673582992852 #z=?

# minmax_r: [6.2820493e-07 1.0528672e+02]
# minmax_z: [-41.96949005  19.41954994]
# min obj r diff: 0.30159785562828567
# min obj theta diff: 0.004240331838859766
# *************** val stats ***************
# minmax_r: [1.95443789e-07 1.04814201e+02]
# minmax_z: [-17.71610641  19.14627457]
# min obj r diff: 0.2809323077904651
# min obj theta diff: 0.004915250254696506

# 
# total scene num: 850
# exist scene num: 255
# v1.0-trainval: train scene(212), val scene(43)
# +++ original num train:  8414
# +++ original num val:  1xxx.0

# *************** train stats (on trainval dataset)***************
# minmax_r: [5.46677148e-09 1.05304642e+02]
# minmax_z: [-47.87574005  19.82554626]
# min obj r diff: 0.09529104036030134
# min obj theta diff: 0.0002110331676963787
# *************** val stats ***************
# minmax_r: [3.17518158e-07 1.05275330e+02]
# minmax_z: [-52.75299835  19.53935242]
# min obj r diff: 0.1804886880936678
# min obj theta diff: 0.00205207835714611
# +++ filtered num train:  7573.0
# +++ filtered num val:  1560.0

# *************** train stats ***************
# num_occupied/num_grid mean: 0.002076540569033785
# *************** val stats ***************
# num_occupied/num_grid mean: 0.0021160105128347137


################# spherical ########################
# *************** train stats ***************
# num_occupied/num_grid mean: 0.001780598525292364
# *************** val stats ***************
# num_occupied/num_grid mean: 0.001792379367498704
# *************** train stats ***************
# minmax_r: [6.29704289e-07 1.05315407e+02]
# minmax_z: [0.30892828 3.14124727]
# min obj r diff: 0.3831761195698178
# min obj theta diff: 0.00041288984058162335
# *************** val stats ***************
# minmax_r: [2.02664751e-07 1.04992676e+02]
# minmax_z: [1.30305099 2.61821651]
# min obj r diff: 0.37969745867435023
# min obj theta diff: 0.0017507719731423446
# +++ filtered num train:  323.0
# +++ filtered num val:  81.0






