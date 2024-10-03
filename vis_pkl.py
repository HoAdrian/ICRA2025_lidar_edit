import pickle
import os
import numpy as np

# root = "/home/shinghei/lidar_generation/CenterPoint/data/nuScenes"
# with open(os.path.join(root, "infos_val_01sweeps_withvelo_filter_True.pkl"), 'rb') as f:
#    infos_val_dict = pickle.load(f)

# with open(os.path.join(root, "infos_train_01sweeps_withvelo_filter_True.pkl"), 'rb') as f:
#    infos_train_dict = pickle.load(f)

# with open(os.path.join(root, "dbinfos_train_1sweeps_withvelo.pkl"), 'rb') as f:
#    db_info_dict = pickle.load(f)

# # print("========= infos val dict ========")
# # print(infos_val_dict[0])

# # print("========= infos train dict ========")
# # print(infos_train_dict)

# print("========= db infos dict ========")
# print(db_info_dict)

# pc_path = "/home/shinghei/lidar_generation/mmdetection3d/data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin"
# pc_path = "/home/shinghei/generated_lidar/has_obj/v1.0-mini/lidar_point_clouds/val_6a97481174074729a9d0ffa096eaa498.bin"
# x = np.fromfile(pc_path)
# print(x.reshape(-1,5))