import pickle
import os
import numpy as np
import time

root1 = "/home/shinghei/Documents/ours/v1.0-trainval"
root2 = "/home/shinghei/lidar_generation/OpenPCDet_minghan/data/nuscenes/v1.0-trainval/ours/v1.0-trainval"
with open(os.path.join(root1, "token2sample.pickle"), 'rb') as f:
   minghan_dict = pickle.load(f)

with open(os.path.join(root2, "token2sample.pickle"), 'rb') as f:
   black_dict = pickle.load(f)

print("minghan dict", len(minghan_dict))
print("black widow dict", len(black_dict))
time.sleep(10)

count = 0
for key in minghan_dict.keys():
   print(count)
   black_dict[key] = minghan_dict[key]
   count+=1

print("black widow dict", len(black_dict))
time.sleep(10)

with open(os.path.join(root2, "token2sample.pickle"), 'wb') as handle:
    pickle.dump(black_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)