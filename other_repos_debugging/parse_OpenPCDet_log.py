

import re
import csv
import os

# Path to your log file
root = '/home/shinghei/lidar_generation/OpenPCDet_minghan/output/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext/default'

#log_file_path = f'{root}/output/tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext/default/train_20241101-115709.log'
# log_file_path = f'{root}/output/tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext/default/train_20241101-124052.log'
#log_file_path = f'{root}/output/tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext/default/train_20241107-171801.log'
#log_file_path = f"{root}/output/tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext/default/train_20241108-114649.log"

################## only one log file ################
#og_file_path = f"{root}/output/tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext/default/train_20241114-013903.log"
# log_file_path = '/home/shinghei/lidar_generation/OpenPCDet_minghan/output/tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext/default/train_20241115-163717.log'
# output_csv = './OpenPCDet_loss_data.csv'
# # Regular expression pattern to match timestamp and loss value
# pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}.*?Loss: (\d+\.\d+)'
# # List to store (timestamp, loss) tuples
# loss_data = []
# # Read and parse the log file
# with open(log_file_path, 'r') as file:
#     for i, line in enumerate(file):
#         match = re.search(pattern, line)
#         if match:
#             timestamp = match.group(1)
#             loss = float(match.group(2))
#             loss_data.append((i, loss))

################# Multiple log files ##########################
# log_file_paths = [f"{root}/output/tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext/default/train_20241114-013903.log", \
# '/home/shinghei/lidar_generation/OpenPCDet_minghan/output/tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext/default/train_20241115-163717.log',\
#     "/home/shinghei/lidar_generation/OpenPCDet_minghan/output/tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext/default/train_20241117-193631.log",\
#         "/home/shinghei/lidar_generation/OpenPCDet_minghan/output/tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext/default/train_20241119-123100.log",\
#             "/home/shinghei/lidar_generation/OpenPCDet_minghan/output/tools/cfgs/nuscenes_models/custom_cbgs_voxel0075_voxelnext/default/train_20241120-002704.log"]
log_file_paths = [os.path.join(root,file) for file in os.listdir(root) if file.endswith(".log")]
print(log_file_paths)
output_csv = './loss_data.csv'

# Regular expression pattern to match timestamp and loss value
pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}.*?Loss: (\d+\.\d+)'

# List to store (timestamp, loss) tuples
loss_data = []
i = 0
# Read and parse the log file
for log_file_path in log_file_paths:
    with open(log_file_path, 'r') as file:
        for j, line in enumerate(file):
            match = re.search(pattern, line)
            if match:
                timestamp = match.group(1)
                loss = float(match.group(2))
                loss_data.append((i, loss))
                i+=1

# Option 1: Print the extracted data
print(len(loss_data))

# Option 2: Save to CSV
with open(output_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Timestamp', 'Loss'])  # Header
    csv_writer.writerows(loss_data)

print(f"Loss data saved to {output_csv}")



import csv
import matplotlib.pyplot as plt
from datetime import datetime

# Load data from CSV
timestamps = []
loss_values = []

with open(output_csv, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip header
    for row in csv_reader:
        timestamps.append(float(row[0]))
        loss_values.append(float(row[1]))

# Plot loss over time
plt.figure(figsize=(10, 6))
plt.plot(timestamps[50:][::100], loss_values[50:][::100], label='Loss', color='blue', marker='o', markersize=0.02, linewidth=1)
plt.xlabel('Timestamp')
plt.ylabel('Loss')
plt.title('Loss over Time')
plt.tight_layout()

# Save plot as an image file
plt.savefig('OpenPCDet_loss_over_time_plot.png', format='png', dpi=300)  # Save as PNG with high resolution
