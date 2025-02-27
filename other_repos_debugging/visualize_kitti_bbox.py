import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 1.805793734317190058e+01,-2.015446776583888155e+00,-1.484559818636626005e+00
# 1.796555374991702791e+01 -3.579999520190100704e+00,-1.484559818636626005e+00
# 2.111822019836969844e+01 -3.766157932574975664e+00 -1.484559818636626005e+00
# 2.121060379162457110e+01 -2.201605188968763116e+00 -1.484559818636626005e+00
# 1.805793734317190058e+01 -2.015446776583888155e+00 -7.129091863662595507e-02
# 1.796555374991702791e+01 -3.579999520190100704e+00 -7.129091863662595507e-02
# 2.111822019836969844e+01 -3.766157932574975664e+00 -7.129091863662595507e-02
# 2.121060379162457110e+01 -2.201605188968763116e+00 -7.129091863662595507e-02

box_path = '/home/shinghei/lidar_generation/data_anchor/KITTI/bboxes/frame_0_car_0.txt'
with open(box_path, 'r') as f:
    lines = f.readlines()
corners = []
for line in lines:
    line = line.strip()
    corner =line.split(' ')
    corner = [float(x) for x in corner]
    corners.append(corner)

corners = np.array(corners)
print(corners.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
colors = np.linspace(0, 1, len(corners))
ax.scatter(corners[0,0], corners[0,1], corners[0,2],marker='x', s=80)
scatter = ax.scatter(corners[:,0], corners[:,1], corners[:,2], c=colors, cmap='jet', marker='o', s=50)

cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Color Scale (Mapped to Z-values)')

# Labels
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("kitti bounding box")



# Create a legend with color swatches
legend_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])  # Side legend area
legend_ax.set_xticks([])  # Hide ticks
legend_ax.set_yticks([])

norm = mcolors.Normalize(vmin=min(colors), vmax=max(colors))  
cmap = cm.get_cmap('jet')  
mapped_colors = [cmap(norm(value)) for value in colors]
# Plot color boxes and text labels
for i, (val, color) in enumerate(zip(colors, mapped_colors)):
    legend_ax.add_patch(plt.Rectangle((0, i), 1, 1, color=color))
    legend_ax.text(1.2, i + 0.3, f"{i}", va='center', fontsize=10)

legend_ax.set_ylim(0, len(colors))  # Adjust limits


plt.show()