import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import torch

import pyemd
import numpy as np
import concurrent.futures
from functools import partial
from scipy.linalg import toeplitz


'''
Evaluate quality of the generated point clouds
'''

####################################################
############# Chamfer distance ####################
####################################################
def chamfer_distance_numpy(points1, points2):
    """
    Compute the Chamfer Distance between two point clouds.

    Parameters:
    points1 (numpy.ndarray): First point cloud of shape (N, D), where N is the number of points, D is the dimension.
    points2 (numpy.ndarray): Second point cloud of shape (M, D), where M is the number of points, D is the dimension.

    Returns:
    float: Chamfer distance between the two point clouds.
    """
    diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    
    dist1 = np.min(dist, axis=1)
    dist2 = np.min(dist, axis=0)
    chamfer_dist = np.sum(dist1) + np.sum(dist2)
    
    return chamfer_dist

def chamfer_distance_pytorch(points1, points2):
    """
    Compute the Chamfer Distance between two point clouds.

    Parameters:
        points1 (torch.Tensor): First point cloud of shape (N, D), where N is the number of points, D is the dimension.
        points2 (torch.Tensor): Second point cloud of shape (M, D), where M is the number of points, D is the dimension.

    Returns:
        torch.Tensor: Scalar tensor representing the Chamfer Distance between the two point clouds.
    """
    points1 = points1.float()
    points2 = points2.float()
    # points1: (N, D) -> (N, 1, D)
    # points2: (M, D) -> (1, M, D)
    diff = points1.unsqueeze(1) - points2.unsqueeze(0)  # (N, M, D)
    dist_squared = torch.sum(diff ** 2, dim=2)  # (N, M)

    min_dist1, _ = torch.min(dist_squared, dim=1)  # (N,)
    min_dist2, _ = torch.min(dist_squared, dim=0)  # (M,)
    chamfer_dist = torch.mean(min_dist1) + torch.mean(min_dist2)

    return chamfer_dist

######################################################
####################### MMD ########################
########################################################
def point_cloud_to_histogram_square(field_size, bins, point_cloud):
    '''
    Convert point cloud to histogram that is defined as a square of grids spatially
    point_cloud: np array of size (N,3) in cartesian coordinates
    bins: number of bins
    field_size: the length of the square over which we define a histogram 
    '''

    point_cloud_flat = point_cloud[:,0:2] # BEV view in cartesian coordinates

    square_size = field_size / bins # size of each grid cells

    halfway_offset = 0
    if(bins % 2 == 0):
        halfway_offset = (bins / 2) * square_size
    else:
        print('ERROR')
    
    histogram = np.histogramdd(point_cloud_flat, bins=bins, range=([-halfway_offset, halfway_offset], [-halfway_offset, halfway_offset]))[0]

    return histogram

def point_cloud_to_histogram_2D(min_bound, max_bound, grid_size, point_cloud):
    '''
    Convert point cloud to histogram (BEV view) defined by min_bound, max_bound, grid_size
    point_cloud: np array of size (N,3)
    min_bound: (2,), lower bound for each coordinate
    max_bound: (2,), upper bound for each coordinate
    grid_size: (2,), number of bins for each dimension
    '''

    point_cloud_flat = point_cloud[:,0:2] # BEV view
    range_ = ([min_bound[0], max_bound[0]], [min_bound[1], max_bound[1]])
    
    histogram = np.histogramdd(point_cloud_flat, bins=grid_size[:2], range=range_)[0]
    print("len histogram: ", histogram.shape)
    print("num nonzero bins: ", np.sum(histogram!=0))
    print("histogram: ", histogram)
    #histogram[histogram!=0] = 1.0

    return histogram


def gaussian_kernel(x, y, sigma=0.5):  
  '''
  Gaussian kernel (rbf kernel)
  x: (N,)
  y: (M,)
  '''
  support_size = max(len(x), len(y))
  # convert histogram values x and y to float, and make them equal len
  x = x.astype(np.float64)
  y = y.astype(np.float64)
  if len(x) < len(y):
    x = np.hstack((x, [0.0] * (support_size - len(x))))
  elif len(y) < len(x):
    y = np.hstack((y, [0.0] * (support_size - len(y))))
 
  dist = np.linalg.norm(x - y, 2)
  return np.exp(-dist * dist / (2 * sigma * sigma))

def kernel_parallel_unpacked(x, samples2, kernel):
  '''
  kernel between x and each sample in samples2
  '''
  d = 0
  for s2 in samples2:
    d += kernel(x, s2)
  return d

def kernel_parallel_worker(t):
  '''
  wrapper of kernel parallel unpacked
  '''
  return kernel_parallel_unpacked(*t)

def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
  ''' Discrepancy between 2 set of samples computed by summing up the k(x_i, x_j) for each pair of (x_i,x_j) each of which is from samples1 and samples2 respectively'''
  d = 0

  if not is_parallel:
    for s1 in samples1:
      for s2 in samples2:
        d += kernel(s1, s2, *args, **kwargs)
  else:

    with concurrent.futures.ThreadPoolExecutor() as executor:
      for dist in executor.map(kernel_parallel_worker, [
          (s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1
      ]):
        d += dist

  d /= len(samples1) * len(samples2)
  return d

def compute_mmd(samples1, samples2, kernel=gaussian_kernel, is_hist=True, *args, **kwargs):
    ''' 
    Driver method 

    MMD between two set of samples 
        samples is a list of histograms (vectors)
    '''
    # normalize histograms into pmf  
    if is_hist:
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]
    # print('cross: ', disc(samples1, samples2, kernel, *args, **kwargs))
    # print('===============================')
    return disc(samples1, samples1, kernel, *args, **kwargs) + \
            disc(samples2, samples2, kernel, *args, **kwargs) - \
            2 * disc(samples1, samples2, kernel, *args, **kwargs)

#################### estimate standard deviation of Gaussian kernel ################
# Role of Length Scale (
# σ):The length scale 
# σ controls the "width" of the Gaussian kernel. It determines how quickly the similarity between points decreases as their distance increases.
# A small σ means the kernel is sensitive to small distances, so points need to be very close to each other to be considered similar.
# A large σ makes the kernel less sensitive to distance, so even points that are far apart may still be considered similar.

def euclidean_dist(x, y, sigma=180.3):
  '''
  Compute Euclidean distance between to histograms
  x: (N,)
  y: (M, )
  '''
  support_size = max(len(x), len(y))
  # convert histogram values x and y to float, and make them equal len
  x = x.astype(np.float64)
  y = y.astype(np.float64)
  if len(x) < len(y):
    x = np.hstack((x, [0.0] * (support_size - len(x))))
  elif len(y) < len(x):
    y = np.hstack((y, [0.0] * (support_size - len(y))))

  dist = np.linalg.norm(x - y, 2)
  return dist

def calc_sigma(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
  ''' 
  Estimate the standard deviation of gaussian kernel on two sets of samples

  Return the variance and median of euclidean distance
  '''
  d = []

  if not is_parallel:
    for s1 in samples1:
      for s2 in samples2:
        d.append(kernel(s1, s2, *args, **kwargs))
  else:
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for dist in executor.map(kernel_parallel_worker, [
          (s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1
      ]):
        d.append(dist)

  return np.var(d), np.median(d)

def compute_mmd_sigma(samples1, samples2, is_hist=True, *args, **kwargs):
  ''' 
  Driver method
    
        Empirical Sigma Estimate 

  Return the variance and median of euclidean distance
  '''
  # normalize histograms into pmf  
  if is_hist:
    samples1 = [s1 / np.sum(s1) for s1 in samples1]
    samples2 = [s2 / np.sum(s2) for s2 in samples2]

    # print("sam1:", samples1)
    # print("sam2:", samples2)
    
  return calc_sigma(samples1, samples2, euclidean_dist, *args, **kwargs)


#################################################################
####################             EMD ############################
##################################################################
def emd_histogram(x, y, distance_scaling=1.0):
  '''
  compute the earth mover distance between these two distributions
  x: (N,) histogram, representing a probability distribution
  y: (M,) histogram, representing a probability distribution
  Let K = max(N,M)

  https://pypi.org/project/pyemd/0.0.10/
  '''
  support_size = max(len(x), len(y))
  # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toeplitz.html
  # toeplitz matrix is a square matrix where each diagonal is constant
  d_mat = toeplitz(range(support_size)).astype(np.float64) #(K, K), D[i,j] is the distance to move mass from bin i to bin j
  distance_mat = d_mat / distance_scaling 

  # convert histogram values x and y to float, and make them equal len
  x = x.astype(np.float64)
  y = y.astype(np.float64)
  if len(x) < len(y):
    x = np.hstack((x, [0.0] * (support_size - len(x))))
  elif len(y) < len(x):
    y = np.hstack((y, [0.0] * (support_size - len(y))))

  emd = pyemd.emd(x, y, distance_mat)
  return emd

def compute_emd(samples1, samples2, distance_scaling=1.0, is_hist=True):
  '''
  Driver method of emd, return a list of emd distances
  '''
  if is_hist:
    samples1 = [s1 / np.sum(s1) for s1 in samples1]
    samples2 = [s2 / np.sum(s2) for s2 in samples2]
    
  d = []
  assert(len(samples1)==len(samples2))
  for i in range(len(samples1)):
        d.append(emd_histogram(samples1[i], samples2[i], distance_scaling=distance_scaling))

  return d

######################################################
################ Jensen-Shannon Divergence ##########
######################################################

def kl_histogram(p, q):
    '''
    kl divergence between histograms KL(P||Q)
    Assuming p and q are normalized already 
    '''
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    return np.sum(np.where((p != 0)&(q != 0), p * np.log(p / q), 0))

def jsd_histogram(p,q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5*(p+q)
    jsd = 0.5*kl_histogram(p,m) + 0.5*kl_histogram(q,m)
    #print("jsd", jsd)
    # if jsd==0:
    #   print("--p,m", kl_histogram(p,m))
    #   print("--q,m", kl_histogram(q,m))
    return jsd




def compute_jsd(samples1, samples2, is_hist=True):
  '''
  Driver method of emd, return a list of emd distances
  '''
  if is_hist:
    samples1 = [s1 / np.sum(s1) for s1 in samples1]
    samples2 = [s2 / np.sum(s2) for s2 in samples2]
  
  d = []
  assert(len(samples1)==len(samples2))
  for i in range(len(samples1)):
    d.append(jsd_histogram(samples1[i], samples2[i]))
  d = np.array(d)
  return d

def jsd_parallel_unpacked(p,qs):
    p = np.asarray(p, dtype=np.float64)
    qs = [np.asarray(q, dtype=np.float64) for q in qs]
    jsd = 0
    for i, q in enumerate(qs):
      #m = ms[i]

      jsd += jsd_histogram(p,q)
      
    return jsd


def jsd_parallel_worker(t):
  '''
  wrapper of kernel parallel unpacked
  '''
  return jsd_parallel_unpacked(*t)


def parallel_jsd(samples1, samples2):
  ''' jsd between datasets'''
  d = 0

  # for s1 in samples1:
  #    for s2 in samples2:
  #       print("1")
  #       d+=jsd_histogram(s1, s2)

  with concurrent.futures.ThreadPoolExecutor() as executor:
    for dist in executor.map(jsd_parallel_worker, [
        (s1, samples2) for s1 in samples1
    ]):
      d += dist

  d /= (len(samples1) * len(samples2))
  return d

def compute_jsd_between_sets(samples1, samples2, is_hist=True):
  if is_hist:
    samples1 = [s1 / np.sum(s1) for s1 in samples1]
    samples2 = [s2 / np.sum(s2) for s2 in samples2]
  
  assert(len(samples1)==len(samples2))

  return parallel_jsd(samples1, samples2)


def point_cloud_to_range_image(point_cloud, fov_up=10, fov_down=-30, img_width=512, img_height=32, max_range=100.0):
    '''
    point_cloud: shape (N,3)
    fov_up: max angle of elevation in degrees
    fov_down: min angle of elevation in degrees
    '''
    assert(len(point_cloud.shape)==2)
    assert(point_cloud.shape[-1]==3)
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]

    r = np.sqrt(x**2 + y**2 + z**2)
    
    # azimuth (horizontal angle) and elevation (vertical angle)
    r[r==0] = 1e-6
    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z / r)
    assert(np.max(elevation)<fov_up)
    assert(np.min(elevation)>fov_down)

    fov_up_rad = np.radians(fov_up)
    fov_down_rad = np.radians(fov_down)
    assert(fov_down_rad<0)
    
    # Normalize angles to image coordinates
    u = (azimuth + np.pi) / (2*np.pi) * img_width  # Horizontal coordinates
    v = (1-(elevation - fov_down_rad) / (fov_up_rad - fov_down_rad)) * img_height  # Vertical coordinates
    
    # u = (azimuth - np.min(azimuth)) / (np.max(azimuth)-np.min(azimuth)+1e-6) * img_width  # Horizontal coordinates
    # v = (1-(elevation - np.min(elevation)) / (np.max(elevation) - np.min(elevation)+1e-6)) * img_height  # Vertical coordinates

    u = np.clip(np.round(u), 0, img_width - 1).astype(int) #(N,)
    v = np.clip(np.round(v), 0, img_height - 1).astype(int) #(N,)

    range_image = np.full((img_height, img_width), max_range)
  
    # for i in range(len(r)):
    #     if r[i] < range_image[v[i], u[i]]:
    #         range_image[v[i], u[i]] = r[i]
    
    # order in decreasing depth
    order = np.argsort(r)[::-1]
    r = r[order]
    u = u[order]
    v = v[order]
    range_image[v,u] = r
            
    assert(np.any(range_image<max_range))
    
    return range_image
  

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def plot_range_img(img, path, name, vis=False):
  # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 20))  # Adjusted to plot one image
    
    # Plot the range image
    img = np.copy(img)
    # img/=np.max(img)
    im = ax.imshow(img, cmap="viridis")
    ax.set_title('Range image')
    
    # Add a colorbar to indicate the range
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Layout adjustment
    plt.tight_layout()
    
    # Show the image if vis is set to True
    if vis:
        plt.show()

    # Save the figure if path and name are provided
    if path is not None and name is not None:
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/{name}.png")
        print(f"Range image Figure {name}.png saved to {path}")
    plt.close(fig)  # Close the figure after saving to free up memory
    
def compute_ssim(image_list1, image_list2):
  '''
  structural similarity
  '''
  assert(len(image_list1)==len(image_list2))
  ssim_list = []
  full_ssim_list = []
  for i in range(len(image_list2)):
    img1 = image_list1[i]
    img2 = image_list2[i]
    max_data = max([img1.max(), img2.max()])
    min_data = min([img1.min(), img2.min()])
    ssim_value , full_ssim = ssim(img1, img2, data_range=max_data-min_data, full=True)
    ssim_list.append(ssim_value)
    full_ssim_list.append(full_ssim)
  return np.array(ssim_list), full_ssim_list
    