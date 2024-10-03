import copy
import os
import numpy as np
import argparse

os.system(f"pwd")

import sys
sys.path.append("./")
sys.path.append("./datasets")
sys.path.append("./models")
from datasets.data_utils import *
from datasets.dataset import Voxelizer, PolarDataset, collate_fn_BEV
from datasets.dataset_nuscenes import Nuscenes
import torch
import configs.nuscenes_config as config

from models.vqvae_transformers import VQVAETrans
import pickle

import timeit

from sklearn.metrics import confusion_matrix
import configs.nuscenes_config as config


'''
For profiling
'''

def profiling_dataloader(dataloader):
    start_time = timeit.default_timer()

    num_trials = 5
    for trial in range(num_trials):
        for batch_idx, data_tuple in enumerate(dataloader):
            has, no, voxel_label, BEV_label = data_tuple
            grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
            grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
            voxels_mask = voxel_label
            BEV_mask = BEV_label.to(device)

            if batch_idx % 10 == 0:
                print('Progress [{}/{} ({:.0f}%)]'.format(
                    batch_idx * len(voxels_mask), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader)))
                
            torch.cuda.empty_cache()
    
    mid_time = timeit.default_timer()

    time = (mid_time - start_time)/num_trials

    print(f"{time:.2f} seconds" )

    return time
        
def profiling_conf_mat(method, dataloader, method_name):
    start_time = timeit.default_timer()
    num_batch = 0
   
    for batch_idx, data_tuple in enumerate(dataloader):
        num_batch+=1
        has, no, voxel_label, BEV_label = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label
        BEV_mask = BEV_label.to(device)

        if method_name=="np bin" or method_name=="sklearns":
            method(voxels_occupancy_has.reshape(-1).detach().cpu().numpy(), voxels_occupancy_no.reshape(-1).detach().cpu().numpy(), labels=np.arange(2))
        else:
            method(voxels_occupancy_has.reshape(-1).detach().cpu(), voxels_occupancy_no.reshape(-1).detach().cpu(), labels=np.arange(2))

        if batch_idx % 10 == 0:
            print('Progress [{}/{} ({:.0f}%)]'.format(
                batch_idx * len(voxels_mask), len(dataloader.dataset),
                100. * batch_idx / len(dataloader)))
    
    mid_time = timeit.default_timer()

    return (mid_time - start_time)/num_batch

def profiling_auprc(dataloader):
    start_time = timeit.default_timer()
    num_batch = 0
   
    for batch_idx, data_tuple in enumerate(dataloader):
        num_batch+=1
        has, no, voxel_label, BEV_label = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label
        BEV_mask = BEV_label.to(device)

        k = np.random.randint(low=0, high=len(voxels_occupancy_has))
        compute_auprc(voxels_occupancy_has[k].reshape(-1).detach().cpu().numpy(), voxels_occupancy_no[k].reshape(-1).detach().cpu().numpy())
       
        if batch_idx % 10 == 0:
            print('Progress [{}/{} ({:.0f}%)]'.format(
                batch_idx * len(voxels_mask), len(dataloader.dataset),
                100. * batch_idx / len(dataloader)))
    
    mid_time = timeit.default_timer()

    return (mid_time - start_time)/num_batch

def train(dataloader, model, device, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    num_batch = 0
    total_loss = 0
    for batch_idx, data_tuple in enumerate(dataloader):
        num_batch += 1

        has, no, voxel_label, BEV_label = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label
        BEV_mask = BEV_label.to(device)

        voxels_occupancy_has = voxels_occupancy_has.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        loss_has, rec_x_has, rec_x_logit_has, perplexity_has, min_encodings_has, min_encoding_indices_has = model.compute_loss(voxels_occupancy_has)
        loss = loss_has

        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        model.update_reservoir_codebook(min_encodings_has, min_encoding_indices_has)
        #vqvae.update_reservoir_codebook(min_encodings_no, min_encoding_indices_no)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(voxels_mask), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))

    avg_loss = total_loss/num_batch
   

    return avg_loss, num_batch


def profiling_model(dataloader, model, device, optimizer):
    start_time = timeit.default_timer()

    _, num_batch = train(dataloader, model, device, optimizer, epoch=0)
    
    mid_time = timeit.default_timer()

    time_per_batch = (mid_time - start_time)/num_batch

    print(f"{time_per_batch:.2f} seconds" )

    return time_per_batch

def approx_train_time(time_model_per_batch_train, time_model_per_batch_val, time_dataloader_train, time_dataloader_val, num_train, num_val, batch_size, num_epoch=401):
    num_train_batch = num_train/batch_size
    num_val_batch = num_val/batch_size
    return time_model_per_batch_train*(num_train_batch)*num_epoch/3600 + time_model_per_batch_val*(num_val_batch)*num_epoch/3600 \
        #+ time_dataloader_val*num_epoch/3600 + time_dataloader_train*num_epoch/3600


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--trainval_data_path', type=str, help="path to training and validation examples")
    parser.add_argument('--data_version', type=str, help="versioin of the dataset e.g. nuscenes")
    args = parser.parse_args()

    device = torch.device(config.device)
    print("--- device: ", device)

    train_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'train', exhaustive=False, get_stat=False, filter_valid_scene=False)
    val_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'val', exhaustive=False, filter_valid_scene=False)


    voxelizer = Voxelizer(grid_size=config.grid_size, max_bound=config.max_bound, min_bound=config.min_bound)
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, is_test=True, flip_aug=True, rotate_aug=True)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = True, rotate_aug = True)


    print("+++ original num train: ", len(train_pt_dataset))
    

    batch_size = 5
   
    train_dataset_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                    batch_size = batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = True,
                                                    num_workers=4)
    
    val_dataset_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                    batch_size = batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = True,
                                                    num_workers=4)


    ################################# vqvae model and dataloader#########################
    vqvae_config = config.vqvae_trans_config

    window_size=vqvae_config["window_size"]
    patch_size=vqvae_config["patch_size"]
    patch_embed_dim = vqvae_config["patch_embed_dim"]
    num_heads = vqvae_config["num_heads"]
    depth = vqvae_config["depth"]
    codebook_dim = vqvae_config["codebook_dim"]
    num_code = vqvae_config["num_code"]
    beta = vqvae_config["beta"]
    dead_limit = vqvae_config["dead_limit"]


    vqvae = VQVAETrans(
        img_size=voxelizer.grid_size[0:2],
        in_chans=voxelizer.grid_size[2],
        patch_size=patch_size,
        window_size=window_size,
        patch_embed_dim=patch_embed_dim,
        num_heads=num_heads,
        depth=depth,
        codebook_dim=codebook_dim,
        num_code=num_code,
        beta=beta,
        device=device,
        dead_limit=dead_limit
    ).to(device)
    
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=0.0005)

    # time_dataloader_train = profiling_dataloader(train_dataset_loader)
    # time_dataloader_val = profiling_dataloader(val_dataset_loader)

    ## batchsize=2, numworkers=4 => average 22.28 seconds, with empty cache 22.18 seconds
    ## batchsize=5, numworkers=4 => average 22.53 seconds, with empty cache 21.89 seconds

    time_model_per_batch_train = profiling_model(train_dataset_loader, vqvae, device, optimizer)
    time_model_per_batch_val = time_model_per_batch_train #profiling_model(val_dataset_loader, vqvae, device, optimizer)

    

    num_train = len(train_dataset)
    num_val = len(val_dataset)

    #time_dataloader_train, time_dataloader_val, time_model_per_batch_train, time_model_per_batch_val = [21.787292333401275, 5.376376827800414, 0.7701112374618578, 0.8167545882358408]
    print("each result for training: ", [time_model_per_batch_train, time_model_per_batch_val], "seconds")

    T = approx_train_time(time_model_per_batch_train, time_model_per_batch_val, None, None, num_train, num_val, batch_size, num_epoch=250)
    print("approx total training time: ", T, "hours")

    #### mini: 4.373653490934489 hours for 250 epochs,  4.331129864938597 hours without returning point information
    #### train val: 106.98035358405134 hours for 250 epochs

     ####################### confusion matrix ####################
    # time = []
    # conf_method_names = ["sklearn", "torch sparse", "torch bin", "np bin"]
    # conf_mat_methods = [confusion_matrix, confusion_matrix_1, confusion_matrix_2, confusion_matrix_2_numpy]
    # for i, method in enumerate(conf_mat_methods):
    #     t = profiling_conf_mat(method, train_dataset_loader, conf_method_names[i])
    #     time.append(t)

    # print(time, "seconds") #[1.9573713315494163, 0.4422388454567911, 0.2208690351727893, 0.21608269341986333] seconds

    ############################ auprc ##############################
    # time = profiling_auprc(train_dataset_loader)
    # print(time, "seconds") #2.1550845427962844 seconds, single example 1.0494306557037492 seconds