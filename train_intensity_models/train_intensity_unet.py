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
from datasets.dataset import Voxelizer, PolarDataset, collate_fn_range_intensity
from datasets.dataset_nuscenes import Nuscenes
import torch
import configs.nuscenes_config as config

from models.unet import UNet
import pickle
import timeit

num_classes = 2

def train(dataloader, model, device, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    num_batch = 0
    total_loss = 0
    for batch_idx, data_tuple in enumerate(dataloader):
        num_batch += 1
        # TODO: remember to put tensors to the correct device

        # grid_ind: list of B items, each item i is a the grid indices of shape (num_points_i, 3)
        # return_points: list of B items, each item i is a pointcloud of shape (num_points_i, 6)
        # voxel_centers: list of B items, each item i is a pointcloud of shape (num_points_i, 3)
        # voxels_occupancy: (B,#r,#theta,#z)
        # voxel_label: (B,#r,#theta,#z)
        # BEV_label: (B,#r,#theta)

        has, no, voxel_label, BEV_label, range_intensity_data = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label
        BEV_mask = BEV_label.to(device)
        range_image, intensity_image = range_intensity_data
        range_image = range_image.to(device) #(B,H,W)
        intensity_image = intensity_image.to(device) #(B,H,W)

        #voxels_occupancy_has = voxels_occupancy_has.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        range_image = range_image.unsqueeze(1).float()
        intensity_image = intensity_image.unsqueeze(1).float()
        mask = (range_image<config.max_bound[0]+100)
        # print(intensity_image.shape)
        pred_intensity_image = model(range_image)
        loss = torch.mean((pred_intensity_image[mask] - intensity_image[mask])**2)
        

        total_loss += loss.item()
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(voxels_mask), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))

    avg_loss = total_loss/num_batch

    return avg_loss

import time
def test(dataloader, model, device, epoch):
    size = len(dataloader.dataset)
    model.eval()
    num_batch = 0
    total_loss = 0
    avg_l2_error_over_batch = 0
    with torch.no_grad():
        for batch_idx, data_tuple in enumerate(dataloader):
            num_batch += 1
            # TODO: remember to put tensors to the correct device
            has, no, voxel_label, BEV_label, range_intensity_data = data_tuple
            grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
            grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
            voxels_mask = voxel_label
            BEV_mask = BEV_label.to(device)
            range_image, intensity_image = range_intensity_data
            range_image = range_image.to(device)
            intensity_image = intensity_image.to(device)

            range_image = range_image.unsqueeze(1).float()
            intensity_image = intensity_image.unsqueeze(1).float()
            pred_intensity_image = model(range_image)
            mask = (range_image<config.max_bound[0]+100)
            loss = torch.mean((pred_intensity_image[mask] - intensity_image[mask])**2)
        
            #voxels_occupancy_has = voxels_occupancy_has.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
            

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(voxels_mask), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.item()))
                
            
        avg_loss = total_loss/num_batch
    
    return avg_loss 
        


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--trainval_data_path', type=str, help="path to training and validation examples")
    parser.add_argument('--data_version', type=str, help="versioin of the dataset e.g. nuscenes")
    parser.add_argument('--weight_path', type=str, help="path to save your weights")
    parser.add_argument('--num_epochs', type=int, default=400, help="number of epochs")
    parser.add_argument('--resume_epoch', type=int, help="epoch number of the model you want to resume training")
    args = parser.parse_args()

    device = torch.device(config.device)
    print("--- device: ", device)

    os.makedirs(args.weight_path, exist_ok=True)

    ######### use spherical or polar coordinates
    mode = config.mode
    use_z = False
    if mode=="spherical":
        use_z=True

    train_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'train', filter_valid_scene=False, use_z=use_z, mode=mode)
    val_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'val', filter_valid_scene=False, use_z=use_z, mode=mode)

    voxelizer = Voxelizer(grid_size=config.grid_size, max_bound=config.max_bound, min_bound=config.min_bound)
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, flip_aug = True, rotate_aug = True, use_range_proj=True)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = True, rotate_aug = True, use_range_proj=True)

       
    print("+++ original num train: ", len(train_dataset))
    print("+++ original num val: ", len(val_dataset))

    batch_size = 2
   
    train_dataset_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                    batch_size = batch_size,
                                                    collate_fn = collate_fn_range_intensity,
                                                    shuffle = True,
                                                    num_workers=4)
    val_dataset_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                    batch_size = batch_size,
                                                    collate_fn = collate_fn_range_intensity,
                                                    shuffle = True,
                                                    num_workers=4)
    
    model = UNet(in_channels=1, out_channels=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) #0.0005, 0.0002
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 80, gamma=0.1)

    start_epoch = 0

    if args.resume_epoch != None:
        start_epoch = args.resume_epoch
        checkpoint = torch.load(os.path.join(args.weight_path, f"epoch_{start_epoch}"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch+=1
    
   

    start_time = timeit.default_timer() 
    train_losses = []
    test_losses = []

    epochs = args.num_epochs
    
    for epoch in range(start_epoch, start_epoch+args.num_epochs):
        print("######### epoch: ", epoch)
        train_loss = train(train_dataset_loader, model, device, optimizer, epoch)
        train_losses.append(train_loss)
        
        scheduler.step()

        val_loss = test(val_dataset_loader, model, device, epoch)
        test_losses.append(val_loss)
        
        if epoch%2==0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch
                }
            torch.save(checkpoint, os.path.join(args.weight_path, "epoch_" + str(epoch)))

        if epoch%2==0:
            curr_epoch = start_epoch+len(train_losses)-1
            print(f"So far, you used {(timeit.default_timer() - start_time):.2f} seconds" )

            rows = [[curr_epoch, train_losses[-1], test_losses[-1]]]
            write_csv_rows("./figures/intensity_unet/train_val_loss.csv", rows, overwrite=False)
    