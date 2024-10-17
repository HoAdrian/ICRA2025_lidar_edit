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
from datasets.dataset import Voxelizer, PolarDataset, collate_fn_BEV_intensity
from datasets.dataset_nuscenes import Nuscenes
import torch
import configs.nuscenes_config as config

from models.vqvae_transformers import VQVAETrans
import pickle
import timeit

num_classes = 2

def train(dataloader, model, device, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    num_batch = 0
    total_loss = 0
    total_auprc = 0
    TPs = np.zeros(num_classes,).astype(np.float64)
    FPs = np.zeros(num_classes,).astype(np.float64)
    FNs = np.zeros(num_classes,).astype(np.float64)
    TNs = np.zeros(num_classes,).astype(np.float64)
    for batch_idx, data_tuple in enumerate(dataloader):
        num_batch += 1
        # TODO: remember to put tensors to the correct device

        # grid_ind: list of B items, each item i is a the grid indices of shape (num_points_i, 3)
        # return_points: list of B items, each item i is a pointcloud of shape (num_points_i, 6)
        # voxel_centers: list of B items, each item i is a pointcloud of shape (num_points_i, 3)
        # voxels_occupancy: (B,#r,#theta,#z)
        # voxel_label: (B,#r,#theta,#z)
        # BEV_label: (B,#r,#theta)

        has, no, voxel_label, BEV_label, intensity_grid = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label
        BEV_mask = BEV_label.to(device)

        voxels_occupancy_has = voxels_occupancy_has.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        intensity_grid = intensity_grid.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        loss_has, rec_x_has, rec_x_logit_has, perplexity_has, min_encodings_has, min_encoding_indices_has = vqvae.compute_intensity_loss(voxels_occupancy_has, intensity_grid)
        loss = loss_has

        # compute confusion matrix
        _, _, TPs_, FPs_, FNs_, TNs_ \
        =confusion_matrix_wrapper((intensity_grid>0).long().reshape(-1).detach().cpu().numpy(), (rec_x_has>0).long().reshape(-1).detach().cpu().numpy(), labels=np.arange(num_classes))
        TPs += TPs_
        FPs += FPs_
        FNs += FNs_
        TNs += TNs_

        total_loss += loss.item()

        # pred_probs = rec_x_logit_has.sigmoid().reshape(-1).detach().cpu().numpy()
        # true_labels = voxels_occupancy_has.reshape(-1).detach().cpu().numpy()
        auprc = 0 #compute_auprc(true_labels, pred_probs)
        total_auprc += auprc
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        vqvae.update_reservoir_codebook(min_encodings_has, min_encoding_indices_has)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(voxels_mask), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))
            
        code_util,code_uniformity = vqvae.track_progress()
        # print("code_util:", code_util)
        # print("code_uniformity:", code_uniformity)
        num_rec, num_gt = vqvae.occupancy_ratio((rec_x_has>0).long(), (intensity_grid>0).long())
        ratio = num_rec/num_gt

        


    avg_loss = total_loss/num_batch
    avg_auprc = total_auprc/num_batch

    return avg_loss, TPs, FPs, FNs, TNs, avg_auprc, code_util, code_uniformity, ratio

def test(dataloader, model, device, epoch):
    size = len(dataloader.dataset)
    model.train()
    num_batch = 0
    total_loss = 0
    total_auprc = 0
    TPs = np.zeros(num_classes,).astype(np.float64)
    FPs = np.zeros(num_classes,).astype(np.float64)
    FNs = np.zeros(num_classes,).astype(np.float64)
    TNs = np.zeros(num_classes,).astype(np.float64)
    with torch.no_grad():
        for batch_idx, data_tuple in enumerate(dataloader):
            num_batch += 1
            # TODO: remember to put tensors to the correct device

            # grid_ind: list of B items, each item i is a the grid indices of shape (num_points_i, 3)
            # return_points: list of B items, each item i is a pointcloud of shape (num_points_i, 6)
            # voxel_centers: list of B items, each item i is a pointcloud of shape (num_points_i, 3)
            # voxels_occupancy: (B,#r,#theta,#z)
            # voxel_label: (B,#r,#theta,#z)
            # BEV_label: (B,#r,#theta)

            has, no, voxel_label, BEV_label, intensity_grid = data_tuple
            grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
            grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
            voxels_mask = voxel_label
            BEV_mask = BEV_label.to(device)

            voxels_occupancy_has = voxels_occupancy_has.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
            intensity_grid = intensity_grid.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
            loss_has, rec_x_has, rec_x_logit_has, perplexity_has, min_encodings_has, min_encoding_indices_has = vqvae.compute_intensity_loss(voxels_occupancy_has, intensity_grid)
            loss = loss_has

            total_loss += loss.item()

            # compute confusion matrix
            _, _, TPs_, FPs_, FNs_, TNs_ \
            =confusion_matrix_wrapper((intensity_grid>0).long().reshape(-1).detach().cpu().numpy(), (rec_x_has>0).long().reshape(-1).detach().cpu().numpy(), labels=np.arange(num_classes))
            TPs += TPs_
            FPs += FPs_
            FNs += FNs_
            TNs += TNs_

            # pred_probs = rec_x_logit_has.sigmoid().reshape(-1).detach().cpu().numpy()
            # true_labels = voxels_occupancy_has.reshape(-1).detach().cpu().numpy()
            auprc = 0 #compute_auprc(true_labels, pred_probs)
            total_auprc += auprc

            if batch_idx % 10 == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(voxels_mask), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.item()))
                
            code_util, code_uniformity = vqvae.track_progress()
            # print("code_util:", code_util)
            # print("code_uniformity:", code_uniformity)
            num_rec, num_gt = vqvae.occupancy_ratio((rec_x_has>0).long(), (intensity_grid>0).long())
            ratio = num_rec/num_gt

            
        avg_loss = total_loss/num_batch
        avg_auprc = total_auprc/num_batch

    return avg_loss, TPs, FPs, FNs, TNs, avg_auprc, code_util, code_uniformity, ratio
        


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
    train_dataset=PolarDataset(train_pt_dataset, voxelizer, flip_aug = True, rotate_aug = True, use_intensity_grid=True)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = True, rotate_aug = True, use_intensity_grid=True)

       
    print("+++ original num train: ", len(train_dataset))
    print("+++ original num val: ", len(val_dataset))

    batch_size = 2
   
    train_dataset_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                    batch_size = batch_size,
                                                    collate_fn = collate_fn_BEV_intensity,
                                                    shuffle = True,
                                                    num_workers=4)
    val_dataset_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                    batch_size = batch_size,
                                                    collate_fn = collate_fn_BEV_intensity,
                                                    shuffle = True,
                                                    num_workers=4)
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

    torch.cuda.empty_cache()


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

    optimizer = torch.optim.Adam(vqvae.parameters(), lr=0.0001) #0.0005, 0.0002
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 80, gamma=0.1)

    start_epoch = 0

    if args.resume_epoch != None:
        start_epoch = args.resume_epoch
        checkpoint = torch.load(os.path.join(args.weight_path, f"epoch_{start_epoch}"))
        vqvae.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch+=1
    
   

    start_time = timeit.default_timer() 
    train_losses = []
    test_losses = []
    train_auprcs = []
    test_auprcs = []

    epochs = args.num_epochs
    
    train_TPs = np.zeros((epochs, num_classes))
    train_FPs = np.zeros((epochs, num_classes))
    train_FNs = np.zeros((epochs, num_classes))
    train_TNs = np.zeros((epochs, num_classes))

    test_TPs = np.zeros((epochs, num_classes))
    test_FPs = np.zeros((epochs, num_classes))
    test_FNs = np.zeros((epochs, num_classes))
    test_TNs = np.zeros((epochs, num_classes))

    train_code_util = []
    train_code_uniformity = []
    test_code_util = []
    test_code_uniformity = []
    train_occ_ratio = []
    test_occ_ratio = []

    for epoch in range(start_epoch, start_epoch+args.num_epochs):
        print("######### epoch: ", epoch)
        train_loss, TPs_tr, FPs_tr, FNs_tr, TNs_tr, auprc_tr, code_util_tr, code_uniformity_tr, ratio_tr = train(train_dataset_loader, vqvae, device, optimizer, epoch)
        train_losses.append(train_loss)
        train_auprcs.append(auprc_tr)
        train_code_util.append(code_util_tr)
        train_code_uniformity.append(code_uniformity_tr)
        train_occ_ratio.append(ratio_tr)

        scheduler.step()

        val_loss, TPs_te, FPs_te, FNs_te, TNs_te, auprc_te, code_util_te, code_uniformity_te, ratio_te = test(val_dataset_loader, vqvae, device, epoch)
        test_losses.append(val_loss)
        test_auprcs.append(auprc_te)
        test_code_util.append(code_util_te)
        test_code_uniformity.append(code_uniformity_te)
        test_occ_ratio.append(ratio_te)

        train_TPs[epoch, :] = TPs_tr
        train_FPs[epoch, :] = FPs_tr
        train_FNs[epoch, :] = FNs_tr
        train_TNs[epoch, :] = TNs_tr

        test_TPs[epoch, :] = TPs_te
        test_FPs[epoch, :] = FPs_te
        test_FNs[epoch, :] = FNs_te
        test_TNs[epoch, :] = TNs_te
        
        if epoch%2==0:
            checkpoint = {
                'model_state_dict': vqvae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch
                }
            torch.save(checkpoint, os.path.join(args.weight_path, "epoch_" + str(epoch)))

        if epoch%2==0:
            curr_epoch = start_epoch+len(train_losses)-1

            accuracy_tr, precision_tr, recall_tr, f1_score_tr, specificity_tr, TPR_tr, FPR_tr = compute_perf_metrics(train_TPs[:epoch+1], train_FPs[:epoch+1], train_FNs[:epoch+1], train_TNs[:epoch+1])  
            accuracy_te, precision_te, recall_te, f1_score_te, specificity_te, TPR_te, FPR_te = compute_perf_metrics(test_TPs[:epoch+1], test_FPs[:epoch+1], test_FNs[:epoch+1], test_TNs[:epoch+1])

            print("metric for all classes at last iteration: ")
            print(f"precision: {precision_te[-1, 1]}")
            print(f"recall: {recall_te[-1, 1]}")
            print(f"f1 score: {f1_score_te[-1, 1]}")
            print(f"sepcificity: {specificity_te[-1, 1]}")
            print(f"TPR: {TPR_te[-1, 1]}")
            print(f"FPR: {FPR_te[-1, 1]}")

        if epoch%2==0:
            print(f"So far, you used {(timeit.default_timer() - start_time):.2f} seconds" )

            rows = [[curr_epoch, train_losses[-1], test_losses[-1]]]
            write_csv_rows("./figures/intensity_vqvae/train_val_loss.csv", rows, overwrite=False)

            rows = [[curr_epoch, accuracy_tr[-1,1], accuracy_te[-1,1]]]
            write_csv_rows("./figures/intensity_vqvae/train_val_accuracy.csv", rows, overwrite=False)

            rows = [[curr_epoch, precision_tr[-1,1], precision_te[-1,1]]]
            write_csv_rows("./figures/intensity_vqvae/train_val_precision.csv", rows, overwrite=False)

            rows = [[curr_epoch, recall_tr[-1,1], recall_te[-1,1]]]
            write_csv_rows("./figures/intensity_vqvae/train_val_recall.csv", rows, overwrite=False)

            rows = [[curr_epoch, f1_score_tr[-1,1], f1_score_te[-1,1]]]
            write_csv_rows("./figures/intensity_vqvae/train_val_f1score.csv", rows, overwrite=False)

            rows = [[curr_epoch, specificity_tr[-1,1], specificity_te[-1,1]]]
            write_csv_rows("./figures/intensity_vqvae/train_val_specificity.csv", rows, overwrite=False)

            rows = [[curr_epoch, TPR_tr[-1,1], TPR_te[-1,1]]]
            write_csv_rows("./figures/intensity_vqvae/train_val_TPR.csv", rows, overwrite=False)

            rows = [[curr_epoch, FPR_tr[-1,1], FPR_te[-1,1]]]
            write_csv_rows("./figures/intensity_vqvae/train_val_FPR.csv", rows, overwrite=False)

            rows = [[curr_epoch, train_code_util[-1], test_code_util[-1]]]
            write_csv_rows("./figures/intensity_vqvae/train_val_code_alive.csv", rows, overwrite=False)

            rows = [[curr_epoch, train_code_uniformity[-1], test_code_uniformity[-1]]]
            write_csv_rows("./figures/intensity_vqvae/train_val_code_uniformity.csv", rows, overwrite=False)

            rows = [[curr_epoch, train_occ_ratio[-1], test_occ_ratio[-1]]]
            write_csv_rows("./figures/intensity_vqvae/train_val_occ_ratio.csv", rows, overwrite=False)

    