import copy
import os
import numpy as np
import argparse

import sys
sys.path.append("./")
sys.path.append("./datasets")
sys.path.append("./models")
from datasets.data_utils import *
from datasets.dataset import Voxelizer, PolarDataset, collate_fn_BEV
from datasets.dataset_nuscenes import Nuscenes
import torch
from torch import autograd
import configs.nuscenes_config as config

from models.vqvae_transformers import VQVAETrans, MaskGIT
import pickle
import timeit

from torch.utils.tensorboard import SummaryWriter

num_classes = 2

def train(dataloader, model, device, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    num_batch = 0
    total_loss = 0
    total_acc = 0
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
        has, no, voxel_label, BEV_label = data_tuple
        grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
        grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
        voxels_mask = voxel_label
        BEV_mask = BEV_label.to(device)

        voxels_occupancy_has = voxels_occupancy_has.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        voxels_occupancy_no = voxels_occupancy_no.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
        
        #with autograd.detect_anomaly():
        loss, acc, cache = mask_git.compute_loss(voxels_occupancy_no, voxels_occupancy_has, BEV_mask)

        ### evaluate discrepancy between generated occupancy grid and ground truth
        # with torch.no_grad():
        #     upsampled_patch_mask = cache["upsampled_patch_mask"].unsqueeze(1).expand(-1,voxels_occupancy_has.shape[1],-1,-1) #(B,in_chans,H,W)
        #     pred_code_indices_logit = cache["pred_logit"] #(B, total, code_dim)
        #     _, rec_occupancy = mask_git.one_step_decode(pred_code_indices_logit) #(B, in_chans, H, W)

        #     masked_rec_occupancy = (rec_occupancy)[upsampled_patch_mask==1].reshape(-1).detach().cpu().numpy()
        #     masked_GT_occupancy = (voxels_occupancy_has)[upsampled_patch_mask==1].reshape(-1).detach().cpu().numpy()

        #     _, _, TPs_, FPs_, FNs_, TNs_ \
        #     =confusion_matrix_wrapper(masked_GT_occupancy, masked_rec_occupancy, labels=np.arange(num_classes))
        #     TPs += TPs_
        #     FPs += FPs_
        #     FNs += FNs_
        #     TNs += TNs_

        # accuracy, precision, recall, f1_score, specificity, TPR, FPR = compute_perf_metrics(TPs, FPs, FNs, TNs)
        # print(">>>>>>>>")
        # print("metric for all classes at last iteration: ")
        # print(f"precision: {precision[1]}")
        # print(f"recall: {recall[1]}")
        # print(f"f1 score: {f1_score[1]}")
        # print(f"sepcificity: {specificity[1]}")
        # print(f"TPR: {TPR[1]}")
        # print(f"FPR: {FPR[1]}")
        

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 10 == 0:
            print('########### Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(voxels_mask), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))

        total_loss += loss.item()
        total_acc += acc.item()
        

    avg_loss = total_loss/num_batch
    avg_acc = total_acc/num_batch

    return avg_loss, avg_acc, TPs, FPs, FNs, TNs

def test(dataloader, model, device, epoch):
    size = len(dataloader.dataset)
    model.train()
    num_batch = 0
    total_loss = 0
    total_acc = 0
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
            has, no, voxel_label, BEV_label = data_tuple
            grid_ind_has, return_points_has, voxel_centers_has, voxels_occupancy_has = has
            grid_ind_no, return_points_no, voxel_centers_no, voxels_occupancy_no = no
            voxels_mask = voxel_label
            BEV_mask = BEV_label.to(device)

            voxels_occupancy_has = voxels_occupancy_has.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
            voxels_occupancy_no = voxels_occupancy_no.permute(0,3,1,2).to(device).float() #(B, in_chans, H, W)
           
            loss, acc, cache = mask_git.compute_loss(voxels_occupancy_no, voxels_occupancy_has, BEV_mask)
            
            ### evaluate discrepancy between generated occupancy grid and ground truth
            # upsampled_patch_mask = cache["upsampled_patch_mask"].unsqueeze(1).expand(-1,voxels_occupancy_has.shape[1],-1,-1) #(B,in_chans,H,W)
            # pred_code_indices_logit = cache["pred_logit"] #(B, total, code_dim)
            # _, rec_occupancy = mask_git.one_step_decode(pred_code_indices_logit) #(B, in_chans, H, W)

            # masked_rec_occupancy = (rec_occupancy)[upsampled_patch_mask==1].reshape(-1).detach().cpu().numpy()
            # masked_GT_occupancy = (voxels_occupancy_has)[upsampled_patch_mask==1].reshape(-1).detach().cpu().numpy()

            # _, _, TPs_, FPs_, FNs_, TNs_ \
            # =confusion_matrix_wrapper(masked_GT_occupancy, masked_rec_occupancy, labels=np.arange(num_classes))
            # TPs += TPs_
            # FPs += FPs_
            # FNs += FNs_
            # TNs += TNs_

            if batch_idx % 10 == 0:
                print('######## Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(voxels_mask), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.item()))

            total_loss += loss.item()
            total_acc += acc.item()

        avg_loss = total_loss/num_batch
        avg_acc = total_acc/num_batch

    return avg_loss, avg_acc, TPs, FPs, FNs, TNs
        
def tensorboard_logging(writer, log_dict, epoch):
    train_loss = log_dict["train_loss"]
    val_loss = log_dict["val_loss"]
    train_acc = log_dict["train_acc"]
    val_acc = log_dict["val_acc"]
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)
    writer.flush()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--trainval_data_path', type=str, help="path to training and validation examples")
    parser.add_argument('--data_version', type=str, help="versioin of the dataset e.g. nuscenes")
    parser.add_argument('--vqvae_path', type=str, help="path to the trained vqvae's weights of a specific epoch")
    parser.add_argument('--weight_path', type=str, help="path to save your weights")
    parser.add_argument('--num_epochs', type=int, default=400, help="number of epochs")
    parser.add_argument('--resume_epoch', type=int, help="epoch number of the model you want to resume training")
    args = parser.parse_args()

    device = torch.device(config.device)
    print("--- device: ", device)

    os.makedirs(args.weight_path, exist_ok=True)

    writer = SummaryWriter("./figures/maskgit_trans/runs")

    ### don't create object mask, we are randomly masking patches instead
    filter_valid_scene = False
    print("+++ create object occlusion mask?: ", filter_valid_scene)

    ########## use spherical or polar coordinates
    mode = config.mode
    use_z = False
    if mode=="spherical":
        use_z=True
    
    voxelizer = Voxelizer(grid_size=config.grid_size, max_bound=config.max_bound, min_bound=config.min_bound)
    train_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'train', filter_valid_scene=filter_valid_scene, exhaustive=True, voxelizer=voxelizer, use_z=use_z, mode=mode)
    val_pt_dataset = Nuscenes(args.trainval_data_path, version = args.data_version, split = 'val', filter_valid_scene=filter_valid_scene, exhaustive=True, voxelizer=voxelizer, use_z=use_z, mode=mode)

    train_dataset=PolarDataset(train_pt_dataset, voxelizer, flip_aug = True, rotate_aug = True, use_voxel_random_mask=True)
    val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = True, rotate_aug = True, use_voxel_random_mask=True)

    assert(train_dataset.use_random_mask==True)
    assert(val_dataset.use_random_mask==True)
   
   
    print("+++ original num train: ", len(train_dataset))
    print("+++ original num val: ", len(val_dataset))

    ### only get a subset of valid examples
    train_valid_scene_idxs_path = config.train_valid_scene_idxs_path #os.path.join(".", "train_valid_scene_idxs.pickle")
    val_valid_scene_idxs_path = config.val_valid_scene_idxs_path#os.path.join(".", "val_valid_scene_idxs.pickle")

    if filter_valid_scene:
        if os.path.isfile(train_valid_scene_idxs_path):
            print("++++ we found cached train_valid_scene_idxs")
            with open(train_valid_scene_idxs_path, 'rb') as handle:
                train_valid_scene_idxs = pickle.load(handle)
            train_all_idxs = np.arange(len(train_valid_scene_idxs))
            train_pt_dataset = torch.utils.data.Subset(train_pt_dataset, train_all_idxs[train_valid_scene_idxs==1])
            train_dataset=PolarDataset(train_pt_dataset, voxelizer, flip_aug = True, rotate_aug = True)
            print("+++ after filter num train: ", np.sum(train_valid_scene_idxs))

        if os.path.isfile(val_valid_scene_idxs_path):
            print("++++ we found cached val_valid_scene_idxs")
            with open(val_valid_scene_idxs_path, 'rb') as handle:
                val_valid_scene_idxs = pickle.load(handle)
            val_all_idxs = np.arange(len(val_valid_scene_idxs))
            val_pt_dataset = torch.utils.data.Subset(val_pt_dataset, val_all_idxs[val_valid_scene_idxs==1])
            val_dataset=PolarDataset(val_pt_dataset, voxelizer, flip_aug = True, rotate_aug = True)
            print("+++ after filter num val: ", np.sum(val_valid_scene_idxs))

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
    
    vqvae.load_state_dict(torch.load(args.vqvae_path)["model_state_dict"])

    maskgit_config = config.maskgit_trans_config

    mask_git = MaskGIT(vqvae=vqvae, voxelizer=voxelizer, hidden_dim=maskgit_config["hidden_dim"], depth=maskgit_config["depth"], num_heads=maskgit_config["num_heads"]).to(device)
    optimizer = torch.optim.Adam(mask_git.parameters(), lr=1e-4, betas=(0.0, 0.999)) #0.001, 1e-4 ,0.0005, half, train for 10 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 80, gamma=0.1)


    assert(maskgit_config["depth"]==24)
    assert(maskgit_config["num_heads"]==8)


    start_epoch = 0
    if args.resume_epoch != None:
        print("++++ maskgit loading weights ... ")
        start_epoch = args.resume_epoch
        #mask_git.load_state_dict(torch.load(os.path.join(args.weight_path, f"epoch_{start_epoch}")))
        checkpoint = torch.load(os.path.join(args.weight_path, f"epoch_{start_epoch}"))
        mask_git.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch+=1
        
    

    start_time = timeit.default_timer() 
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    epochs = args.num_epochs
    train_TPs = np.zeros((epochs, num_classes))
    train_FPs = np.zeros((epochs, num_classes))
    train_FNs = np.zeros((epochs, num_classes))
    train_TNs = np.zeros((epochs, num_classes))

    test_TPs = np.zeros((epochs, num_classes))
    test_FPs = np.zeros((epochs, num_classes))
    test_FNs = np.zeros((epochs, num_classes))
    test_TNs = np.zeros((epochs, num_classes))

    for epoch in range(start_epoch, start_epoch+args.num_epochs):
        train_loss, train_acc,  TPs_tr, FPs_tr, FNs_tr, TNs_tr = train(train_dataset_loader, mask_git, device, optimizer, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        scheduler.step()

        val_loss, val_acc, TPs_te, FPs_te, FNs_te, TNs_te = test(val_dataset_loader, mask_git, device, epoch)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if filter_valid_scene:
            train_pt_dataset.dataset.valid_scene_idxs[:] = 1
            val_pt_dataset.dataset.valid_scene_idxs[:] = 1
        else:
            train_pt_dataset.valid_scene_idxs[:] = 1
            val_pt_dataset.valid_scene_idxs[:] = 1

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
                'model_state_dict': mask_git.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch
                }
            torch.save(checkpoint, os.path.join(args.weight_path, "epoch_" + str(epoch)))

        if epoch%2==0:
            print(f"#### So far, you used {(timeit.default_timer() - start_time):.2f} seconds" )

            curr_epoch = start_epoch+len(train_losses)-1 #equivalent to epoch

            # accuracy_tr, precision_tr, recall_tr, f1_score_tr, specificity_tr, TPR_tr, FPR_tr = compute_perf_metrics(train_TPs[epoch:epoch+1], train_FPs[epoch:epoch+1], train_FNs[epoch:epoch+1], train_TNs[epoch:epoch+1])  
            # accuracy_te, precision_te, recall_te, f1_score_te, specificity_te, TPR_te, FPR_te = compute_perf_metrics(test_TPs[epoch:epoch+1], test_FPs[epoch:epoch+1], test_FNs[epoch:epoch+1], test_TNs[epoch:epoch+1])

            # print("metric for all classes at last iteration: ")
            # print(f"precision: {precision_te[-1, 1]}")
            # print(f"recall: {recall_te[-1, 1]}")
            # print(f"f1 score: {f1_score_te[-1, 1]}")
            # print(f"sepcificity: {specificity_te[-1, 1]}")
            # print(f"TPR: {TPR_te[-1, 1]}")
            # print(f"FPR: {FPR_te[-1, 1]}")

            # code prediction preformance metrics
            rows = [[curr_epoch, train_losses[-1], val_losses[-1]]]
            write_csv_rows("./figures/maskgit_trans/train_val_loss.csv", rows, overwrite=False)
            rows = [[curr_epoch, train_accs[-1], val_accs[-1]]]
            write_csv_rows("./figures/maskgit_trans/train_val_accuracy.csv", rows, overwrite=False)

            # reconstruction performance metrics
            # rows = [[curr_epoch, precision_tr[-1,1], precision_te[-1,1]]]
            # write_csv_rows("./figures/maskgit_trans/train_val_precision.csv", rows, overwrite=False)

            # rows = [[curr_epoch, recall_tr[-1,1], recall_te[-1,1]]]
            # write_csv_rows("./figures/maskgit_trans/train_val_recall.csv", rows, overwrite=False)

            # rows = [[curr_epoch, f1_score_tr[-1,1], f1_score_te[-1,1]]]
            # write_csv_rows("./figures/maskgit_trans/train_val_f1score.csv", rows, overwrite=False)

            # rows = [[curr_epoch, specificity_tr[-1,1], specificity_te[-1,1]]]
            # write_csv_rows("./figures/maskgit_trans/train_val_specificity.csv", rows, overwrite=False)

            # rows = [[curr_epoch, TPR_tr[-1,1], TPR_te[-1,1]]]
            # write_csv_rows("./figures/maskgit_trans/train_val_TPR.csv", rows, overwrite=False)

            # rows = [[curr_epoch, FPR_tr[-1,1], FPR_te[-1,1]]]
            # write_csv_rows("./figures/maskgit_trans/train_val_FPR.csv", rows, overwrite=False)

            log_dict = {"train_loss":train_loss, "val_loss":val_loss, "train_acc":train_acc, "val_acc":val_acc}
            tensorboard_logging(writer, log_dict, epoch=epoch)

        torch.cuda.empty_cache()

    writer.close()
    xs = np.arange(start_epoch, start_epoch+len(train_losses))
    plot_xy(xs=xs, ys_list=[train_losses, val_losses], labels_list=["train", "val"], title="train val loss", x_label="epoch", y_label="loss", name="train_val_loss", path="./figures/maskgit_trans", vis=False)
    plot_xy(xs=xs, ys_list=[train_accs, val_accs], labels_list=["train", "val"], title="train val accuracy", x_label="epoch", y_label="loss", name="train_val_acc", path="./figures/maskgit_trans", vis=False)