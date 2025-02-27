import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from timm.models.swin_transformer import BasicLayer, PatchMerging
from timm.models.vision_transformer import PatchEmbed

from quantizer import VectorQuantizer
from transformers import BidirectionalTransformer, VQEncoderTrans, VQDecoderTrans
import sys
sys.path.append("../datasets")
from data_utils import polar2cart, plot_points_and_voxels
from dataset import Voxelizer
# import cv2
# import open3d as o3d
import pickle
import os

from scipy.cluster.vq import kmeans2

# import torch.backends.cudnn as cudnn
# cudnn.benchmark=False

'''
Two stages:
1. train VQVAE using transformer encoder and decoder
2. train transformer to predict code indices
'''

############## helper methods ##############

def _sample_logistic(shape, out=None):
    '''
    Using inverse transform sampling, 
    sample from the standard logistic distribution, which has the cdf as the logistic function and the inverse of the logistic function is the logit function.

    '''
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return torch.log(U) - torch.log(1 - U)


def _sigmoid_sample(logits, tau=1):
    """
    Implementation of Bernouilli reparametrization based on Maddison et al. 2017
    """
    dims = logits.dim()
    logistic_noise = _sample_logistic(logits.size(), out=logits.data.new())
    y = logits + logistic_noise
    return torch.sigmoid(y / tau)


def gumbel_sigmoid(logits, tau=1, hard=False):
    gumbel_sigmoid_coeff = 1.0
    y_soft = _sigmoid_sample(logits * gumbel_sigmoid_coeff, tau=tau)
    if hard:
        y_hard = torch.where(y_soft > 0.5, torch.ones_like(y_soft), torch.zeros_like(y_soft))
        y = y_hard.data - y_soft.data + y_soft
    else:
        y = y_soft
    return y


def voxels2points(voxelizer, voxels, mode="polar"):
    '''
    - voxelizer: Voxelizer object from dataset.py
    - voxels: binary, shape (B, in_chans, H, W), assume in_chans corresponds to z, H and W corresponds to r and theta. 

    return: 
    - list of numpy array of point cloud in cartesian coordinate (each may have different number of points)
    '''
    B, _, _, _ = voxels.shape
    point_clouds = []
    for b in range(B):
        voxels_b = voxels[b]
        voxels_b = voxels_b.permute(1,2,0) # (H, W, in_chans)
        non_zero_indices = torch.nonzero(voxels_b) #(num_non_zero_voxel, 3)
        ## convert non zero voxels to points
        intervals = torch.tensor(voxelizer.intervals).to(voxels.device).unsqueeze(0) #(1,3)
        min_bound = torch.tensor(voxelizer.min_bound).to(voxels.device).unsqueeze(0) #(1,3)
        xyz_pol = ((non_zero_indices[:, :]+0.5) * intervals) + min_bound # use voxel center coordinate
        xyz_pol = xyz_pol.cpu().detach().numpy()
        xyz = polar2cart(xyz_pol, mode=mode)

        point_clouds.append(xyz)

    return point_clouds #a list of (num_non_zero_voxel_of_the_batch, 3)

def count(idx):
    '''
    idx: any shape

    return indices sorted by their frequency and their sorted frequency
    '''
    unique_elements, counts = torch.unique(idx, return_counts=True)
    sorted_indices = torch.argsort(counts, descending=True)
    sorted_elements = unique_elements[sorted_indices]
    sorted_counts = counts[sorted_indices]
    return sorted_elements, sorted_counts

################### models #################

class VQVAETrans(nn.Module):
    '''
    encoder, quantizer and decoder

    intensity: whether to reconstructs a grid of intensity values
    '''
    def __init__(
        self,
        img_size,
        in_chans=40,
        patch_size=8,
        window_size=8,
        patch_embed_dim=512,
        num_heads=16,
        depth=12,
        codebook_dim=1024,
        num_code=1024,
        beta=0.1,
        device="cpu",
        dead_limit=256,
    ):
        super().__init__()

        self.num_code = num_code
        self.code_dim = codebook_dim
        self.window_size = window_size

        self.quantizer = VectorQuantizer(n_e=num_code, e_dim=codebook_dim, beta=beta, device=device)

        self.encoder = VQEncoderTrans(img_size=img_size, 
                                 patch_size=patch_size, 
                                 window_size=window_size,
                                 in_chans=in_chans,
                                 embed_dim=patch_embed_dim,
                                 num_heads=num_heads,
                                 depth=depth,
                                 codebook_dim=codebook_dim)
        
        self.decoder = VQDecoderTrans(img_size=img_size, 
                                 num_patches=self.encoder.num_patches, 
                                 patch_size=patch_size, 
                                 window_size=window_size, 
                                 in_chans=in_chans, 
                                 embed_dim=patch_embed_dim, 
                                 num_heads=num_heads, 
                                 depth=depth, 
                                 codebook_dim=codebook_dim, 
                                 bias_init=-3)

        self.dead_limit = dead_limit
        self.code_age = torch.zeros((num_code,))#.to(device)
        self.code_usage = torch.zeros((num_code,))#.to(device)
        self.reservoir = torch.zeros(num_code * 10, codebook_dim)#.to(device)
        # self.register_buffer("reservoir", torch.zeros(num_code * 10, codebook_dim))
        # self.register_buffer("code_age", torch.zeros((num_code,)))
        self.num_iter=0

    def encode_quantize(self, x):
        '''
        input:
        - x: shape (B,in_chans,H,W)

        denote total = H/patch_size*W/patch_size

        output:
        - codebook_commitment_loss
        - z_q: (B, codebook_dim, H/patch_size, W/patch_size)
        - perplexity
        - min_encodings: (B*total, num_code)
        - min_encoding_indices: (B*total, 1)
        '''
        B, _, _, _ = x.shape
        
        z_e = self.encoder(x) #(B, total, codebook_dim)

        z_e = z_e.reshape(B, self.encoder.h, self.encoder.w, self.quantizer.e_dim).permute(0,3,1,2)
        codebook_commitment_loss, z_q, perplexity, min_encodings, min_encoding_indices = self.quantizer(z_e)
        # z_q: (B, codebook_dim, H/patch_size, W/patch_size)
        # min_encodings: (B*total, num_code)
        # min_encoding_indices: (B*total, 1)

        return codebook_commitment_loss, z_q, perplexity, min_encodings, min_encoding_indices




    def compute_loss(self, x):
        '''
        x: shape (B,in_chans,H,W)

        compute the loss
        '''
        rec_x_logit, codebook_commitment_loss, perplexity, min_encodings, min_encoding_indices = self.forward(x)
        B,C,H,W = x.shape

        # on average, the occupancy ratio of grid cells is 0.002
        # avg_dataset_occupancy_ratio = 0.002
        # weight = (1-avg_dataset_occupancy_ratio)/avg_dataset_occupancy_ratio*0.01
        with torch.no_grad():
            batch_occupancy_ratio = (torch.sum(x)/(B*C*H*W)).item()
            weight = (1-batch_occupancy_ratio)/batch_occupancy_ratio*0.02#*0.01
            # print("batch_occupancy_ratio: ", batch_occupancy_ratio)
            # print("pos weight: ", weight)
        loss_weight = torch.tensor([weight]).to(x.device).float() # weight for the positive class
        rec_loss = (F.binary_cross_entropy_with_logits(rec_x_logit, x, reduction="none", pos_weight=loss_weight) * 100).mean()

        #rec_loss = (F.binary_cross_entropy_with_logits(rec_x_logit, x, reduction="none") * 100).mean()

        loss = rec_loss + codebook_commitment_loss

        # gumbels = -torch.empty_like(rec_x_logit, memory_format=torch.legacy_contiguous_format).exponential_().log()
        # rec_x_logit += 0.8*gumbels

        rec_x = rec_x_logit.sigmoid()
        rec_x = (rec_x >= 0.5).float()

        # rec_x_logit = rec_x_logit.reshape(B, -1)
        # rec_x = (torch.nn.functional.gumbel_softmax(rec_x_logit, tau=2)>=0.01).float()
        # print(rec_x)
        # rec_x = rec_x.reshape(B, C, H, W)

        return loss, rec_x, rec_x_logit, perplexity, min_encodings, min_encoding_indices

    def encode_decode_no_quantize(self, x):
        '''
        x: shape (B,in_chans,H,W)
        denote total = H/patch_size*W/patch_size

        no quantize

        output:
        - rec_x_logit: (B,in_chans,H,W)
        '''
        B, _, _, _ = x.shape
        
        z_e = self.encoder(x) #(B, total, codebook_dim)
        rec_x_logit = self.decoder(z_e) #(B,in_chans,H,W)

        return rec_x_logit
    
    def compute_intensity_loss(self, x, y):
        '''
        x: shape (B,in_chans,H,W), x should be the ground truth occupancy grid
        y: shape (B,in_chans,H,W), y should be the ground truth intensity grid

        compute the loss of reconstructed intensity (Note: intensity at empty voxel is zero)
        rec x logit is regressed to be between 0 and 255 as intensity value
        '''
        rec_x_logit = self.encode_decode_no_quantize(x)
        B,C,H,W = x.shape
        rec_x_logit = rec_x_logit #.sigmoid() #normalized intensity

        ########### weight for the positive intensity
        # on average, the occupancy ratio of grid cells is 0.002
        # avg_dataset_occupancy_ratio = 0.002
        # weight = (1-avg_dataset_occupancy_ratio)/avg_dataset_occupancy_ratio*0.01
        # with torch.no_grad():
        #     batch_occupancy_ratio = (torch.sum(x)/(B*C*H*W)).item()
        #     weight = (1-batch_occupancy_ratio)/batch_occupancy_ratio*0.02#*0.01
        #     # print("batch_occupancy_ratio: ", batch_occupancy_ratio)
        #     # print("pos weight: ", weight)
        # positive_class_mask = (x>0) # upweigh occupied voxels (not nonzero intensity)
        # negative_class_mask = (x<=0)
        # rec_loss_pos = torch.mean((y[positive_class_mask] - rec_x_logit[positive_class_mask])**2)
        # rec_loss_neg = torch.mean((y[negative_class_mask] - rec_x_logit[negative_class_mask])**2)
        # loss = weight*rec_loss_pos + rec_loss_neg
        # with torch.no_grad():
        #     print(f"########### FFFFFFFFFFF ######### fraction nonzero intensity/totol num cell:  {torch.sum(y>50)/torch.sum(x!=0)}")
        #weight = torch.sum(y>50)/torch.sum(x!=0)
        occupied_logit = rec_x_logit[x!=0]
        occupied_ground_truth_intensity = y[x!=0]
        loss = torch.mean((occupied_ground_truth_intensity+0.1)*((occupied_ground_truth_intensity - occupied_logit)**2))
        
        return loss, rec_x_logit


        
    def forward(self, x):
        '''
        x: shape (B,in_chans,H,W)
        denote total = H/patch_size*W/patch_size

        output:
        - rec_x_logit: (B,in_chans,H,W)
        - codebook_commitment_loss
        - perplexity
        - min_encodings: (B*total, num_code)
        - min_encoding_indices: (B*total, 1)
        '''
        B, _, _, _ = x.shape

        codebook_commitment_loss, z_q, perplexity, min_encodings, min_encoding_indices = self.encode_quantize(x)

        z_q = z_q.permute(0,2,3,1).reshape(B, -1, self.quantizer.e_dim) #(B, total, codebook_dim)
        rec_x_logit = self.decoder(z_q) #(B,in_chans,H,W)

        return rec_x_logit, codebook_commitment_loss, perplexity, min_encodings, min_encoding_indices
    
    def track_progress(self):
        '''
        code_util: percentage of code alive
        code_uniformity: percentage of the top k total code usage count
        '''
        code_util = (self.code_age < self.dead_limit).sum() / self.code_age.numel()
        code_uniformity = self.code_usage.topk(10)[0].sum() / self.code_usage.sum()

        return code_util, code_uniformity
    
    def occupancy_ratio(self, rec_occupancy, gt_occupancy):
        '''
        rec_occupancy: binary tensor of shape (B,C,H,W)
        gt_occupancy: binary tensor of shape (B,C,H,W)

        return the avg number of occupied voxels along the batch dimension for both occupancy grid
        '''
        B = rec_occupancy.shape[0]
        rec_occupancy = rec_occupancy.reshape(B,-1)
        gt_occupancy = gt_occupancy.reshape(B,-1)
        num_rec = torch.sum(rec_occupancy==1, dim=-1).float()
        num_gt = torch.sum(gt_occupancy==1, dim=-1).float()

        num_rec = torch.mean(num_rec)
        num_gt = torch.mean(num_gt)

        return num_rec, num_gt


    def update_reservoir_codebook(self, min_encodings, min_encoding_indices):
        '''
        -min encodings: (B*total, num_code), binary, indicates which code is used for each row
        -min encoding indices: (B*total,1), the corresponding actual code indices
        '''
        self.num_iter += 1

        unused_code_idxs = torch.nonzero(min_encodings==0)[:,1].cpu() #(num_zero,)
        used_code_idxs = torch.nonzero(min_encodings)[:,1].cpu() #(num_nonzero,)

        # update code age: codes that are not used
        self.code_age[unused_code_idxs] += 1 # add 1 to age of code that is not used in some batch
        self.code_age[used_code_idxs] = 0 # reset age to 0 if the code is used in some other batch
        # update code usages: for each time a code is used, add its count by 1
        self.code_usage.index_add_(0, used_code_idxs, torch.ones_like(used_code_idxs, dtype=self.code_usage.dtype))

        ### update reservoir
        z = self.quantizer.embedding(min_encoding_indices.squeeze()) #(B*total, code_dim)
        z_flattened = z.reshape(-1, self.code_dim)
        rp = torch.randperm(z_flattened.size(0))
        num_sample = self.reservoir.shape[0] // 100
        # put num_sample amount of predicted code into the queue in the reservoir
        self.reservoir = torch.cat([self.reservoir[num_sample:], z_flattened[rp[:num_sample]].data.cpu().detach()]) 
        
        #### reinitialize codebook using KMeans
        if ((self.code_age >= self.dead_limit).sum() / self.num_code) > 0.03 and (self.num_iter > 1000):
            self.reinit_codebook()
            self.num_iter = 0

    def reinit_codebook(self):
        '''
        reinitialize codebook using living code and embeddings in the reservoir and 
        '''
        live_code = self.quantizer.embedding.weight[self.code_age < self.dead_limit].data.cpu().detach()
        live_code_num = live_code.shape[0]
        all_z = torch.cat([self.reservoir, live_code]) #(num_live+reservoir_size, code_dim)
        rp = torch.randperm(all_z.shape[0])
        all_z = all_z[rp]

        init = torch.cat(
                [live_code, self.reservoir[torch.randperm(self.reservoir.shape[0])[: (self.num_code - live_code_num)]]]
            )
        print("init kmeans: ", init.shape)
        init = init.data.cpu().numpy()
        print(
            "------- running kmeans!!    num code:", self.num_code, "live code num: ",live_code_num
        )  
        # data driven initialization for the embeddings
        centroid, assignment = kmeans2(
            all_z.cpu().numpy(),
            init,
            minit="matrix",
            iter=50,
        )
        z_dist = (all_z - torch.from_numpy(centroid[assignment]).to(all_z.device)).norm(dim=1).sum().item()
        self.quantizer.embedding.weight.data = torch.from_numpy(centroid).to(self.quantizer.embedding.weight.device)

        print(f"===== num centroid: {len(centroid)}")
        print(f"===== quantizer embedding shape: {self.quantizer.embedding.weight.data.shape}")
        print("------ finish kmeans", z_dist)

        self.code_age.fill_(0)
        self.code_usage.fill_(0)







class MaskGIT(nn.Module):
    '''
    Transformer that generate code indices from the output of a trained VQ quantizer and encoder. 

    '''
    def __init__(self, vqvae: VQVAETrans, voxelizer: Voxelizer, hidden_dim=512, depth=24, num_heads=16, object_free_training=True):
        '''
        if object_free_training==True, then only train on voxels without foreground objects
        otherwise, train on any voxels
        '''
        super().__init__()

        #self.register_buffer('vqvae', vqvae, persistent=False)
        self.voxelizer = voxelizer
        self.vqvae = vqvae
        self.quantizer = self.vqvae.quantizer
        self.encoder = self.vqvae.encoder
        self.decoder = self.vqvae.decoder
        self.patch_size = self.decoder.patch_size
        self.img_size_patched = (self.encoder.h, self.encoder.w)
        self.blank_code = None

        self.num_code = self.vqvae.quantizer.n_e
        self.code_dim = self.vqvae.quantizer.e_dim
        self.transformer =  BidirectionalTransformer(num_class=self.num_code, input_dim=self.code_dim, img_size=(self.encoder.h, self.encoder.w), hidden_dim=hidden_dim, depth=depth, num_heads=num_heads, window_size=self.vqvae.window_size) #window size=8
        self.object_free_training = object_free_training
        
    # def load_state_dict(self, state_dict, strict=True):
    #     # Create a copy of the state_dict to modify
    #     state_dict = state_dict.copy()
    #     # Remove 'vqvae' if it exists
    #     state_dict.pop('vqvae', None)
    #     # Load the state dict into the parent class
    #     super(MaskGIT, self).load_state_dict(state_dict, strict)
   

    def forward(self, x, mask):
        '''
        Input:
        - x: shape (B,in_chans,H,W), binary occupancy voxels
        - mask: binary mask of shape (B, total)

        denote total = H/patch_size*W/patch_size. Replace the masked tokens with mask tokens and pass them to transformer. 

        output:
        - pred_code_indices_logit: (B, total, num_code)

        '''
        B, _, _, _ = x.shape

        # 
        with torch.no_grad():
            _, code, _, _, code_indices = self.vqvae.encode_quantize(x)
            code = code.permute(0,2,3,1).reshape(B, -1, self.code_dim)
        
        #### mask code deterministically ####
        masked_code = self.mask_code_deterministic(code, self.transformer.mask_token, mask) # replace masked code with special mask token

        pred_code_indices_logit = self.transformer(masked_code) #(B, total, num_code)

        return pred_code_indices_logit
    
    def one_step_decode(self, pred_code_indices_logit):
        '''
        Predict the code with maximum likelihood for each patch and decode them back to an image (occupancy grid)

        pred_code_indices_logit: (B, total, code_dim)
        '''
        with torch.no_grad():
            pred_indices = pred_code_indices_logit.max(dim=-1)[1] #(B,total)
            code = self.quantizer.embedding(pred_indices) #(B, total, code_dim)
            rec_lidar_logit = self.decoder(code) #(B,in_chans,H,W)
            rec_binary_voxels = (rec_lidar_logit.sigmoid()>0.5).float() 

        return rec_lidar_logit, rec_binary_voxels

    
    def get_random_patch_mask(self, shape, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        shape: [N, L]. N=batch size, L=sequence length, D=dimension per token

        Outputs:
        - mask: A binary mask of shape [N, L] where 0 indicates the element is kept, and 1 indicates the element is removed, random for each sequence.
        """
        N, L = shape.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=shape.device)  # noise in [0, 1]

        # sort noise for each sample
        # sort the noise and use the indices from argsort to shuffle each sequence of vectors, keep the first subset of each sequence
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1) # can be used to restore the sequence shuffled by indexing with ids_shuffle

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=shape.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask


    def mask_code_deterministic(self, code, mask_token, mask):
        '''
        Inputs:
        - code: The input code sequence of shape [N, L, D].
        - mask_token: mask token (special token to replace the masked features), shape [1, 1, D].
        - mask: binary mask over each sequence, shape [N,L]
        
        Outputs:
        - The final sequence with mask tokens in place of the masked positions.  #and optional conditioning information concatenated.
        '''

        mask = mask.unsqueeze(-1) #(N,L,1)
        x = torch.where(mask==1, mask_token, code)

        return x
    
    def compute_loss(self, x_input, x_GT, voxel_mask):
        '''
        Used for training. 

        Input:
        - x_input: shape (B,in_chans,H,W), binary occupancy voxels of the point cloud with background points
        - x_GT: shape (B,in_chans,H,W), binary occupancy voxels of the point cloud with background points
        - voxel_mask (BEV): shape (B,H,W), 1 if the voxel contains foreground object points, otherwise 0

        denote total = H/patch_size*W/patch_size

        Output:
        - loss: cross entropy loss between the code indices prediction and the ground truth code indices in the masked region
        - acc: accuracy of the prediction in the masked region
        - cache: some prediction results
        '''
        B, _, _, _ = x_input.shape

        # get ground truth
        with torch.no_grad():
            _, code_GT, _, _, code_indices_GT = self.vqvae.encode_quantize(x_GT)
            # code_GT: (B*total, num_code)
            # code_indices_GT: (B*total, 1)

        ### which patch contains object points, assuming voxel_mask gives 1 to voxel containing object points
        patch_mask_obj = self.voxel2patch_label(voxel_mask, patch_size=self.patch_size) #(B, H/patch_size, W/patch_size)
        _, H_p, W_p = patch_mask_obj.shape
        patch_mask_obj = patch_mask_obj.reshape(B, -1)
        ### which patch should be masked
        patch_mask = self.get_random_patch_mask(patch_mask_obj, mask_ratio=self.voxelizer.mask_schedule(np.random.uniform()))
        ### ignore patches containing object points
        if self.object_free_training:
            patch_mask = torch.logical_and(patch_mask==1, patch_mask_obj==0).float()
        else:
            patch_mask = patch_mask.float()

        ### remove masked occupancy voxel
        upsampled_patch_mask = self.patch2voxel_label(patch_mask.reshape(B,H_p,W_p), patch_size=self.patch_size) #(B, H, W)
        masked_grid_idxs = torch.nonzero(upsampled_patch_mask)
        x_input[masked_grid_idxs[:,0], :, masked_grid_idxs[:,1], masked_grid_idxs[:,2]] = 0
        

        #### PREDICTION 
        pred_code_indices_logit = self.forward(x_input, patch_mask) # (B, total, num_code)
        patch_mask = patch_mask.reshape(-1)

        assert (torch.sum(torch.logical_or(code_indices_GT<0, code_indices_GT>=self.num_code))==0)
        
        # logit: (B*total, num_code), target: (B*total,), patch_mask: (B*total,)
        cross_ent = F.cross_entropy(pred_code_indices_logit.flatten(0, 1), code_indices_GT.reshape(-1), reduction="none", label_smoothing=0.1)
        loss = (
            cross_ent * patch_mask
        ).sum() / torch.clamp(patch_mask.sum(),min=1)

        b_total= patch_mask.shape[0]
        B, H, W = voxel_mask.shape

        # print("--voxel mask percentage: ", voxel_mask.sum()/(B*H*W))
        # print("--patch mask percentage: ", patch_mask.sum()/(b_total))
        # print("--upsampled patch mask to percentage: ", upsampled_patch_mask.sum()/(B*H*W))
        # with torch.no_grad():
        #     print("--cross ents: ", cross_ent.sum().item())
        assert(patch_mask.sum()!=0)

        acc = (pred_code_indices_logit.flatten(0, 1).max(dim=-1)[1] == code_indices_GT.reshape(-1))[patch_mask > 0].float().mean()

        cache = {"pred_logit":pred_code_indices_logit, "code_indices_GT":code_indices_GT.reshape(-1), "patch_mask":patch_mask, "upsampled_patch_mask":upsampled_patch_mask}

        return loss, acc, cache

    
    def voxel2patch_label(self, binary_voxel_label, patch_size):
        """
        convert voxel-wise label to patch-wise label. If the patch is associated with a voxel labeled 1, then that patch is labeled 1.
        
        Inputs:
            binary_voxel_label (torch tensor): Binary label array of shape (batch, H, W); label on each "pixel" (bird eye view)
            patch_size (int): Size of each patch (p)
        
        Returns:
            patch_label (torch tensor): Array of shape (batch, H/p, W/p) with binary 1 or 0 label
        """
        B, H, W = binary_voxel_label.shape
        patched_height = H // patch_size
        patched_width = W // patch_size

        # Reshape binary_label to make it easier to iterate over patches
        binary_label_reshaped = binary_voxel_label.reshape(B, patched_height, patch_size, patched_width, patch_size)
        
        # Sum the values within each patch to check if any pixel within the patch is labeled as 1
        patch_label = binary_label_reshaped.sum(dim=(2, 4)) > 0
        
        return patch_label.long()
    
    def patch2voxel_label(self, binary_patch_label, patch_size):
        """
        convert voxel-wise label to patch-wise label. If the patch is associated with a voxel labeled 1, then that patch is labeled 1.
        
        Inputs:
            binary_patch_label (torch tensor): Binary label array of shape (batch, H/p, W/p); label on each "pixel" (bird eye view)
            patch_size (int): Size of each patch (p)
        
        Returns:
            voxel_label (torch tensor): Array of shape (batch, H, W) with binary 1 or 0 label
        """
        B, H_p, W_p = binary_patch_label.shape
        H = H_p*patch_size
        W = W_p*patch_size

        # Reshape back to voxel label
        binary_label_reshaped = binary_patch_label.reshape(B, H_p, 1, W_p, 1)
        binary_label_reshaped = binary_label_reshaped.expand(-1,-1,patch_size,-1,patch_size)
        binary_label_reshaped = binary_label_reshaped.reshape(B, H, W)
        
        return binary_label_reshaped.long()
    
    def one_step_predict(self, x_input, x_GT, voxel_mask):
        '''
        For prediction at test time. 

        Input:
        - x_input: shape (B,in_chans,H,W), binary occupancy voxels of the point cloud with background points
        - x_GT: shape (B,in_chans,H,W), binary occupancy voxels of the point cloud with background points
        - voxel_mask (BEV): shape (B,H,W), 1 if the voxel is the region of interest (region occluded by object), otherwise 0

        denote total = H/patch_size*W/patch_size

        Output:
        - loss: cross entropy loss between the code indices prediction and the ground truth code indices in the masked region
        - acc: accuracy of the prediction in the masked region
        - cache: some prediction results
        '''
        B, _, _, _ = x_input.shape

        # get ground truth
        with torch.no_grad():
            _, code_GT, _, _, code_indices_GT = self.vqvae.encode_quantize(x_GT)
            # code_GT: (B*total, num_code)
            # code_indices_GT: (B*total, 1)

        ### which patch is occluded by object
        patch_mask_obj = self.voxel2patch_label(voxel_mask, patch_size=self.patch_size) #(B, H/patch_size, W/patch_size)
        _, H_p, W_p = patch_mask_obj.shape
        patch_mask = patch_mask_obj.reshape(B, -1)

        ### remove masked occupancy voxel
        upsampled_patch_mask = self.patch2voxel_label(patch_mask.reshape(B,H_p,W_p), patch_size=self.patch_size) #(B, H, W)
        masked_grid_idxs = torch.nonzero(upsampled_patch_mask)
        x_input[masked_grid_idxs[:,0], :, masked_grid_idxs[:,1], masked_grid_idxs[:,2]] = 0
        

        #### PREDICTION 
        pred_code_indices_logit = self.forward(x_input, patch_mask) # (B, total, num_code)
        patch_mask = patch_mask.reshape(-1)

        assert (torch.sum(torch.logical_or(code_indices_GT<0, code_indices_GT>=self.num_code))==0)
        
        # logit: (B*total, num_code), target: (B*total,), patch_mask: (B*total,)
        cross_ent = F.cross_entropy(pred_code_indices_logit.flatten(0, 1), code_indices_GT.reshape(-1), reduction="none", label_smoothing=0.1)
        loss = (
            cross_ent * patch_mask
        ).sum() / torch.clamp(patch_mask.sum(),min=1)

        b_total= patch_mask.shape[0]
        B, H, W = voxel_mask.shape

        print("--voxel mask percentage: ", voxel_mask.sum()/(B*H*W))
        print("--patch mask percentage: ", patch_mask.sum()/(b_total))
        print("--upsampled patch mask to percentage: ", upsampled_patch_mask.sum()/(B*H*W))
        with torch.no_grad():
            print("--cross ents: ", cross_ent.sum().item())
        assert(patch_mask.sum()!=0)

        acc = (pred_code_indices_logit.flatten(0, 1).max(dim=-1)[1] == code_indices_GT.reshape(-1))[patch_mask > 0].float().mean()

        cache = {"pred_logit":pred_code_indices_logit, "code_indices_GT":code_indices_GT.reshape(-1), "patch_mask":patch_mask, "upsampled_patch_mask":upsampled_patch_mask}

        return loss, acc, cache

    def conditional_generation(self, x, voxel_mask, T=10, verbose=True):
        '''
        Iteratively sample codes in the masked region, keep the confident ones and repeat

        Input:
        - x: shape (B,in_chans,H,W), binary occupancy voxels of the point cloud with background points
        - voxel_mask: shape (B,H,W), 1 if the voxel is the region of interest (region occluded by object), otherwise 0
        - T = the number of iterations the generation process takes

        denote total = H/patch_size*W/patch_size

        Output:
        - xyzs_list: list of point clouds, each point cloud corresponds to a reconstructed voxel grid
        - rec_lidar_logit: (B,in_chans,H,W)
        - rec_binary_voxels: (B,in_chans,H,W)
        '''
        B, _, _, _ = x.shape
        total = self.img_size_patched[0]*self.img_size_patched[1]

        ##### keeps track of which code is masked
        code_idx = torch.ones((B, total), dtype=torch.int64, device=x.device) * -1
        
        #### Initialize patch mask and occupancy voxels
        patch_mask = self.voxel2patch_label(voxel_mask, patch_size=self.patch_size) #(B, H/patch_size, W/patch_size)
        ### remove masked occupancy voxel (in place)
        upsampled_patch_mask = self.patch2voxel_label(patch_mask, patch_size=self.patch_size) #(B, H, W)
        #upsampled_patch_mask = voxel_mask
        masked_grid_idxs = torch.nonzero(upsampled_patch_mask)
        x[masked_grid_idxs[:,0], :, masked_grid_idxs[:,1], masked_grid_idxs[:,2]] = 0
        patch_mask = patch_mask.reshape(B, -1) #(B,total)

        ### Get the codes for unmasked region
        _, code, _, _, code_indices = self.vqvae.encode_quantize(x)
        code_indices = code_indices.squeeze().reshape(B, -1)
        code_idx[patch_mask==0] = code_indices[patch_mask==0]


        code = code.permute(0,2,3,1).reshape(B, -1, self.code_dim) #(B, total, code_dim)
        # replace masked code with special mask token
        code = self.mask_code_deterministic(code, self.transformer.mask_token, patch_mask) #(B, total, code_dim)

        num_unknown_code = (code_idx == -1).sum(dim=-1) #(B,)

        choice_temperature = 2.0

        for t in range(T):
            pred_logit = self.transformer(code) # (B, total, num_code)

            if t < 10:
                # suppress blank code generation at early stage
                pred_logit[..., self.blank_code] = -10000

            #pred_idx = torch.argmax(pred_logit, dim=-1).type(torch.int64) # (B, total)
            sample_ids = torch.distributions.Categorical(logits=pred_logit).sample() #(B, total)
            prob = torch.softmax(pred_logit, dim=-1) # (B, total, num_code)
            prob = torch.gather(prob, dim=-1, index=sample_ids.unsqueeze(-1)) # (B, total, 1), each element is the probability corresponding to each sampled id
            prob = prob[..., 0] # (B, total)

            # keep the ones that are not masked
            sample_ids[code_idx != -1] = code_idx[code_idx != -1]
            prob[code_idx != -1] = 1e10

            # mask ratio large to small throughout iterations
            # ratio small to large throughout iterations
            ratio = 1.0 * (t + 1) / T
            mask_ratio = self.voxelizer.mask_schedule(ratio)
            mask_len = num_unknown_code * mask_ratio
            mask_len = torch.minimum(mask_len, num_unknown_code) #num_unknown_code- 1
            mask_len = mask_len.clamp(min=0).long() #min=1

            if verbose:
                print(f"##### t:{t}/{T}")
                print("mask_len: ", mask_len)
                
            
            temperature = choice_temperature* (1.0 - ratio)
            gumbels = -torch.empty_like(prob, memory_format=torch.legacy_contiguous_format).exponential_().log()
            # see gumbel softmax paper: https://arxiv.org/pdf/1611.01144, we add gumbel noise to the log probability
            confidence = prob.log() + temperature * gumbels #(B, total)

             # torch.sort(confidence, dim=-1)[0] is confidence sorted along each row
             # cutoff is the confidence of the most confident sample in the top (mask_len+1) confident samples
            cutoff = torch.sort(confidence, dim=-1)[0][
                torch.arange(mask_len.shape[0], device=mask_len.device), mask_len
            ].unsqueeze(1) #(B, 1)

            

            mask = confidence < cutoff #(B, total)
            mask[patch_mask==0] = False # keep the initially unmasked code

            code = self.quantizer.embedding(sample_ids) #(B, total, code_dim)

            code_idx = sample_ids.clone()

            if t != T - 1:
                code_idx[mask] = -1
                code = torch.where(mask.unsqueeze(-1), self.transformer.mask_token, code)

                num_unknown_code = (code_idx == -1).sum(dim=-1) #(B,)

                print("num unknown code for batch 0: ", num_unknown_code[0])


        rec_lidar_logit = self.decoder(code) #(B,in_chans,H,W)

        rec_binary_voxels = (rec_lidar_logit.sigmoid()>0.5).float() #gumbel_sigmoid(rec_lidar_logit, hard=True) #
        #rec_binary_voxels = (rec_binary_voxels >= 0.5).float()

        xyzs_list = voxels2points(self.voxelizer, rec_binary_voxels)

        return xyzs_list, rec_lidar_logit, rec_binary_voxels
    

    def iterative_generation_driver(self, voxels_mask, voxels_occupancy_has, BEV_mask, generation_iter, denoise_iter, mode):
        '''
        The driver method to generate occupancy voxels

        voxels_mask: (B, H, W, in_chans), indicate which voxel is mask
        voxels_occupancy_has: (B, in_chans, H, W)
        BEV_mask: (B, H, W), indicates which voxel in bird-eye view is masked, should be consistent with boxels_mask
        generation_iter: how many iterations to do the conditional generation
        denoise_iter: how many iterations to run denoising
        mode: "polar" or "spherical
        '''
        voxels_occupancy_no = torch.clone(voxels_occupancy_has)
        print(f"+++++ ITERATIVE GENERATION")
        #### compute iterative generation performance
        xyzs_list, rec_lidar_logit, gen_binary_voxels = self.conditional_generation(voxels_occupancy_no, BEV_mask, T=generation_iter) #logit and voxel: (B,in_chans,H,W)
        gen_binary_voxels[voxels_mask.permute(0,3,1,2)==0] = voxels_occupancy_has[voxels_mask.permute(0,3,1,2)==0]

        #### denoising ######
        for i in range(denoise_iter):
            print(f"$$$$$ denoising iteration {i} $$$$$$")
            ## randomly drop some of the mask, and generate again
            if mode=="polar":
                mask_ratio = 0.5
            else:
                mask_ratio = 0.5 #0.02
            num_masked = len(BEV_mask[BEV_mask==1])
            drop_mask = np.ones((num_masked, ))
            drop_mask[:(int)((1-mask_ratio)*num_masked)] = 0
            np.random.shuffle(drop_mask)
            rand_BEV_mask = torch.clone(BEV_mask)
            rand_BEV_mask[rand_BEV_mask==1] = torch.tensor(drop_mask).to(gen_binary_voxels.device).long()

            loss, acc, cache = self.one_step_predict(gen_binary_voxels, voxels_occupancy_has, rand_BEV_mask)
            pred_code_indices_logit = cache["pred_logit"] #(B, total, code_dim)
            _, gen_binary_voxels = self.one_step_decode(pred_code_indices_logit) #(B, in_chans, H, W)

            gen_binary_voxels[voxels_mask.permute(0,3,1,2)==0] = voxels_occupancy_has[voxels_mask.permute(0,3,1,2)==0]

        return gen_binary_voxels
    
    def get_blank_code(self, path, name="blank_code", iter=20):
        '''
        Get blank code, which are the top 20 indices that most frequently occur during sampling
        '''
        B = 1
        total = self.img_size_patched[0]*self.img_size_patched[1]
        x = self.transformer.mask_token.repeat(B, total, 1) #(B, total, code_dim)
        
        for i in range(iter):
            print(f"{i}")
            pred = self.transformer(x) #(B, total, num_codes)
            sample_ids = torch.distributions.Categorical(logits=pred).sample() #(B, total)
            codes, counts = count(sample_ids)
            code_dict = {i:0 for i in range(self.num_code)}
            for code_idx, conut_nbs in zip(codes, counts):
                code_dict[int(code_idx.data)] += conut_nbs
                
        blank_code = [k for k, v in sorted(code_dict.items(), key=lambda item: item[1], reverse=True)[:20]] #(top 20 most frequent code indices)
        print("\nThe blank code has already been successfully counted, and will exit automatically soon")
        with open(os.path.join(path, f"{name}.pickle"), 'wb') as handle:
            pickle.dump(blank_code, handle, protocol=3)   

        self.blank_code = blank_code






if __name__=="__main__":
    device = torch.device("cpu") # TODO: need to switch gpu, cuda out of memory
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- device: ", device)


    h,w = 512,512
    B = 1
    in_chans = 40
    img_size = [h,w]

    window_size=8
    patch_size=8
    
    # voxel occupancy
    x = torch.randint(low=0, high=2, size=(B, in_chans, h, w)).float().to(device)

    vqvae = VQVAETrans(
        img_size,
        in_chans=in_chans,
        patch_size=patch_size,
        window_size=window_size,
        patch_embed_dim=512,
        num_heads=16,
        depth=12,
        codebook_dim=1024,
        num_code=1024,
        beta=0.1,
        device=device
    ).to(device)

    # loss, rec_x, rec_x_logit, perplexity, min_encodings, min_encoding_indices = vqvae.compute_loss(x)
    # vqvae.update_reservoir_codebook(min_encodings, min_encoding_indices)

    voxel_mask = torch.randint(low=0, high=2, size=(B, h, w)).float().to(device)

    voxelizer = Voxelizer(grid_size=[h,w,in_chans], max_bound=[50,2*np.pi,3], min_bound=[0,0,-5])
    
    mask_git = MaskGIT(vqvae=vqvae, voxelizer=voxelizer).to(device)
   
    #loss, acc = mask_git.compute_loss(x, x, voxel_mask)

    
    xyzs_list, rec_lidar_logit, rec_binary_voxels = mask_git.conditional_generation(x, voxel_mask, T=1)
    #mask_git.get_blank_code(path=".")

    intensity = np.ones((len(xyzs_list[0]),))
    plot_points_and_voxels(lidar_xyz=xyzs_list[0], intensity=intensity, voxel_xyz=None, labels=None, xlim=[-20,20], ylim=[-20,20], vis=True, title="reconstructed lidar_points", path=None, name=None)
    voxelizer.vis_BEV_binary_voxel(rec_binary_voxels[0].permute(1,2,0), points_xyz=None, intensity=None, vis=True, path=None, name=None)
