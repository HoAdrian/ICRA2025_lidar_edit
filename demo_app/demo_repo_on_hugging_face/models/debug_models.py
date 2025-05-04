
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from timm.models.swin_transformer import BasicLayer, PatchMerging
from timm.models.vision_transformer import PatchEmbed

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence. N=batch size, L=sequence length, D=dimension per token

    Outputs:
    - x_masked: The masked sequence of shape [N, len_keep, D], remaining code after masking.
    - mask: A binary mask of shape [N, L] where 0 indicates the element is kept, and 1 indicates the element is removed.
    - ids_restore: Indices to restore the original order of the sequence, shape [N,L].
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    # sort the noise and use the indices from argsort to shuffle each sequence of vectors, keep the first subset of each sequence
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1) # can be used to restore the sequence shuffled by indexing with ids_shuffle

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) #[N, len_keep, D]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore # x_keeped, | binary mask: 0 is keep, 1 is remove, | indices for each sequence

def mask_code(code, mask_token, mask_ratio, cond=None):
    '''
    Inputs:
    - code: The input code sequence of shape [N, L, D].
    - mask_token: A tensor representing the mask token, typically of shape [1, 1, D].
    - cond: Optional conditioning information to be concatenated with the masked sequence.
   
    Outputs:
    - The final sequence with mask tokens in place of the masked positions and optional conditioning information concatenated.
    - The binary mask indicating which positions were masked.
    - The indices to restore the original order.
    '''

    N,L,D = code.shape
    x, mask, ids_restore = random_masking(code, mask_ratio)
    num_keep = x.shape[1]
    # x_keep: (N, lens_keep, D)
    # mask: (N, L)
    # ids_restore: (N, L)

    mask_tokens = mask_token.repeat(x.shape[0], L - num_keep, 1) #(N, L-lens_keep, D)
    x = torch.cat([x, mask_tokens], dim=1)
    x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # unshuffle

    if cond is not None:
        x = torch.cat([x, cond], dim=-1)

    return x, mask, ids_restore



def mask_code_mine(code, mask_token, mask):
        '''
        Inputs:
        - code: The input code sequence of shape [N, L, D].
        - mask_token: mask token (special token to replace the masked features), shape [1, 1, D].
        - mask: binary mask over each sequence, shape [N,L]
        
        Outputs:
        - The final sequence with mask tokens in place of the masked positions and optional conditioning information concatenated.
        '''

        mask = mask.unsqueeze(-1)
        print("mask: ", mask)
        print("x: ", code)
        #print("after mask x: ", x)
        x = torch.where(mask==1, mask_token, code)
        print("after mask x: ", x.shape)
        x = code
        print(x[mask.repeat(1,1,3)==1].shape)
        print("* after mask x: ", x)

        return x


# def random_masking(self, x, mask_ratio):
#     """
#     Perform per-sample random masking by per-sample shuffling.
#     Per-sample shuffling is done by argsort random noise.
#     x: [N, L, D], sequence. N=batch size, L=sequence length, D=dimension per token

#     Outputs:
#     - x_masked: The masked sequence of shape [N, len_keep, D], remaining code after masking.
#     - mask: A binary mask of shape [N, L] where 0 indicates the element is kept, and 1 indicates the element is removed.
#     - ids_restore: Indices to restore the original order of the sequence, shape [N,L].
#     """
#     N, L, D = x.shape  # batch, length, dim
#     len_keep = int(L * (1 - mask_ratio))

#     noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

#     # sort noise for each sample
#     # sort the noise and use the indices from argsort to shuffle each sequence of vectors, keep the first subset of each sequence
#     ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
#     ids_restore = torch.argsort(ids_shuffle, dim=1) # can be used to restore the sequence shuffled by indexing with ids_shuffle

#     # keep the first subset
#     ids_keep = ids_shuffle[:, :len_keep]
#     x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) #[N, len_keep, D]

#     # generate the binary mask: 0 is keep, 1 is remove
#     mask = torch.ones([N, L], device=x.device)
#     mask[:, :len_keep] = 0
#     # unshuffle to get the binary mask
#     mask = torch.gather(mask, dim=1, index=ids_restore)

#     return x_masked, mask, ids_restore # x_keeped, | binary mask: 0 is keep, 1 is remove, | indices for each sequence

# def mask_code_random(self, code, mask_token, mask_ratio):
#     '''
#     Inputs:
#     - code: The input code sequence of shape [N, L, D].
#     - mask_token: A tensor representing the mask token, typically of shape [1, 1, D].

#     Outputs:
#     - x: [N,L,D], tokens with token specified by random mask replaced with mask token.
#     - mask: [N,L], the random mask on each sequence
#     '''

#     N,L,D = code.shape
#     x, mask, ids_restore = self.random_masking(code, mask_ratio)
#     num_keep = x.shape[1]
#     # x_keep: (N, lens_keep, D)
#     # mask: (N, L)
#     # ids_restore: (N, L)

#     mask_tokens = mask_token.repeat(x.shape[0], L - num_keep, 1) #(N, L-lens_keep, D)
#     x = torch.cat([x, mask_tokens], dim=1)
#     x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # unshuffle

#     return x, mask


if __name__=="__main__":
    device = torch.device("cpu") # TODO: need to switch gpu, cuda out of memory
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- device: ", device)


    h,w = 2,2
    B = 2
    in_chans = 3
    img_size = [h,w]

    window_size=8
    patch_size=8
    
    # voxel occupancy
    x = torch.randint(low=0, high=2, size=(B, in_chans, h, w)).float().to(device)
    x = x.permute(0,2,3,1).reshape(B, -1, in_chans)
    code = x
    mask = torch.zeros(B, h*w)
    mask[:,0] = 1
    mask_token = torch.ones(1, 1, in_chans)*-2

    #mask_code_mine(code, mask_token, mask)
    #random_masking(code, mask_ratio=0.5)
    mask_code(code, mask_token, mask_ratio=0.5, cond=None)

    # arr = torch.tensor([0,1,2,3])
    # arr[[0,1,0,1]]+=1
    # print(arr)
