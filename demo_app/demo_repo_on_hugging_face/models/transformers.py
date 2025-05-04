import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from timm.models.swin_transformer import BasicLayer, PatchMerging
from timm.models.vision_transformer import PatchEmbed



'''
Transformers for encoder, decoder and prediction of code indices.
Training examples are assumed to be:
- input: occupancy grid (B,#z,#r,#theta)
- label: binary mask (1/0) (B,#r,#theta)
'''

################################
# Positional embedding of a grid
################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: grid height and width [H,W]
    return:
    pos_embed: [H*W, embed_dim] 
                or [1+H*W, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0) # shape: (2, H, W)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position (D)
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0 # divisible by 2? the pos embed will be concatenation of sine and cosine embedding
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

################################################
########################## Transformers ########
################################################

class BidirectionalTransformer(nn.Module):
    '''
    Maps each vector of an image to logits of a categorical distribution over classes. In our case, the classes are the indices of codebook embeddings. 

    decoder_embed (map to hidden_dim) --> + pos_embed --> swin transformer --> norm --> pred (map to num_class)
    
    input vector size: input_dim (embedding dim), each vector corresponds to a pixel
    output vector size: num_class (categorical distribution over embedding vector's indices)
    '''
    def __init__(self, num_class, input_dim, img_size, hidden_dim=512, depth=24, num_heads=16, window_size=8):
        super().__init__()
        assert(len(img_size)==2)
        self.num_class = num_class
        self.input_dim = input_dim
        self.H = img_size[0]
        self.W = img_size[1]
        self.hidden_dim = hidden_dim
        self.decoder_embed = nn.Linear(input_dim, hidden_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, input_dim), requires_grad=True)
        token_size = self.H*self.W
        self.pos_embed = nn.Parameter(torch.zeros(1, token_size, hidden_dim), requires_grad=False)
        self.blocks = BasicLayer(
            hidden_dim,
            (self.H, self.W),
            depth,
            num_heads=num_heads,
            window_size=window_size,
            downsample=None,
            # use_checkpoint=True,
        )
        self.norm = nn.Sequential(nn.LayerNorm(hidden_dim), nn.GELU())
        self.pred = nn.Linear(hidden_dim, num_class, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.hidden_dim, (self.H, self.W), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        '''
        x: shape (B (optional), H*W, e_dim)
        '''
        # embed tokens
        x = self.decoder_embed(x) #(B, H*W, hidden_dim)

        # add pos embed
        x = x + self.pos_embed #(B, H*W, hidden_dim)

        # apply Transformer blocks
        x = self.blocks(x) #(B, H*W, hidden_dim)
        x = self.norm(x)

        # predictor projection
        x = self.pred(x) #(B, H*W, n_e)

        return x
    
class VQEncoderTrans(nn.Module):
    '''
    Encoder using transformer. Maps image to embeddings of codebook_dim. 

    img_size: (H,W) of the input image of shape (B,in_chans,H,W)
    patch_size: for patch embedding (apply convolution with kernel of size patch_size and stride of patch_size)
    window_size: for swin transformer
    num_heads: for swin transformer
    embed_dim: the new number of channels after patch embed
    depth: number of swin transformer blocks
    codebook_dim: dimension of the codebook embedding
    '''
    def __init__(
        self,
        img_size,
        patch_size=8,
        window_size=8,
        in_chans=40,
        embed_dim=512,
        num_heads=16,
        depth=12,
        codebook_dim=1024,
    ):
        super().__init__()

        norm_layer = nn.LayerNorm
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, norm_layer=norm_layer)
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.h = img_size[0] // patch_size
        self.w = img_size[1] // patch_size

        self.blocks = [
            BasicLayer(
                embed_dim,
                (self.h, self.w),
                depth,
                num_heads=num_heads,
                window_size=window_size,
                downsample=None,
                # use_checkpoint=False,
            ),
        ]

        self.blocks = nn.Sequential(*self.blocks)

        self.norm = nn.Sequential(norm_layer(embed_dim), nn.GELU())
        self.pre_quant = nn.Linear(embed_dim, codebook_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # nn.init.constant_(self.pre_quant.weight, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x) #(B, H/patch_size*W/patch_size, embed_dim)

        # add pos embed w/o cls token
        x = x + self.pos_embed #(B, H/patch_size*W/patch_size, embed_dim)

        # apply Transformer blocks
        x = self.blocks(x) #(B, H/patch_size*W/patch_size, embed_dim)
        x = self.norm(x)

        # map to codebook dim
        x = self.pre_quant(x) #(B, H/patch_size*W/patch_size, codebook_dim)

        return x
    
    
class VQDecoderTrans(nn.Module):
    '''
    Decoder using transformer. Maps embeddings in the codebook back to the original image.

    img_size: (H,W) of the input image of shape (B,in_chans,H,W) to the encoder
    num_patches: number of patches after patch embedding
    patch_size: for patch embedding (apply convolution with kernel of size patch_size and stride of patch_size)
    window_size: for swin transformer
    num_heads: for swin transformer
    embed_dim: the new number of channels after patch embed
    depth: number of swin transformer blocks
    codebook_dim: dimension of the codebook embedding
    '''
    def __init__(
        self,
        img_size,
        num_patches,
        patch_size=8,
        window_size=8,
        in_chans=40,
        embed_dim=512,
        num_heads=16,
        depth=12,
        codebook_dim=1024,
        bias_init=-3,
    ):
        super().__init__()

        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        norm_layer = nn.LayerNorm
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.h = img_size[0] // patch_size
        self.w = img_size[1] // patch_size
        self.num_patches = num_patches
        self.decoder_embed = nn.Linear(codebook_dim, embed_dim, bias=True)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = BasicLayer(
            embed_dim,
            (self.h, self.w),
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
        )

        self.norm = nn.Sequential(norm_layer(embed_dim), nn.GELU())
        self.pred = nn.Linear(embed_dim, patch_size**2 * in_chans, bias=True)
        self.initialize_weights()
        nn.init.constant_(self.pred.bias, bias_init)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        p = self.patch_size
        h, w = self.h, self.w
        assert h * w == x.shape[1]

        B = x.shape[0]
        x = x.reshape(shape=(B, h, w, p, p, self.in_chans))
        x = torch.einsum("nhwpqc->nchpwq", x) #(B,in_chans,h,p,w,p)
        imgs = x.reshape(shape=(B, self.in_chans, h * p, w * p))

        return imgs

    def forward(self, x):
        '''
        x: (B, num_patches, codebook_dim)
        '''
        # embed tokens
        x = self.decoder_embed(x) #(B, num_patches, embed_dim)

        # add pos embed
        x = x + self.pos_embed #(B, num_patches, embed_dim)

        # apply Transformer blocks
        x = self.blocks(x) #(B, num_patches, embed_dim)
        x = self.norm(x) #(B, num_patches, embed_dim)

        # predictor projection
        x = self.pred(x) #(B, num_patches, patch_size**2*in_chans)
        x = self.unpatchify(x) # the dimension of the original image: (B, in_chans, img_height, img_width)
        
        return x
    

    
if __name__=="__main__":
    device = torch.device("cpu") # TODO: need to switch gpu, cuda out of memory
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- device: ", device)

    #### test pos embed
    # pos_emb = get_2d_sincos_pos_embed(embed_dim=100, grid_size=[40,30], cls_token=False)
    # print("pos_embed", pos_emb.shape)
    
    #### test the encoder and decoder
    n_e = 1024
    e_dim=256
    h,w = 512,512
    B = 2
    in_chan = 40
    img_size = [h,w]
    window_size=8
    patch_size=8
    
    x = np.random.random_sample((B, in_chan, h, w))
    x = torch.tensor(x).float().to(device)
    
    encoder = VQEncoderTrans(
        img_size,
        patch_size=patch_size,
        window_size=window_size,
        in_chans=in_chan,
        embed_dim=512,
        num_heads=16,
        depth=12,
        codebook_dim=e_dim,
    ).to(device)
    
    z_e = encoder(x)
    print("encoded: ", z_e.shape)

    decoder = VQDecoderTrans(
        img_size,
        num_patches=encoder.num_patches,
        patch_size=patch_size,
        window_size=window_size,
        in_chans=in_chan,
        embed_dim=512,
        num_heads=16,
        depth=12,
        codebook_dim=e_dim,
        bias_init=-3,
    ).to(device)

    decoded_x = decoder(z_e)

    ##### test transformer
    # z = torch.randn((B, h,w, e_dim)).float().to(device)
    # z = z.reshape(B, -1, e_dim)
    # transformer = BidirectionalTransformer(num_class=n_e, input_dim=e_dim, img_size=img_size, hidden_dim=512, depth=24, num_heads=16, window_size=8).to(device)
    # logit = transformer(z)
    # print("transformer logit: ", logit.shape)