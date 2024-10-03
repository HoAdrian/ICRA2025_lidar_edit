import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta, device):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.device = device


        self.embedding = nn.Embedding(self.n_e, self.e_dim) #(num_embedding, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        assert(self.e_dim==z.shape[1])

        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t()) # shape: (B*H*W, num_embedding)
        
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1) # shape: (B*H*W, 1)

        #  min_encodings[i,j] is 1 if the ith patch's closest embedding vector is self.embedding[j]
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(self.device) #(B*H*W, num_embedding)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape) # shape: (B*H*W, e_dim) --> (B,H,W,e_dim)
        #print("z_q: ", z_q.shape)

        # compute loss for embedding
        ### TODO: beta should multiply the first term, possible bug
        # loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
        #     torch.mean((z_q - z.detach()) ** 2)

        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10))) # exp{entropy(e_mean)}

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous() # shape: (B,e_dim,H,W)

        return loss, z_q, perplexity, min_encodings, min_encoding_indices

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # random data
    x = np.random.random_sample((50, 256, 40, 30))
    x = torch.tensor(x).float().to(device)

    # test encoder
    quantizer = VectorQuantizer(n_e=1024, e_dim=256, beta=1, device=device).to(device)
    loss, z_q, perplexity, min_encodings, min_encoding_indices = quantizer(x)