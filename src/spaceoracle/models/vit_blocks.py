import numpy as np
import torch 
import torch.nn as nn 


class ViT(nn.Module):
    def __init__(self, betas, in_channels, spatial_dim, n_patches=16, n_blocks=2, hidden_d=8, n_heads=2):
        super().__init__()
        self.betas = betas
        self.dim = betas.shape[0] # number of TFs

        # Attributes
        chw = (in_channels, spatial_dim, spatial_dim) # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))       # Consider removing
        self.pos_embed = nn.Parameter(get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d))
        self.pos_embed.requires_grad = False
        
        self.blocks = nn.ModuleList([ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        
        self.mlp = nn.Linear(self.hidden_d, self.dim)

    def forward(self, images, inputs_labels):
        n, c, h, w = images.shape 
        patches = patchify(images, self.n_patches).to(self.pos_embed.device)
        
        tokens = self.linear_mapper(patches)
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        out = tokens + self.pos_embed.repeat(n, 1, 1)
        
        for block in self.blocks:
            out = block(out)
            
        # Get only the classification token
        out = out[:, 0]

        # Pass through mlp to get betas
        betas = self.mlp(out)
        return betas
        
    
    def get_att_weights(self, images, inputs_x, inputs_labels):
        n, c, h, w = images.shape 
        patches = patchify(images, self.n_patches).to(self.pos_embed.device)
        
        tokens = self.linear_mapper(patches)
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        out = tokens + self.pos_embed.repeat(n, 1, 1)
        
        att_weights = []   # (n_blocks, batch, n_heads, seqs, seqs) where seqs is flattened patches
        for block in self.blocks:
            out, att = block.forward_att(out)
            att_weights.append(att)
        
        # make sure outs are the same as forward 
        out = out[:, 0]
        x = self.mlp(out)
        y_pred = x[:, 0]*self.betas[0]
        for w in range(self.dim-1):
            y_pred += x[:, w+1]*inputs_x[:, w]*self.betas[w+1]

        return y_pred, x, att_weights

    

class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out
    
    def forward_att(self, x):
        out, att = self.mhsa.forward_att(self.norm1(x))
        out += x
        out = out + self.mlp(self.norm2(out))
        return out, att


class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super().__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
    
    def forward_att(self, sequences):
        result = []
        atts = []

        for sequence in sequences:
            seq_result = []
            att_result = []

            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
                att_result.append(attention) 
            
            atts.append(torch.stack(att_result, dim=0))
            result.append(torch.hstack(seq_result))

        outs = torch.cat([torch.unsqueeze(r, dim=0) for r in result])

        return outs, atts


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only."

    patch_size = h // n_patches

    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(n, n_patches**2, c * patch_size**2)

    return patches