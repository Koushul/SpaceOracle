import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn 
import torch.nn.functional as F


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
        
        self.pos_embed = nn.Parameter(get_positional_embeddings(self.n_patches ** 2, self.hidden_d))
        self.pos_embed.requires_grad = False
        self.label_embed = nn.Embedding(in_channels, self.hidden_d) # for cell-type information
        
        self.blocks = nn.ModuleList([ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        
        self.mlp = nn.Linear(self.hidden_d, self.dim)

    def forward(self, images, inputs_labels):
        n, c, h, w = images.shape 
        patches = patchify(images, self.n_patches).to(self.pos_embed.device)

        tokens = self.linear_mapper(patches)
        pos_embed = self.pos_embed.repeat(n, 1, 1)
        out = tokens + pos_embed
        for block in self.blocks:
            out = block(out)

        # Max pool over patches
        out = torch.mean(out, dim=1)

        label_embed = self.label_embed(inputs_labels)
        label_embed = label_embed * 0.1                 # Scale down
        out = label_embed + out

        # Pass through mlp to get betas
        betas = self.mlp(out)
        return betas
        
    
    def get_att_weights(self, images):
        n, c, h, w = images.shape 
        patches = patchify(images, self.n_patches).to(self.pos_embed.device)
        
        tokens = self.linear_mapper(patches)
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        out = tokens + self.pos_embed.repeat(n, 1, 1)
        
        att_weights = []   # (n_blocks, batch, n_heads, seqs, seqs) where seqs is flattened patches
        for block in self.blocks:
            out, att = block.forward_att(out)
            att_weights.append(att)
        
        return att_weights

    

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

    # def forward(self, sequences):
    #     # Sequences has shape (N, seq_length, token_dim)
    #     # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
    #     # And come back to    (N, seq_length, item_dim)  (through concatenation)
    #     result = []
    #     for sequence in sequences:
    #         seq_result = []
    #         for head in range(self.n_heads):
    #             q_mapping = self.q_mappings[head]
    #             k_mapping = self.k_mappings[head]
    #             v_mapping = self.v_mappings[head]

    #             seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
    #             q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

    #             attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
    #             seq_result.append(attention @ v)
    #         result.append(torch.hstack(seq_result))
    #     return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
    


    def forward(self, sequences):
        N, seq_length, token_dim = sequences.shape
        sequences = sequences.view(N, seq_length, self.n_heads, self.d_head)
        q = torch.stack([q_map(sequences[:,:,i,:]) for i, q_map in enumerate(self.q_mappings)], dim=2)
        k = torch.stack([k_map(sequences[:,:,i,:]) for i, k_map in enumerate(self.k_mappings)], dim=2)
        v = torch.stack([v_map(sequences[:,:,i,:]) for i, v_map in enumerate(self.v_mappings)], dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        output = F.scaled_dot_product_attention(q, k, v)
        return output.transpose(1, 2).contiguous().view(N, seq_length, -1)
    
    # def forward_att(self, sequences):
    #     result = []
    #     atts = []

    #     for sequence in sequences:
    #         seq_result = []
    #         att_result = []

    #         for head in range(self.n_heads):
    #             q_mapping = self.q_mappings[head]
    #             k_mapping = self.k_mappings[head]
    #             v_mapping = self.v_mappings[head]

    #             seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
    #             q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

    #             attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
    #             seq_result.append(attention @ v)
    #             att_result.append(attention) 
            
    #         atts.append(torch.stack(att_result, dim=0))
    #         result.append(torch.hstack(seq_result))

    #     outs = torch.cat([torch.unsqueeze(r, dim=0) for r in result])

    #     return outs, atts

    def forward_att(self, sequences):
        N, seq_length, token_dim = sequences.shape
        sequences = sequences.view(N, seq_length, self.n_heads, self.d_head)
        
        q = torch.stack([q_map(sequences[:,:,i,:]) for i, q_map in enumerate(self.q_mappings)], dim=2)
        k = torch.stack([k_map(sequences[:,:,i,:]) for i, k_map in enumerate(self.k_mappings)], dim=2)
        v = torch.stack([v_map(sequences[:,:,i,:]) for i, v_map in enumerate(self.v_mappings)], dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        
        outs = attention.transpose(1, 2).contiguous().view(N, seq_length, -1)
        atts = attention.softmax(dim=-1)  # Extract attention weights
        
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
