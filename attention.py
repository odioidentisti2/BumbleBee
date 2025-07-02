import torch
import torch.nn as nn

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.ln0 = nn.LayerNorm(dim_V)
        self.ln1 = nn.LayerNorm(dim_V)
        self.fc_r = nn.Sequential(
            nn.Linear(dim_V, dim_V),
            nn.ReLU(),
            nn.Linear(dim_V, dim_V)
        )

    def forward(self, Q, K, adj_mask=None):
        Q_ = self.fc_q(Q)
        K_ = self.fc_k(K)
        V_ = self.fc_v(K)
        # Reshape for MHA: split the last dimension into heads and cat along batch dimension 
        Q_ = torch.cat(Q_.split(Q_.size(-1) // self.num_heads, dim=2), dim=0)
        K_ = torch.cat(K_.split(K_.size(-1) // self.num_heads, dim=2), dim=0)
        V_ = torch.cat(V_.split(V_.size(-1) // self.num_heads, dim=2), dim=0)
        # [B*H, N, D], where B=batch, H=heads, N=items

        # Compute raw attention scores
        attn = Q_.bmm(K_.transpose(1, 2)) / (K_.size(-1) ** 0.5)

        print("attn.shape:", attn.shape)
        print("adj_mask.shape:", adj_mask.shape)

        # Apply mask if provided (mask shape: [B, N, N] or [B*H, N, N])
        if adj_mask is not None:
            # If mask is [B, N, N], repeat for heads to [B*H, N, N]
            if adj_mask.shape[0] != attn.shape[0]:
                adj_mask = adj_mask.repeat_interleave(self.num_heads, dim=0)
            # Set masked-out positions to a large negative value
            attn = attn.masked_fill(adj_mask == 0, float('-inf'))

        A = torch.softmax(attn, dim=2)
        # Reshape after MHA: apply attention to V and merges heads back (split and cat)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), dim=2), dim=2)
        
        # Applies layer normalization, then adds a two-layer MLP (with ReLU and linear), then another layer norm
        O = self.ln0(O)
        O = O + self.fc_r(O)
        return self.ln1(O)

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads)
    def forward(self, X, adj_mask=None):
        return self.mab(X, X, adj_mask=adj_mask)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads)
    def forward(self, X, adj_mask=None):
        S = self.S.repeat(X.size(0), 1, 1)
        return self.mab(S, X, adj_mask=adj_mask)