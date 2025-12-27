import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_ 

from parameters import GLOB

COUNTER = 0  # For debugging


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout

        # Projection matrices
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        # Weight initialization
        xavier_normal_(self.fc_q.weight)
        xavier_normal_(self.fc_k.weight)
        xavier_normal_(self.fc_v.weight)
        xavier_normal_(self.fc_o.weight)

        # Additional LN for queries/keys
        self.ln_q = nn.LayerNorm(dim_Q, eps=1e-8)
        self.ln_k = nn.LayerNorm(dim_K, eps=1e-8)

    def forward(self, Q, K, mask, bond_modulation, return_attention=False):
        """
        Bond-Modulated Multi-Head Attention.
        
        Args:
            Q: [batch, seq_len, dim_Q]
            K: [batch, seq_len, dim_K]
            mask: [batch, num_heads, seq_len, seq_len] - attention mask
            bond_modulation: [batch, seq_len, seq_len, dim_V] - projected bond features
            return_attention: bool - whether to return attention weights
        
        Returns:
            out: [batch, seq_len, dim_V]
            attn_weights (optional): [batch, seq_len, seq_len] - averaged over heads
        """
        # Project Q, K, V
        Q = self.fc_q(Q)
        V = self.fc_v(K)
        K = self.fc_k(K)

        # Additional normalisation
        Q = self.ln_q(Q)
        K = self.ln_k(K)

        # Reshape for multi-head attention
        batch_size = Q.size(0)
        embedding_dim = self.fc_q.out_features
        assert embedding_dim % self.num_heads == 0
        head_dim = embedding_dim // self.num_heads
        
        Q = Q.view(batch_size, -1, self.num_heads, head_dim)
        K = K.view(batch_size, -1, self.num_heads, head_dim)
        V = V.view(batch_size, -1, self.num_heads, head_dim)

        Q = Q.transpose(1, 2)  # [batch, num_heads, seq_len_q, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Bond-modulated attention
        # Reshape bond_modulation: [batch, seq_len, seq_len, embedding_dim]
        # -> [batch, num_heads, seq_len_q, seq_len_k, head_dim]
        bond_mod_reshaped = bond_modulation.view(
            batch_size, bond_modulation.size(1), bond_modulation.size(2), 
            self.num_heads, head_dim
        )
        bond_mod_reshaped = bond_mod_reshaped.permute(0, 3, 1, 2, 4)
        
        # Modulate keys based on bonds
        # Each query position sees different keys based on bond features
        K_modulated = K.unsqueeze(2) + bond_mod_reshaped
        # K: [batch, num_heads, seq_len_k, head_dim]
        # K_modulated: [batch, num_heads, seq_len_q, seq_len_k, head_dim]
        
        # Compute attention scores with bond-modulated keys
        scale = Q.size(-1) ** -0.5
        # Q: [batch, num_heads, seq_len_q, head_dim]
        # Use einsum for per-query key modulation
        attn_scores = torch.einsum('bhqd,bhqkd->bhqk', Q, K_modulated) * scale
        # [batch, num_heads, seq_len_q, seq_len_k]
        
        # Apply mask
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Dropout
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)
        # [batch, num_heads, seq_len_q, head_dim]

        # Transpose back and flatten
        out = out.transpose(1, 2).reshape(batch_size, -1, self.num_heads * head_dim)
        
        # Final output projection
        out = out + F.mish(self.fc_o(out))
        
        if return_attention:
            attn_weights = attn_weights.mean(dim=1)  # Average over heads
            return out, attn_weights
        return out


# Same input for both Q and K
class SetAttention(nn.Module):
    """
    Self-Attention for sets/graphs with bond modulation.
    Q = K = X (self-attention within the set).
    """

    def __init__(self, dim_in, dim_out, num_heads):
        super(SetAttention, self).__init__()
        self.mha = MultiHeadAttention(dim_in, dim_in, dim_out, num_heads, GLOB['SAB_dropout'])

    def forward(self, X, mask, bond_modulation, return_attention=False):
        """
        Args:
            X: [batch, seq_len, dim_in]
            mask: [batch, seq_len, seq_len] - adjacency/padding mask
            bond_modulation: [batch, seq_len, seq_len, dim_out] - projected bond features
            return_attention: bool
        """
        # Expand mask for multi-head attention
        mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
        mask = mask.expand(-1, self.mha.num_heads, -1, -1)
        
        return self.mha(X, X, mask, bond_modulation, return_attention)


class PMA(nn.Module):
    """
    Pooling by Multihead Attention.
    Cross-attention from learnable seed vectors to input set.
    No bond modulation (seeds have no bond structure).
    """

    def __init__(self, dim, num_heads, num_seeds=32):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        xavier_uniform_(self.S)
        self.mha = MultiHeadAttention(dim, dim, dim, num_heads, GLOB['PMA_dropout'])

    def forward(self, X, mask):
        """
        Args:
            X: [batch, seq_len, dim]
            mask: [batch, seq_len] - padding mask
        """
        batch_size, seq_len, dim = X.shape
        queries = self.S.repeat(batch_size, 1, 1)  # [batch, num_seeds, dim]
        num_seeds = queries.size(1)
        
        # Create dummy bond modulation (zeros - no bond structure for seeds)
        bond_modulation = torch.zeros(
            batch_size, num_seeds, seq_len, dim, 
            device=X.device
        )
        
        # Expand mask for multi-head attention
        mask = mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
        mask = mask.expand(-1, self.mha.num_heads, num_seeds, -1)
        
        return self.mha(queries, X, mask, bond_modulation)
