import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_  # , xavier_uniform_, constant_ 
from torch.nn.attention import SDPBackend, sdpa_kernel

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout  # rate of train elements randomly set to zero in each forward pass (prevent overfitting)

        # Projection matrices to project Q, K, V and output to the desired dimension (dim_V)
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        # Weight initialization
        # NOTE: xavier_uniform_ might work better for a few datasets
        xavier_normal_(self.fc_q.weight)
        xavier_normal_(self.fc_k.weight)
        xavier_normal_(self.fc_v.weight)
        xavier_normal_(self.fc_o.weight)

        # NOTE: this constant bias might work better for a few datasets
        # constant_(self.fc_q.bias, 0.01)
        # constant_(self.fc_k.bias, 0.01)
        # constant_(self.fc_v.bias, 0.01)
        # constant_(self.fc_o.bias, 0.01)

        # NOTE: this additional LN for queries/keys might be useful for some datasets (DOCKSTRING)
        # self.ln_q = nn.LayerNorm(dim_Q, eps=1e-8)
        # self.ln_k = nn.LayerNorm(dim_K, eps=1e-8)

    def forward(self, Q, K, adj_mask=None):
        # Project Q, K, V
        Q = self.fc_q(Q)
        V = self.fc_v(K)
        K = self.fc_k(K)

        # Additional normalisation for queries/keys. See above
        # Q = self.ln_q(Q).to(torch.bfloat16)
        # K = self.ln_k(K).to(torch.bfloat16)

        # Reshape for multi-head attention: [batch, seq_len, num_heads, head_dim]
        batch_size = Q.size(0)
        embedding_dim = self.fc_q.out_features  # dim_V
        assert embedding_dim % self.num_heads == 0, "Embedding dim is not divisible by num_heads"
        head_dim = embedding_dim // self.num_heads
        Q = Q.view(batch_size, -1, self.num_heads, head_dim)
        K = K.view(batch_size, -1, self.num_heads, head_dim)
        V = V.view(batch_size, -1, self.num_heads, head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention mask
        if adj_mask is not None:
            adj_mask = adj_mask.expand(-1, self.num_heads, -1, -1)

        try:    
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                out = F.scaled_dot_product_attention(
                    Q, K, V, attn_mask=adj_mask, dropout_p=self.dropout if self.training else 0, is_causal=False
                )
        except RuntimeError as e:
            out = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=adj_mask, dropout_p=self.dropout if self.training else 0, is_causal=False
            )
        
        # Transpose back and flatten head dimension
        out = out.transpose(1, 2).reshape(batch_size, -1, self.num_heads * head_dim)
        # Final output projection with a residual connection and nonlinearity (Mish)
        out = out + F.mish(self.fc_o(out))
        return out

# Same input for both Q and K
class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, dropout=0.0):
        super(SelfAttention, self).__init__()
        self.mha = MultiHeadAttention(dim_in, dim_in, dim_out, num_heads, dropout)

    def forward(self, X, adj_mask=None):
        return self.mha(X, X, adj_mask=adj_mask)
    
   
# Pooling by Multihead Attention: Pools a set of elements to a fixed number of outputs (seeds)
# num_seeds = 32 (An end-to-end attention-based approach for learning on graphs, cap. 3.2)
class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds=32, dropout=0.0):
        super(PMA, self).__init__()
        # Learnable seed vectors for pooling
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_normal_(self.S)
        # MultiHeadAttention takes seeds as Q and the input set as K
        self.mha = MultiHeadAttention(dim, dim, dim, num_heads, dropout=dropout)

    def forward(self, X, adj_mask=None):
        # Repeat seeds across batch, use seeds as queries; X as keys/values
        return self.mha(self.S.repeat(X.size(0), 1, 1), X)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, layer_type):
        super(TransformerBlock, self).__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.layer_type = layer_type
        if layer_type == 'P':
            self.attention = PMA(hidden_dim, num_heads)
        else:
            self.attention = SelfAttention(hidden_dim, hidden_dim, num_heads)

    def forward(self, X, adj_mask=None):
        # assert (self.layer_type == 'M') == (adj_mask is not None)
        if self.layer_type != 'M':
            adj_mask = None
        return self.attention(self.norm(X), adj_mask=adj_mask)
    
class ESA(nn.Module):
    # Specify the number and order of layers:
    #   S for self-attention
    #   M for masked self-attention
    #   P for the PMA decoder
    # S and M layers can be alternated in any order as desired.
    # For graph-level tasks, there must be a single P layer specified.
    # The P layer can be followed by S layers (decoder), but not by M layers.
    # Always use nn.ModuleList (or nn.Sequential) for lists of layers in PyTorch!
    def __init__(self, hidden_dim, num_heads, layer_types):
        super(ESA, self).__init__()
        assert layer_types.count('P') == 1
        # Encoder
        enc_layers = layer_types[:layer_types.index('P')]
        self.encoder = nn.ModuleList()
        for layer_type in enc_layers:
            assert layer_type in 'MS'
            self.encoder.append(TransformerBlock(hidden_dim, num_heads, layer_type))
        # Decoder
        dec_layers = layer_types[layer_types.index('P') + 1:]
        self.decoder = nn.Sequential(TransformerBlock(hidden_dim, num_heads, 'P'))
        for layer_type in dec_layers:
            assert layer_type == 'S'
            self.decoder.append(TransformerBlock(hidden_dim, num_heads, layer_type))

    def forward(self, X, adj_mask=None):
        for layer in self.encoder:
            X = layer(X, adj_mask=adj_mask)
        return self.decoder(X)
        
            