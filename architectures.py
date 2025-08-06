import torch.nn as nn
import torch.nn.functional as F
# from attention import *
from attention import *

def mlp(in_dim, inter_dim, out_dim):
    return nn.Sequential(
            nn.Linear(in_dim, inter_dim),
            nn.Mish(),
            nn.Linear(inter_dim, out_dim),
        )


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, layer_type, mlp_hidden_dim=256):
        super(TransformerBlock, self).__init__()
        self.layer_type = layer_type
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.norm_mlp = nn.LayerNorm(hidden_dim, eps=1e-8)
        if layer_type == 'P':
            self.attention = PMA(hidden_dim, num_heads)
        else:
            self.attention = SelfAttention(hidden_dim, hidden_dim, num_heads)
        self.mlp = mlp(hidden_dim, mlp_hidden_dim, hidden_dim)

    def forward(self, X, adj_mask=None, return_attention=False):
        if self.layer_type != 'M':
            adj_mask = None
        # Attention
        if return_attention:
            out, attn_scores = self.attention(self.norm(X), adj_mask=adj_mask, return_attention=True)
        else:
            out = self.attention(self.norm(X), adj_mask=adj_mask)
            attn_scores = None
        if self.layer_type != 'P':
            out = X + out  # Residual connection
        # MLP
        out_mlp = self.mlp(self.norm_mlp(out))  # Pre-LayerNorm
        out = out + out_mlp  # Residual connection
        if return_attention:
            return out, attn_scores
        return out

class ESA(nn.Module):
    # Specify the number and order of layers:
    #   S for self-attention
    #   M for masked self-attention
    #   P for the PMA decoder
    # S and M layers can be alternated in any order as desired.
    # For graph-level tasks, there must be a single P layer specified.
    # The P layer can be followed by S layers (decoder), but not by M layers.
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
        self.decoder = nn.ModuleList()
        self.decoder.append(TransformerBlock(hidden_dim, num_heads, 'P'))
        for layer_type in dec_layers:
            assert layer_type == 'S'
            self.decoder.append(TransformerBlock(hidden_dim, num_heads, layer_type))
        # self.decoder_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)  # no need since graph_dim = hidden_dim?

    def forward(self, X, adj_mask, return_attention=True):
        enc = X
        for layer in self.encoder:
            enc = layer(enc, adj_mask=adj_mask)
        dec = enc + X  # Residual connection
        for layer in self.decoder:
            if return_attention and layer.layer_type == 'P':
                dec, attn_scores = layer(dec, return_attention=return_attention)
                attn_scores = attn_scores.sum(dim=1)  # Aggregate seeds by sum
            else:
                dec = layer(dec)
        out = dec.mean(dim=1)  # Aggregate seeds by mean
        if return_attention:
            attn_weights = F.softmax(attn_scores, dim=-1)
            return F.mish(out), attn_weights
        return F.mish(out)  # Final output projection
        # return F.mish(self.decoder_linear(out))  # Final output projection 