import torch.nn as nn
import torch.nn.functional as F
from attention import *

def getMLP(in_dim, inter_dim, out_dim):
    return nn.Sequential(
            nn.Linear(in_dim, inter_dim),
            nn.Mish(),
            nn.Linear(inter_dim, out_dim),
        )


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
        out = self.attention(self.norm(X), adj_mask=adj_mask)  # Pre-LayerNorm
        if self.layer_type != 'P':
            out = X + out  # Residual connection
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
        self.decoder = nn.Sequential(TransformerBlock(hidden_dim, num_heads, 'P'))
        for layer_type in dec_layers:
            assert layer_type == 'S'
            self.decoder.append(TransformerBlock(hidden_dim, num_heads, layer_type))
        # self.decoder_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)  # no need since graph_dim = hidden_dim?

    def forward(self, X, adj_mask):
         # Squeeze/Unsqueeze batch dimension before/after attention
        X = X.unsqueeze(0) 
        adj_mask = adj_mask.unsqueeze(0)
        enc = X
        for layer in self.encoder:
            enc = layer(enc, adj_mask=adj_mask)
        enc = enc + X  # Residual connection
        out = self.decoder(enc).squeeze(0).mean(dim=0)  # Aggregate seeds by mean
        return F.mish(out)  # Final output projection
        # return F.mish(self.decoder_linear(out))  # Final output projection