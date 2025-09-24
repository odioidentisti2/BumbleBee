import torch.nn as nn
import torch.nn.functional as F
from attention import *


# Multilayer Perceptron
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
        self.attn_weights = None
        self.mlp = mlp(hidden_dim, mlp_hidden_dim, hidden_dim)

    def forward(self, X, adj_mask=None, pad_mask=None):
        if self.layer_type == 'M': 
            mask = adj_mask
            # if pad_mask is not None:
            #     mask = mask & pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)  # [batch, seq_len, seq_len]
        elif self.layer_type == 'S':
            mask = None
            # if pad_mask is not None:
            #     mask = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)  # [batch, seq_len, seq_len]
        else:  # 'P'
            # mask = pad_mask  # [batch, seq_len]
            mask = None
        # Attention
        out, self.attn_weights = self.attention(self.norm(X), mask=mask)
        if self.layer_type != 'P':
            out = X + out  # Residual connection
        # MLP
        out_mlp = self.mlp(self.norm_mlp(out))  # Pre-LayerNorm
        out = out + out_mlp  # Residual connection

        # # Zero out output for padded queries
        # if pad_mask is not None and self.layer_type != 'P':
        #     out = out * pad_mask.unsqueeze(-1)

        return out
    

class ESA(nn.Module):
    """
    ESA (Edge-Set Attention) module for graph-level tasks.

    Args:
        hidden_dim (int): The dimensionality of the hidden representations.
        num_heads (int): Number of attention heads in each transformer block.
        layer_types (str): Specify the order and type of layers.
            - 'S': Self-attention layer.
            - 'M': Masked self-attention layer.
            - 'P': PMA (Pooling by Multihead Attention) decoder layer.
            The string must contain exactly one 'P' layer. 'S' and 'M' layers can be arranged in any order before 'P'.
            After 'P', only 'S' layers are allowed.

    Attributes:
        encoder (nn.ModuleList): List of transformer blocks for the encoder ('S' and 'M' layers).
        decoder (nn.ModuleList): List of transformer blocks for the decoder ('P' and followed by optional 'S' layers).

    Returns:
        torch.Tensor: Output graph-level representation after pooling and non-linearity.
        torch.Tensor (optional): Attention scores from the 'P' layer if `return_attention` is True.
    """
    def __init__(self, hidden_dim, num_heads, layer_types):
        super(ESA, self).__init__()
        assert layer_types.count('P') == 1
        # Encoder
        enc_layers = layer_types[:layer_types.index('P')]
        self.encoder = nn.ModuleList()
        for layer_type in enc_layers:
            assert layer_type in ['M', 'S']
            self.encoder.append(TransformerBlock(hidden_dim, num_heads, layer_type))
        # Decoder
        dec_layers = layer_types[layer_types.index('P') + 1:]
        self.decoder = nn.ModuleList()
        self.decoder.append(TransformerBlock(hidden_dim, num_heads, 'P'))
        for layer_type in dec_layers:
            assert layer_type == 'S'
            self.decoder.append(TransformerBlock(hidden_dim, num_heads, layer_type))
        # self.decoder_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)  # no need since graph_dim = hidden_dim?

    def forward(self, X, adj_mask, padding_mask=None):
        # Encoder
        enc = X
        for layer in self.encoder:
            enc = layer(enc, adj_mask=adj_mask, pad_mask=padding_mask)
        dec = enc + X  # Residual connection
        # Decoder
        for layer in self.decoder:
                dec = layer(dec, pad_mask=padding_mask)
        out = dec.mean(dim=1)  # Aggregate seeds by mean
        return F.mish(out)  
        # return F.mish(self.decoder_linear(out))

    def get_attn_weights(self):
        return self.decoder[-1].attn_weights.mean(dim=1)  # Aggregate seeds by mean (sum?)