import torch.nn as nn
import torch.nn.functional as F
from attention import SetAttention, PMA

from parameters import model_params as PARAMS


# Multilayer Perceptron
def mlp(in_dim, inter_dim, out_dim):

    return nn.Sequential(
            nn.Linear(in_dim, inter_dim),
            nn.Mish(),
            nn.Dropout(PARAMS['ESA_dropout']),  # automatic check for training mode (identity function in eval mode)
            nn.Linear(inter_dim, out_dim),
            # nn.Dropout(dropout)
        )


class TransformerBlock(nn.Module):

    def __init__(self, hidden_dim, layer_type, mlp_expansion):
        super(TransformerBlock, self).__init__()
        self.layer_type = layer_type
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.norm_mlp = nn.LayerNorm(hidden_dim, eps=1e-8)
        if layer_type == 'P':
            self.attention = PMA(hidden_dim)
        else:
            self.attention = SetAttention(hidden_dim, hidden_dim)
        self.mlp = mlp(hidden_dim, hidden_dim * mlp_expansion, hidden_dim)

    def forward(self, X, adj_mask=None, pad_mask=None):
        mask = None
        if self.layer_type == 'M':
            mask = adj_mask
        elif self.layer_type == 'S':
            if pad_mask is not None:
                mask = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)  # [batch, seq_len, seq_len] 
        else:  # 'P'
            mask = pad_mask  # [batch, seq_len]
        # Attention
        if hasattr(self, 'attn_weights'):
            out, self.attn_weights = self.attention(self.norm(X), mask=mask, return_attention=True)
        else:
            out= self.attention(self.norm(X), mask=mask)
        if self.layer_type != 'P':
            out = X + out  # Residual connection
            # Zero out output for padded positions after each layer
            # if pad_mask is not None:
            #     out = out * pad_mask.unsqueeze(-1)

        # MLP
        out_mlp = self.mlp(self.norm_mlp(out))  # Pre-LayerNorm
        out = out + out_mlp  # Residual connection
        # Zero out output for padded positions after MLP
        # if pad_mask is not None and self.layer_type != 'P':
        #     out = out * pad_mask.unsqueeze(-1)
        return out
    

class ESA(nn.Module):
    """
    ESA (Edge-Set Attention) module for graph-level tasks.

    Args:
        hidden_dim (int): The dimensionality of the hidden representations.
        layer_types (str): Specify the order and type of layers.
            - 'S': Self-attention layer.
            - 'M': Masked self-attention layer.
            - 'P': PMA (Pooling by Multihead Attention) decoder layer.
            The string must contain exactly one 'P' layer. 'S' and 'M' layers can be arranged in any order before 'P'.
            After 'P', only 'S' layers are allowed (but they make sense only if PMA seeds (k) > 1).

    Attributes:
        encoder (nn.ModuleList): List of transformer blocks for the encoder ('S' and 'M' layers).
        decoder (nn.ModuleList): List of transformer blocks for the decoder ('P' and followed by optional 'S' layers).

    Returns:
        torch.Tensor: Output graph-level representation after pooling and non-linearity.
        torch.Tensor (optional): Attention scores from the 'P' layer if `return_attention` is True.
    """

    def __init__(self, hidden_dim, layer_types, mlp_expansion):
        super(ESA, self).__init__()
        assert layer_types.count('P') == 1
        self.output_dropout = nn.Dropout(PARAMS['ESA_dropout']) 
        # Encoder
        enc_layers = layer_types[:layer_types.index('P')]
        self.encoder = nn.ModuleList()
        for layer_type in enc_layers:
            assert layer_type[0] in ['M', 'S']
            self.encoder.append(TransformerBlock(hidden_dim, layer_type, mlp_expansion))
        # Decoder
        dec_layers = layer_types[layer_types.index('P') + 1:]
        self.decoder = nn.ModuleList()
        self.decoder.append(TransformerBlock(hidden_dim, 'P', mlp_expansion))
        for layer_type in dec_layers:
            assert layer_type == 'S'
            self.decoder.append(TransformerBlock(hidden_dim, layer_type, mlp_expansion))
        # self.decoder_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)  # no need since graph_dim = hidden_dim?

    def forward(self, X, adj_mask, pad_mask=None):
        # Encoder
        enc = X
        for layer in self.encoder:
            enc = layer(enc, adj_mask=adj_mask, pad_mask=pad_mask)
        dec = enc + X  # Residual connection
        # Decoder
        for layer in self.decoder:
            dec = layer(dec, pad_mask=pad_mask)
            pad_mask = None  # Only use pad_mask in the first decoder layer (PMA)
        out = dec.mean(dim=1)  # Aggregate seeds by mean
        out = F.mish(out)
        return self.output_dropout(out)  # Pre-activation dropout
        # return F.mish(self.decoder_linear(out))

    def expose_attention(self, expose=True):
        if expose:
            for layer in self.decoder:
                layer.attn_weights = None           
        else:
            for layer in self.decoder:
                if hasattr(layer, "attn_weights"):
                    delattr(layer, 'attn_weights')

    def get_attention(self, index=-1):
        # Aggregate seeds by mean (try sum?)
        return self.decoder[index].attn_weights.mean(dim=1)  # [batch, seq_len]
    


        # # Get top-5 attended edges for each seed
        # top_edges_per_seed = self.decoder[index].attn_weights[0, :, :].topk(5, dim=-1).indices  # [num_seeds, 5]
        # print("\nTop-5 edge indices per seed:")
        # for i in range(min(5, top_edges_per_seed.size(0))):
        #     print(f"  Seed {i:02d}: {top_edges_per_seed[i].tolist()}")
        
        # # Check overlap: how many seeds attend to the same top-5 edges?
        # unique_top_edges = torch.unique(top_edges_per_seed)
        # print(f"\nUnique edges in top-5 across all seeds: {len(unique_top_edges)} / {top_edges_per_seed.numel()}")


        # attn_per_seed = self.decoder[index].attn_weights  # preserve [seeds, tokens] (or [batch, seeds, tokens])
        # if attn_per_seed.dim() == 3:
        #     attn_per_seed = attn_per_seed.squeeze(0)  # remove batch if present
        
        # print(f"Attention (seeds × tokens): {attn_per_seed.shape}")
        # for seed_idx in range(attn_per_seed.shape[0]):
        #     first_tokens = attn_per_seed[seed_idx, :5]
        #     print(f"Seed {seed_idx:02d}: first 5 tokens = {first_tokens.tolist()}")
        
        # if attn_per_seed.shape[0] > 1:
        #     diffs = []
        #     for i in range(attn_per_seed.shape[0]):
        #         for j in range(i + 1, attn_per_seed.shape[0]):
        #             diffs.append(torch.norm(attn_per_seed[i] - attn_per_seed[j]).item())
        #     max_diff = max(diffs)
        #     print(f"Max pairwise L2 diff between seeds: {max_diff:.6f}")
        #     print(f"Seeds similar within 1e-2 tolerance? {max_diff < 1e-2}")
        
        # return attn_per_seed.mean(dim=0)