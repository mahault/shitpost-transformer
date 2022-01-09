import torch
from torch import *

class Transformer(nn.Module):
    def __init__(
        self, d_model=512, num_heads=8, num_encoders=6, num_decoders=6
    ):
        super().__init__()
        self.encoder = Encoder(d_model, num_heads, num_encoders)
        self.decoder = Decoder(d_model, num_heads, num_decoders)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return dec_out
    
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, num_encoders):
        super().__init__()
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads) for _ in range(num_encoders)],
        )
    
    def forward(self, src, src_mask):
        output = src
        for layer in self.enc_layers:
            output = layer(output, src_mask)
        return output
    
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, num_decoders):
        super().__init__()
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads)for _ in range(num_decoders)],
        )
        
    def forward(self, tgt, enc, tgt_mask, enc_mask):
        output = tgt
        for layer in self.dec_layers:
            output = layer(output, enc, tgt_mask, enc_mask)
        return output
    
    
