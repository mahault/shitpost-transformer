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