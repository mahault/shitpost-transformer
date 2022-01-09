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
    
class EncoderLayer(nn.module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.3):
        super().__init__()
        #attention
        self.attn = MultiHeadedAttention(d_model, num_heads, dropout=dropout)
        #ffn
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # layer norm
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        
    def forward(self, src, src_mask):
        x = src
        x = x+ self.attn(q=x, k=x, v=x, mask=src_mask)
        x = self.attn_norm(x)
        x = self.ffn(x)
        x = self.ffn_norm(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.3):
        super().__init__()
        
        #masked attn
        self.masked_attn = MultiHeadedAttention(
            d_model, num_heads, dropout=dropout
        )
        
        #attn
        self.attn = MultiHeadedAttention(
            d_model, num_heads, dropout=dropout
        )
        
        #ffn
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        #layer norm
        self.masked_attn_norm = nn.LayerNorm(d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = mm.LayerNorm(d_model)
        
    def forward(self, tgt, enc, tgt_mask, enc_mask):
        x = tgt
        x = x + self.masked_attn(q=x, k=x, v=x, mask=tgt_mask)
        x = self.masked_attn_norm(x)
        x = x + self.attn(q=x, k=enc, v=enc, mask=enc_mask)
        x = self.attn_norm(x)
        x = x + self.ffn(x)
        x = self.ffn_norm(x)
        return 
    
    if mask is not None:
        expanded_mask = mask[:, None, :].expand(bs, tgt_len, seq_len)
        susequent_mask = 1 - torch.triu(
            torch.ones((tgt_len, tgt_len), device=mask.device, dtype=torch.uint8), diagonal=1
        )
        subsequent_mask = subsequent_mask[None, :, :].expand(bs, tgt_len, tgt_len)
        scores = scores.masked_fill(expanded_mask == 0, -float("Inf"))
        scores = scores.masked_fill(subsequent_mask == 0, -float("Inf"))


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn_output_size = self.d_model // self.num_heads
        self.attentions = nn.ModuleList(
            [
                SelfAttention(d_model, self.attn_output_size)
                for _ in range(self.num_heads)
            ],
        )
        self.output = nn.Linear(self.d_model, self.d_model)
    
    def forward(self, q, k, v, mask):
        x = torch.cat(
            [
                layer(q,k,v,mask) for layer in self.attentions
            ],dim=-1
        )
        x = self.output(x)
        return x
        
class SelfAttention(nn.Module):
    def __init__(self, d_model, output_size, dropout=0.3):
        super().__init__()
        self.query = nn.Linear(d_model, output_size)
        self.key = nn.Linear(d_model, output_size)
        self.value = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        bs = q.shape[0]
        tgt_len = q.shape[1]
        seq_len = k.shape[1]
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)
        
        dim_k = key.size(-1)
        scores = torch.bmm(query, key.transpose(1,2)) / nq.sqrt(dim_k)
        
        if mask is not None:
            expanded_mask = mask[:, None, :].expand(bs, tgt_len, seq_len)
            scores = scores.masked_fill(expand_mask == 0, -float("Inf"))
        
        weights = F.softmax(scores, dim=-1)
        outputs = torch.bmm(weights, value)
        return outputs
    
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.) / d_model))  # Need to look into this
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0)])
    
    

trans = Transformer()
trans.forward()