import torch
import torch.nn as nn
import math
from einops import rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        x_dim,
        external_cond_dim=0,
        size=128,
        k_emb_size=64,
        num_layers=4,
        nhead=4,
        dim_feedforward=512,
        cond_mask_prob=0.0,
        dropout=0.0,
        **kwargs,
    ):
        """
        Transformer Encoder Architecture
        Joint (state, action) token
        """
        super(TransformerEncoder, self).__init__()
        self.external_cond_dim = external_cond_dim
        self.cond_mask_prob = cond_mask_prob
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        k_embed_dim = k_emb_size
        self.t_embed = SinusoidalPosEmb(dim=size)
        self.k_embed = SinusoidalPosEmb(dim=k_embed_dim)
        self.text_embed = nn.Linear(512, size)
        self.init_mlp = nn.Sequential(
            nn.Linear(x_dim + k_embed_dim, size),
            nn.ReLU(),
            nn.Linear(size, size),
        )
        self.out = nn.Linear(size, x_dim)

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x, k, external_cond=None, force_mask=False, is_causal=False):
        # x.shape (T, B, C), float32
        # k.shape (T, B), int64
        # optional external_cond.shape (T, B, C)
        seq_len, batch_size, _ = x.shape  ## t, b, d
        k_embed = rearrange(self.k_embed(k.flatten()), "(t b) d -> t b d", t=seq_len)   ## add position embedding to the flattened sampled noise level???
        x = torch.cat((x, k_embed), dim=-1)
        x = self.init_mlp(x)
        if external_cond is not None:
            assert self.external_cond_dim > 0
            cond = self.mask_cond(self.text_embed(external_cond), force_mask)
            x = torch.cat((cond.unsqueeze(0), x), dim=0)
        
        x = x + self.t_embed(torch.arange(seq_len if external_cond is None else seq_len+1, device=x.device)[:, None])

        mask = nn.Transformer.generate_square_subsequent_mask(len(x), x.device) if is_causal else None
        x = self.transformer(x, mask=mask, is_causal=is_causal)
        x = self.out(x)

        if external_cond is not None:
            x = x[1:]

        return x

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        x_dim,
        external_cond_dim=0,
        size=128,
        k_emb_size=64,
        num_layers=4,
        nhead=4,
        dim_feedforward=512,
        cond_mask_prob=0.0,
        dropout=0.0,
        **kwargs,
    ):
        """
        Transformer Decoder Architecture
        Joint (state, action) token
        """
        super(TransformerDecoder, self).__init__()
        self.external_cond_dim = external_cond_dim
        self.cond_mask_prob = cond_mask_prob
        self.size = size
        encoder_layer = nn.TransformerDecoderLayer(
            d_model=size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer = nn.TransformerDecoder(encoder_layer, num_layers=num_layers)
        k_embed_dim = k_emb_size #size // 2
        self.t_embed = SinusoidalPosEmb(dim=size)
        self.k_embed = SinusoidalPosEmb(dim=k_embed_dim)
        self.text_embed = nn.Linear(512, size)
        self.init_mlp = nn.Sequential(
            nn.Linear(x_dim + k_embed_dim, size),
            nn.ReLU(),
            nn.Linear(size, size),
        )
        self.out = nn.Linear(size, x_dim)

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x, k, external_cond=None, force_mask=False, is_causal=False):
        # x.shape (T, B, C), float32
        # k.shape (T, B), int64
        # optional external_cond.shape (T, B, C)
        seq_len, batch_size, _ = x.shape  ## t, b, d
        k_embed = rearrange(self.k_embed(k.flatten()), "(t b) d -> t b d", t=seq_len)   ## add position embedding to the flattened sampled noise level???
        x = torch.cat((x, k_embed), dim=-1)
        x = self.init_mlp(x)
        if external_cond is not None:
            if external_cond.ndim == 1:
                external_cond = external_cond.unsqueeze(0)
            cond = self.mask_cond(self.text_embed(external_cond), force_mask)

            if external_cond.shape[-1] == self.size:
                cond = self.mask_cond(external_cond, force_mask)
            cond = cond.unsqueeze(0).expand(seq_len, batch_size, -1)
        else:
            cond = None
        
        x = x + self.t_embed(torch.arange(seq_len, device=x.device)[:, None])

        mask = nn.Transformer.generate_square_subsequent_mask(len(x), x.device) if is_causal else None

        if cond is not None:
            x = self.transformer(x, cond, tgt_mask=mask)
        else:
            for layer in self.transformer.layers:
                x = layer(x, memory=None, tgt_mask=mask)

        x = self.out(x)

        return x
    

