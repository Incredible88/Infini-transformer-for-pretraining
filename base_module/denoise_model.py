#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : MetaIndex 
# @File    : denoise_model.py
# @Author  : zhangchao
# @Date    : 2024/7/23 11:40 
# @Email   : zhangchao5@genomics.cn
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_head, active_fn, dropout, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dim
        self.n_head = n_head
        assert embed_dim % n_head == 0, '`embed_dim` must be divisible by `n_head`!'
        self.depth = hidden_dim // n_head
        self.scale = hidden_dim ** -0.5

        self.w_q = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, hidden_dim, bias=False)

        self.w_out = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(p=dropout),
            active_fn(),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, context=None, attention_mask=None):
        residual = x
        B, L, C = x.size()

        if context is None:
            context = x
        if attention_mask is None:
            attention_mask = torch.ones((B, L), device=x.device, dtype=torch.bool)

        q = self.w_q(x).view(B, -1, self.n_head, self.depth).transpose(1, 2)
        k = self.w_k(context).view(B, -1, self.n_head, self.depth).transpose(1, 2)
        v = self.w_v(context).view(B, -1, self.n_head, self.depth).transpose(1, 2)

        # attention weight
        attn_weight = torch.einsum('bhid, bhjd -> bhij', q, k)
        attn_weight *= self.scale
        attn_weight = attn_weight.masked_fill((1 - attention_mask[:, None, None, :]).bool(), 1e-9)
        attn_weight = torch.nn.functional.softmax(attn_weight, dim=-1)
        out = torch.einsum('bhij, bhjd -> bhid', attn_weight, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_head * self.depth)
        out = self.w_out(out)
        return self.layer_norm(out + residual)


class Layer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_head, active_fn, dropout, **kwargs):
        super().__init__()
        self.attention = Attention(embed_dim, hidden_dim, n_head, active_fn, dropout)
        self.dense = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
            active_fn()
        )
        self.out_layer = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, attention_mask=None):
        x = self.attention(x, context, attention_mask)
        out = self.dense(x)
        return self.dropout(self.out_layer(out + x))


class Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_head, active_fn, dropout, nun_layer, **kwargs):
        super().__init__()
        self.net = nn.ModuleList()
        for _ in range(nun_layer):
            self.net.append(
                Layer(embed_dim, hidden_dim, n_head, active_fn, dropout)
            )
        self.final_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, context=None, attention_mask=None):
        for idx, ly in enumerate(self.net):
            x = ly(x, context, attention_mask)
        return self.final_layer(x)


class DenoiseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, embed_x1, embed_x2, t, **kwargs):
        raise NotImplementedError
