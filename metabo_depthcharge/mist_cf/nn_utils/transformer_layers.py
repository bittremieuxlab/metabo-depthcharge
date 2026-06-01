# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""transformer_layers.py

Pairwise attention enabled transformers.
"""

import math

import torch
from torch.nn import Dropout, LayerNorm, Linear, Module, Parameter
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear


class TransformerEncoderLayer(Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=False,
        additive_attn=False,
        pairwise_featurization=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.pairwise_featurization = pairwise_featurization
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            additive_attn=additive_attn,
            pairwise_featurization=self.pairwise_featurization,
            **factory_kwargs,
        )
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = activation

    def forward(self, src, pairwise_features=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), pairwise_features, src_key_padding_mask
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, pairwise_features, src_key_padding_mask)
            )
            x = self.norm2(x + self._ff_block(x))
        return x, pairwise_features

    def _sa_block(self, x, pairwise_features, key_padding_mask):
        x = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            pairwise_features=pairwise_features,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class MultiheadAttention(Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        additive_attn=False,
        pairwise_featurization=False,
        dropout=0.0,
        batch_first=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self.additive_attn = additive_attn
        self.pairwise_featurization = pairwise_featurization

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim

        if self.additive_attn:
            head_1_input = (
                self.head_dim * 3 if self.pairwise_featurization else self.head_dim * 2
            )
            self.attn_weight_1_weight = Parameter(
                torch.empty(
                    (self.num_heads, head_1_input, self.head_dim), **factory_kwargs
                )
            )
            self.attn_weight_1_bias = Parameter(
                torch.empty((self.num_heads, self.head_dim), **factory_kwargs)
            )
            self.attn_weight_2_weight = Parameter(
                torch.empty((self.num_heads, self.head_dim, 1), **factory_kwargs)
            )
            self.attn_weight_2_bias = Parameter(
                torch.empty((self.num_heads, 1), **factory_kwargs)
            )
        else:
            if self.pairwise_featurization:
                self.bias_u = Parameter(
                    torch.empty((self.num_heads, self.head_dim), **factory_kwargs)
                )
                self.bias_v = Parameter(
                    torch.empty((self.num_heads, self.head_dim), **factory_kwargs)
                )

        self.in_proj_weight = Parameter(
            torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
        )
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=True, **factory_kwargs
        )
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)
        if self.additive_attn:
            xavier_uniform_(self.attn_weight_1_weight)
            xavier_uniform_(self.attn_weight_2_weight)
            constant_(self.attn_weight_1_bias, 0.0)
            constant_(self.attn_weight_2_bias, 0.0)
        else:
            if self.pairwise_featurization:
                constant_(self.bias_u, 0.0)
                constant_(self.bias_v, 0.0)

    def forward(self, query, key, value, key_padding_mask=None, pairwise_features=None):
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        attn_output, attn_output_weights = self._multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            pairwise_features=pairwise_features,
        )

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def _multi_head_attention_forward(
        self,
        query,
        key,
        value,
        embed_dim_to_check,
        num_heads,
        in_proj_weight,
        in_proj_bias,
        dropout_p,
        out_proj_weight,
        out_proj_bias,
        training=True,
        key_padding_mask=None,
        pairwise_features=None,
    ):
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        head_dim = embed_dim // num_heads

        q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

        if pairwise_features is not None:
            pairwise_features = pairwise_features.permute(1, 2, 0, 3).contiguous()
            pairwise_features = pairwise_features.view(
                tgt_len, tgt_len, bsz * num_heads, head_dim
            )
            pairwise_features = pairwise_features.permute(2, 0, 1, 3)

        src_len = k.size(1)

        attn_mask = None
        if key_padding_mask is not None:
            key_padding_mask = (
                key_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, num_heads, -1, -1)
                .reshape(bsz * num_heads, 1, src_len)
            )
            attn_mask = key_padding_mask

        if not training:
            dropout_p = 0.0

        if self.additive_attn:
            attn_output, attn_output_weights = self._additive_attn(
                q, k, v, attn_mask, dropout_p, pairwise_features=pairwise_features
            )
        else:
            attn_output, attn_output_weights = self._scaled_dot_product_attention(
                q, k, v, attn_mask, dropout_p, pairwise_features=pairwise_features
            )

        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        )
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights

    def _additive_attn(
        self, q, k, v, attn_mask=None, dropout_p=0.0, pairwise_features=None
    ):
        B, Nt, E = q.shape
        q_expand = q[:, :, None, :].expand(B, Nt, Nt, E)
        v_expand = v[:, None, :, :].expand(B, Nt, Nt, E)
        cat_ar = [q_expand, v_expand]
        if pairwise_features is not None:
            cat_ar.append(pairwise_features)

        output = torch.cat(cat_ar, -1)
        E_long = E * len(cat_ar)

        output = output.view(-1, self.num_heads, Nt, Nt, E_long)
        output = torch.einsum("bnlwe,neh->bnlwh", output, self.attn_weight_1_weight)
        output = output + self.attn_weight_1_bias[None, :, None, None, :]
        output = F.leaky_relu(output)

        attn = torch.einsum("bnlwh,nhi->bnlwi", output, self.attn_weight_2_weight)
        attn = attn + self.attn_weight_2_bias[None, :, None, None, :]
        attn = attn.contiguous().view(-1, Nt, Nt)
        if attn_mask is not None:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn += attn_mask
        attn = F.softmax(attn, dim=-1)
        output = torch.bmm(attn, v)
        return output, attn

    def _scaled_dot_product_attention(
        self, q, k, v, attn_mask=None, dropout_p=0.0, pairwise_features=None
    ):
        B, Nt, E = q.shape
        q = q / math.sqrt(E)

        if self.pairwise_featurization:
            if pairwise_features is None:
                raise ValueError()

            q = q.view(-1, self.num_heads, Nt, E)
            q_1 = q + self.bias_u[None, :, None, :]
            q_2 = q + self.bias_v[None, :, None, :]

            q_1 = q_1.view(-1, Nt, E)
            q_2 = q_2.view(-1, Nt, E)

            a_c = torch.einsum("ble,bwe->blw", q_1, k)
            b_d = torch.einsum("ble,blwe->blw", q_2, pairwise_features)
            attn = a_c + b_d
        else:
            attn = torch.bmm(q, k.transpose(-2, -1))

        if attn_mask is not None:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn += attn_mask

        attn = F.softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)
        output = torch.bmm(attn, v)
        return output, attn
