import math

import torch
from matplotlib import pyplot as plt
from torch import nn
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.neck.neck_base import (TRACK_NECKS,
                                               VOS_NECKS)


def tensor2transformer(x):
    result = x.view(*x.shape[:2], -1)
    result = result.permute(2, 0, 1).contiguous()

    return result


class SpatialPositionEncodingLearned(nn.Module):
    def __init__(self, d_model, score_size):
        super(SpatialPositionEncodingLearned, self).__init__()
        self.row_embed = nn.Embedding(score_size, d_model // 2)
        self.col_embed = nn.Embedding(score_size, d_model // 2)
        self.spatial_size = score_size
        self.pos = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def get_position_encoding(self, x):
        i = torch.arange(self.spatial_size, device=x.device)
        j = torch.arange(self.spatial_size, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(self.spatial_size, 1, 1),
            y_emb.unsqueeze(1).repeat(1, self.spatial_size, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).contiguous()  # 1, C, H, W
        return pos

    def forward(self, x):
        if self.training:
            self.pos = self.get_position_encoding(x)
        else:
            if self.pos is None:
                self.pos = self.get_position_encoding(x)
        return x + self.pos


"""
TRTR Transformer class.

Copy from DETR, whish has following modification compared to original transformer (torch.nn.Transformer):
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""


class Feedforward(nn.Module):
    def __init__(self, dim_models, min_channel):
        super(Feedforward, self).__init__()

        self.w2 = nn.Linear(dim_models, min_channel)
        self.re = nn.ReLU(min_channel)
        self.w1 = nn.Linear(min_channel, dim_models)
        self.norm = nn.LayerNorm(dim_models)

    def forward(self, x):
        x1 = x
        x = self.w1(self.re(self.w2(x)))

        out = self.norm(x1 + x)
        return out


class Feedforward_dropout(nn.Module):
    def __init__(self, dim_models, min_channel, dropout=0.2):
        super(Feedforward_dropout, self).__init__()

        self.w2 = nn.Linear(dim_models, min_channel)
        self.re = nn.ReLU(min_channel)
        self.w1 = nn.Linear(min_channel, dim_models)
        self.norm = nn.LayerNorm(dim_models)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = x
        x = self.w1(self.re(self.w2(x)))
        x = self.dropout(x)
        out = self.norm(x1 + x)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, dim_models, min_channels, heads, dropout=0.2):
        super(EncoderLayer, self).__init__()
        self.MutiehadAttention = nn.MultiheadAttention(embed_dim=dim_models, num_heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim_models, eps=1e-6)
        self.Feedforward = Feedforward_dropout(dim_models, min_channels, dropout=dropout)

    def forward(self, x):
        x1 = x
        x, enc_slf_attn = self.MutiehadAttention(query=x, key=x, value=x, attn_mask=None)
        out = self.norm(x1 + x)
        out = self.Feedforward(out)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, dim_models, min_channels, heads, dropout=0.2):
        super(DecoderLayer, self).__init__()
        self.MutiehadAttention = nn.MultiheadAttention(embed_dim=dim_models, num_heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim_models, eps=1e-6)
        self.Feedforward = Feedforward_dropout(dim_models, min_channels, dropout=dropout)

    def forward(self, x, z):
        x1 = x
        x, enc_slf_attn = self.MutiehadAttention(query=x, key=x, value=x, attn_mask=None)
        out = self.norm(x1 + x)
        out1, enc_slf_attn1 = self.MutiehadAttention(query=out, key=z, value=z, attn_mask=None)
        out1 = self.norm(out1 + out)
        out2 = self.Feedforward(out1)
        return out2


class DecoderLayerX(nn.Module):
    def __init__(self, dim_models, min_channels, heads, dropout=0.2):
        super(DecoderLayerX, self).__init__()
        self.MutiehadAttention = nn.MultiheadAttention(embed_dim=dim_models, num_heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim_models, eps=1e-6)
        self.Feedforward = Feedforward_dropout(dim_models, min_channels, dropout=dropout)

    def forward(self, x, z):
        x1 = x

        x, enc_slf_attn = self.MutiehadAttention(query=x, key=x, value=x, attn_mask=None)
        out = self.norm(x1 + x)
        out1, enc_slf_attn1 = self.MutiehadAttention(query=x, key=z, value=z, attn_mask=None)
        out1 = self.norm(out1 + x1)
        out2 = self.Feedforward(out1)
        return out2


class DecoderLayer_fusion(nn.Module):
    def __init__(self, dim_models, min_channels, heads, dropout=0.2):
        super(DecoderLayer_fusion, self).__init__()
        self.MutiehadAttention = nn.MultiheadAttention(embed_dim=dim_models, num_heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim_models, eps=1e-6)
        self.Feedforward = Feedforward_dropout(dim_models, min_channels, dropout=dropout)

    def forward(self, feature_x, z_out, feature_z):
        intput = list()
        x1 = feature_x

        intput.append(feature_x)
        intput.append(feature_z)
        x = torch.cat(intput, dim=0)
        x, enc_slf_attn = self.MutiehadAttention(query=x, key=x, value=x, attn_mask=None)
        x = torch.split(x, [feature_x.shape[0], feature_z.shape[0]], dim=0)[-2]
        out = self.norm(x1 + x)
        out1, enc_slf_attn1 = self.MutiehadAttention(query=out, key=z_out, value=z_out, attn_mask=None)
        out1 = self.norm(out1 + out)
        out2 = self.Feedforward(out1)
        return out2


class Encoder(nn.Module):
    def __init__(self, mid_channels_models, min_channels=256, nheads=2, num_encoder_layer=2,dropout=0.1):
        super(Encoder, self).__init__()

        self.encode = nn.ModuleList(
            [EncoderLayer(mid_channels_models, min_channels, nheads,dropout) for _ in range(num_encoder_layer)])

    def forward(self, x):
        for enc_layer in self.encode:
            x = enc_layer(x)

        return x


class Decoder(nn.Module):
    def __init__(self, mid_channels_models, min_channels=256, nheads=2, num_decoder_layer=2,dropout=0.1):
        super().__init__()

        self.decode = nn.ModuleList(
            [DecoderLayer(mid_channels_models, min_channels, nheads,dropout) for _ in range(num_decoder_layer)])

    def forward(self, x, z):
        for decode in self.decode:
            x = decode(x, z)
        return x


class Decoder_fusion(nn.Module):
    def __init__(self, mid_channels_models, min_channels=256, nheads=2, num_decoder_layer=2,dropout=0.1):
        super().__init__()

        self.decode = nn.ModuleList(
            [DecoderLayer_fusion(mid_channels_models, min_channels, nheads,dropout) for _ in range(num_decoder_layer)])

    def forward(self, feature_x, z_out, feature_z):
        for decode in self.decode:
            feature_x = decode(feature_x, z_out, feature_z)
        return feature_x


@TRACK_NECKS.register
class Transformer(ModuleBase):
    default_hyper_params = dict(
        d_model=96,
        nhead=2,
        score_size_z=127,
        score_size_x=289,
        min_channels=256
    )

    def __init__(self):
        super().__init__()

    def update_params(self):
        super().update_params()
        self.score_size_z = self._hyper_params['score_size_z']
        self.score_size_x = self._hyper_params['score_size_x']
        self.d_model = self._hyper_params['d_model']
        self.d_model = self._hyper_params['d_model']
        self.nhead = self._hyper_params['nhead']
        self.pos_emb_z = SpatialPositionEncodingLearned(self.d_model, self.score_size_z)
        self.pos_emb_x = SpatialPositionEncodingLearned(self.d_model, self.score_size_x)
        self.pos_emb_en = SpatialPositionEncodingLearned(self.d_model * 2, self.score_size_x)
        self.min_channels = self._hyper_params['min_channels']
        self.encoder = Encoder(mid_channels_models=self.d_model, min_channels=self.min_channels)
        self.decoder = Decoder(mid_channels_models=self.d_model, min_channels=self.min_channels)

        self.enhance_encode = Encoder(mid_channels_models=self.d_model * 2, min_channels=self.min_channels)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def decode(self, f_x, enc_output):
        dec_output = self.decoder(f_x, enc_output)

        return dec_output

    def forward(self, feature_x, feature_z):
        B, C, H, W = feature_x.shape
        feature_z = self.pos_emb_z(feature_z)
        feature_z = tensor2transformer(feature_z)
        z_out = self.encoder(feature_z)
        x2 = feature_x
        feature_x = self.pos_emb_x(feature_x)
        feature_x = tensor2transformer(feature_x)
        final_output = self.decode(feature_x, z_out)
        dec_output = final_output.permute(1, 2, 0).contiguous()  # B, C, HW
        final_output = dec_output.view(*dec_output.shape[:2], H, W)
        final_output = torch.cat([final_output, x2], dim=1)
        B, C, H, W = final_output.shape
        final_output = self.pos_emb_en(final_output)
        final_output = tensor2transformer(final_output)
        final_output_enhance = self.enhance_encode(final_output)
        final_output_enhance = final_output_enhance.view(H, W, B, -1).permute(2, 3, 0, 1)
        return final_output_enhance


@TRACK_NECKS.register
class Transformer_fusion(ModuleBase):
    default_hyper_params = dict(
        d_model=96,
        nhead=2,
        score_size_z=127,
        score_size_x=289,
        num_encoder_layer=1,
        num_decoder_layer=1,
        Feedforward_min_channels=256,
        dropout=0.1
    )

    def __init__(self):
        super().__init__()

    def update_params(self):
        super().update_params()
        self.score_size_z = self._hyper_params['score_size_z']
        self.score_size_x = self._hyper_params['score_size_x']
        self.d_model = self._hyper_params['d_model']
        self.nhead = self._hyper_params['nhead']
        self.pos_emb_z = SpatialPositionEncodingLearned(self.d_model, self.score_size_z)
        self.pos_emb_x = SpatialPositionEncodingLearned(self.d_model, self.score_size_x)

        # self.encoder_fusion = Encoder(mid_channels_models=self.d_model,num_encoder_layer=self._hyper_params['num_encoder_layer'])
        self.encoder = Encoder(mid_channels_models=self.d_model,
                               num_encoder_layer=self._hyper_params['num_encoder_layer'],
                               dropout=self._hyper_params['dropout'])
        self.decoder = Decoder_fusion(mid_channels_models=self.d_model,
                                      min_channels=self._hyper_params['Feedforward_min_channels'],
                                      num_decoder_layer=self._hyper_params['num_encoder_layer'],
                                      dropout=self._hyper_params['dropout'])

        self.enhance_encode = Encoder(mid_channels_models=self.d_model * 2,
                                      min_channels=self._hyper_params['Feedforward_min_channels'] * 2,
                                      nheads=self._hyper_params['nhead']* 2,
                                      num_encoder_layer=self._hyper_params['num_encoder_layer'],
                                      dropout=self._hyper_params['dropout'])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def decode(self, feature_x, z_out, feature_z):

        dec_output = self.decoder(feature_x, z_out, feature_z)
        final_output = torch.cat([dec_output, feature_x], dim=2)
        return final_output

    def forward(self, feature_x, feature_z):

        B, C, H, W = feature_x.shape
        feature_z = self.pos_emb_z(feature_z)
        feature_z = tensor2transformer(feature_z)

        feature_x = self.pos_emb_x(feature_x)
        feature_x = tensor2transformer(feature_x)

        z_out = self.encoder(feature_z)
        final_output = self.decode(feature_x, z_out, feature_z)

        final_output_enhance = self.enhance_encode(final_output)
        final_output_enhance = final_output_enhance.view(H, W, B, -1).permute(2, 3, 0, 1)

        return final_output_enhance


@TRACK_NECKS.register
class Transformer_fusionX(ModuleBase):
    default_hyper_params = dict(
        d_model=96,
        nhead=2,
        score_size_z=127,
        score_size_x=289,
        num_encoder_layer=1,
        num_decoder_layer=1,
        Feedforward_min_channels=256,
        dropout=0.1
    )

    def __init__(self):
        super().__init__()

    def update_params(self):
        super().update_params()
        self.score_size_z = self._hyper_params['score_size_z']
        self.score_size_x = self._hyper_params['score_size_x']
        self.d_model = self._hyper_params['d_model']
        self.nhead = self._hyper_params['nhead']
        self.pos_emb_z = SpatialPositionEncodingLearned(self.d_model, self.score_size_z)
        self.pos_emb_x = SpatialPositionEncodingLearned(self.d_model, self.score_size_x)

        # self.encoder_fusion = Encoder(mid_channels_models=self.d_model,num_encoder_layer=self._hyper_params['num_encoder_layer'])
        self.encoder = Encoder(mid_channels_models=self.d_model,
                               min_channels=self._hyper_params['Feedforward_min_channels'],
                               num_encoder_layer=self._hyper_params['num_encoder_layer'],
                               dropout=self._hyper_params['dropout'])
        # self.decoder = Decoder_fusion(mid_channels_models=self.d_model,
        #                               min_channels=self._hyper_params['Feedforward_min_channels'],
        #                               num_decoder_layer=self._hyper_params['num_encoder_layer'])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # def decode(self, feature_x, z_out, feature_z):
    #
    #     dec_output = self.decoder(feature_x, z_out, feature_z)
    #     final_output = torch.cat([dec_output, feature_x], dim=2)
    #     return final_output

    def forward(self, feature_x, feature_z):
        input = list()
        B, C, H, W = feature_x.shape
        feature_z = self.pos_emb_z(feature_z)
        feature_z = tensor2transformer(feature_z)

        feature_x = self.pos_emb_x(feature_x)
        feature_x = tensor2transformer(feature_x)
        input.append(feature_z)
        input.append(feature_x)
        dim_out = torch.cat(input, dim=0)
        dim_out = self.encoder(dim_out)
        out = torch.split(dim_out, [feature_x.shape[0], feature_z.shape[0]], dim=0)[-2]
        out = torch.cat([out, feature_x], dim=2)
        HW, B, C = out.shape
        out = out.permute(1, 2, 0).contiguous()
        out = out.view(B, C, H, W)

        return out


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
