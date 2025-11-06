from typing import OrderedDict
import torch.nn as nn
import math

import torch

import sys
import os

from matplotlib import pyplot as plt

from videoanalyst.model.fusion.fusion_base import TRACK_FUSIONS

env_path = os.path.join(os.path.dirname(__file__), '/home/xlsun/xlsun/code/SparseTT')
if env_path not in sys.path:
    sys.path.append(env_path)

from videoanalyst.model.backbone.backbone_base import (TRACK_BACKBONES,
                                                       VOS_BACKBONES)
from videoanalyst.model.common_opr.common_block import conv_bn_relu
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.neck.neck_impl.encoder_enhance import Encoder, Encoder_siamtpn, Encoder_l42


from videoanalyst.model.utils.transformer_layers import (SpatialPositionEncodingLearned,
                                                         MultiHeadAttention,
                                                         PositionWiseFeedForward)
from videoanalyst.model.neck.neck_impl.encoder_enhance import Encoder



@TRACK_FUSIONS.register
class FusionXX(ModuleBase):
    default_hyper_params = dict(
        input_channel=96,
        input_channelX=96,
        feat_fx_size=[8, 8, 4],
        feat_fz_size=[19, 19, 10],
        used_layers=[6, 8, 11],
        num_heads=3,
        num_layers=2,
        prob_dropout=0.1,

    )

    def __init__(self):
        super(FusionXX, self).__init__()
        self.position_embeeding_fx_dict = nn.ModuleDict()
        self.position_embeeding_fz_dict = nn.ModuleDict()
        self.used_layers = None

    def update_params(self):
        super().update_params()

        self.encoder1 = Encoder(mid_channels_model=self._hyper_params['input_channel'],
                                mid_channels_ffn=self._hyper_params['input_channel'] * 4,
                                num_heads=self._hyper_params['num_heads'],
                                num_layers=self._hyper_params['num_layers'],
                                prob_dropout=self._hyper_params['prob_dropout'], )
        self.encoder2 = Encoder(mid_channels_model=self._hyper_params['input_channel'],
                                mid_channels_ffn=self._hyper_params['input_channel'] * 4,
                                num_heads=self._hyper_params['num_heads'],
                                num_layers=self._hyper_params['num_layers'],
                                prob_dropout=self._hyper_params['prob_dropout'], )
        self.used_layers = self._hyper_params['used_layers']
        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fx_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fx_size'][idx])

        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fz_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fz_size'][idx])

        self.encoderX = Encoder(mid_channels_model=self._hyper_params['input_channelX'],
                                mid_channels_ffn=self._hyper_params['input_channelX'] * 4,
                                num_heads=self._hyper_params['num_heads'],
                                num_layers=self._hyper_params['num_layers'],
                                prob_dropout=self._hyper_params['prob_dropout'], )

    def forward(self, x):
        cat_results = list()
        shapes = list()
        cat_results1 = list()
        cat_results2 = list()
        cat_results3 = list()
        cat_out = list()

        for idx in range(len(x)):
            shapes.append(x[idx].shape[-2])
        self.position_embeeding = self.position_embeeding_fx_dict if shapes[
                                                                         -2] == 8 else self.position_embeeding_fz_dict
        for idx, result in enumerate(x):
            result = self.position_embeeding[str(self.used_layers[idx])](result)
            result = result.view(*result.shape[:2], -1)  # B, C, HW
            result = result.permute(2, 0, 1).contiguous()  # HW, B, C
            cat_results.append(result)

        for i, result in enumerate(cat_results):
            if i <= 1:
                cat_results1.append(result)
            if i >= 1:
                cat_results2.append(result)

        cat_out.append(torch.cat(cat_results1, dim=0))
        cat_out.append(torch.cat(cat_results2, dim=0))

        # cat_input = torch.cat(cat_results, dim=0)
        ##############################################################
        enc_output, cat_out1 = self.encoder1(cat_out[0])

        output1 = torch.split(cat_out1, [shapes[0] ** 2, shapes[1] ** 2], dim=0)[-1]
        ##############################################################
        enc_output, cat_out2 = self.encoder2(cat_out[1])

        output2 = torch.split(cat_out2, [shapes[1] ** 2, shapes[2] ** 2], dim=0)[-2]
        ##############################################################
        cat_results3.append(output1)
        cat_results3.append(output2)
        min_out = torch.cat(cat_results3, dim=0)
        enc_output, min_out = self.encoderX(min_out)


        output = torch.split(min_out, [shapes[1] ** 2, shapes[1] ** 2], dim=0)[0]
        # out = (output[0] + output[1]) / 2

        L, B, C = output.shape
        out = enc_output.view(shapes[-2], shapes[-2], B, C).permute(2, 3, 0, 1).contiguous()

        return out

@TRACK_FUSIONS.register
class Fusion_v3(ModuleBase):
    default_hyper_params = dict(
        input_channel=96,
        input_channelX=96,
        feat_fx_size=[8, 8, 4],
        feat_fz_size=[19, 19, 10],
        used_layers=[6, 8, 11],
        num_heads=3,
        num_layers=2,
        prob_dropout=0.1,

    )

    def __init__(self):
        super(Fusion_v3, self).__init__()
        self.position_embeeding_fx_dict = nn.ModuleDict()
        self.position_embeeding_fz_dict = nn.ModuleDict()
        self.used_layers = None

    def update_params(self):
        super().update_params()

        self.encoder1 = Encoder(mid_channels_model=self._hyper_params['input_channel'],
                                mid_channels_ffn=self._hyper_params['input_channel'] * 4,
                                num_heads=self._hyper_params['num_heads'],
                                num_layers=self._hyper_params['num_layers'],
                                prob_dropout=self._hyper_params['prob_dropout'], )
        self.encoder2 = Encoder(mid_channels_model=self._hyper_params['input_channel'],
                                mid_channels_ffn=self._hyper_params['input_channel'] * 4,
                                num_heads=self._hyper_params['num_heads'],
                                num_layers=self._hyper_params['num_layers'],
                                prob_dropout=self._hyper_params['prob_dropout'], )
        self.used_layers = self._hyper_params['used_layers']
        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fx_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fx_size'][idx])

        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fz_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fz_size'][idx])

        self.encoderX = Encoder(mid_channels_model=self._hyper_params['input_channelX'],
                                mid_channels_ffn=self._hyper_params['input_channelX'] * 4,
                                num_heads=self._hyper_params['num_heads'],
                                num_layers=self._hyper_params['num_layers'],
                                prob_dropout=self._hyper_params['prob_dropout'], )

    def forward(self, x):
        cat_results = list()
        shapes = list()
        cat_results1 = list()
        cat_results2 = list()
        cat_results3 = list()
        cat_out = list()

        for idx in range(len(x)):
            shapes.append(x[idx].shape[-2])
        self.position_embeeding = self.position_embeeding_fx_dict if shapes[
                                                                         -2] == 8 else self.position_embeeding_fz_dict
        for idx, result in enumerate(x):
            result = self.position_embeeding[str(self.used_layers[idx])](result)
            result = result.view(*result.shape[:2], -1)  # B, C, HW
            result = result.permute(2, 0, 1).contiguous()  # HW, B, C
            cat_results.append(result)

        for i, result in enumerate(cat_results):
            if i <= 1:
                cat_results1.append(result)
            if i >= 1:
                cat_results2.append(result)

        cat_out.append(torch.cat(cat_results1, dim=0))
        cat_out.append(torch.cat(cat_results2, dim=0))

        # cat_input = torch.cat(cat_results, dim=0)
        ##############################################################
        enc_output, cat_out1 = self.encoder1(cat_out[0])

        output1 = torch.split(enc_output, [shapes[0] ** 2, shapes[1] ** 2], dim=0)[-1]
        ##############################################################
        enc_output, cat_out2 = self.encoder2(cat_out[1])

        output2 = torch.split(enc_output, [shapes[1] ** 2, shapes[2] ** 2], dim=0)[-2]
        ##############################################################
        cat_results3.append(output1)
        cat_results3.append(output2)
        min_out = torch.cat(cat_results3, dim=2)
        enc_output, min_out = self.encoderX(min_out)


        L, B, C = enc_output.shape
        out = enc_output.view(shapes[-2], shapes[-2], B, C).permute(2, 3, 0, 1).contiguous()

        return out

@TRACK_FUSIONS.register
class FusionXXX(ModuleBase):
    default_hyper_params = dict(
        input_channel=96,
        input_channelX=96,
        feat_fx_size=[8, 8, 4],
        feat_fz_size=[19, 19, 10],
        used_layers=[6, 8, 11],
        num_heads=3,
        num_layers=2,
        prob_dropout=0.1,

    )

    def __init__(self):
        super(FusionXXX, self).__init__()
        self.position_embeeding_fx_dict = nn.ModuleDict()
        self.position_embeeding_fz_dict = nn.ModuleDict()
        self.used_layers = None

    def update_params(self):
        super().update_params()

        self.encoder1 = Encoder(mid_channels_model=self._hyper_params['input_channel'],
                                mid_channels_ffn=self._hyper_params['input_channel'] * 4,
                                num_heads=self._hyper_params['num_heads'],
                                num_layers=self._hyper_params['num_layers'],
                                prob_dropout=self._hyper_params['prob_dropout'], )
        self.encoder2 = Encoder(mid_channels_model=self._hyper_params['input_channel'] * 2,
                                mid_channels_ffn=self._hyper_params['input_channel'] * 8,
                                num_heads=self._hyper_params['num_heads'],
                                num_layers=self._hyper_params['num_layers'],
                                prob_dropout=self._hyper_params['prob_dropout'], )
        self.used_layers = self._hyper_params['used_layers']
        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fx_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fx_size'][idx])

        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fz_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fz_size'][idx])


    def forward(self, x):
        cat_results = list()
        shapes = list()

        for idx in range(len(x)):
            shapes.append(x[idx].shape[-2])
        self.position_embeeding = self.position_embeeding_fx_dict if shapes[
                                                                         -2] == 8 else self.position_embeeding_fz_dict
        for idx, result in enumerate(x):
            result = self.position_embeeding[str(self.used_layers[idx])](result)
            result = result.view(*result.shape[:2], -1)  # B, C, HW
            result = result.permute(2, 0, 1).contiguous()  # HW, B, C
            cat_results.append(result)

        result = torch.cat(cat_results, dim=0)

        # cat_input = torch.cat(cat_results, dim=0)
        ##############################################################
        enc_output, cat_out1 = self.encoder1(result)

        # output1 = torch.split(cat_out1, [shapes[0] ** 2, shapes[1] ** 2,shapes[2] ** 2], dim=0)[-2]
        ##############################################################
        enc_output, cat_out2 = self.encoder2(cat_out1)

        output = torch.split(cat_out2, [shapes[0] ** 2, shapes[1] ** 2,shapes[2] ** 2], dim=0)[-2]


        L, B, C = output.shape
        out = output.view(shapes[-2], shapes[-2], B, C).permute(2, 3, 0, 1).contiguous()

        return out

@TRACK_FUSIONS.register
class FusionX(ModuleBase):
    default_hyper_params = dict(
        input_channel=96,
        input_channelX=96,
        feat_fx_size=[8, 8, 4],
        feat_fz_size=[19, 19, 10],
        used_layers=[6, 8, 11],
        num_heads=3,
        num_layers=2,
        prob_dropout=0.1,

    )

    def __init__(self):
        super(FusionX, self).__init__()
        self.position_embeeding_fx_dict = nn.ModuleDict()
        self.position_embeeding_fz_dict = nn.ModuleDict()
        self.used_layers = None

    def update_params(self):
        super().update_params()

        self.encoder1 = Encoder(mid_channels_model=self._hyper_params['input_channel'],
                                mid_channels_ffn=self._hyper_params['input_channel'] * 4,
                                num_heads=self._hyper_params['num_heads'],
                                num_layers=self._hyper_params['num_layers'],
                                prob_dropout=self._hyper_params['prob_dropout'], )
        self.encoder2 = Encoder(mid_channels_model=self._hyper_params['input_channel'],
                                mid_channels_ffn=self._hyper_params['input_channel'] * 4,
                                num_heads=self._hyper_params['num_heads'],
                                num_layers=self._hyper_params['num_layers'],
                                prob_dropout=self._hyper_params['prob_dropout'], )
        self.used_layers = self._hyper_params['used_layers']
        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fx_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fx_size'][idx])

        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fz_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fz_size'][idx])

        self.encoderX = Encoder(mid_channels_model=self._hyper_params['input_channelX'],
                                mid_channels_ffn=self._hyper_params['input_channelX'] * 4,
                                num_heads=self._hyper_params['num_heads'],
                                num_layers=self._hyper_params['num_layers'],
                                prob_dropout=self._hyper_params['prob_dropout'], )

    def forward(self, x):
        cat_results = list()
        shapes = list()
        cat_results1 = list()
        cat_results2 = list()
        cat_results3 = list()
        cat_out = list()

        for idx in range(len(x)):
            shapes.append(x[idx].shape[-2])
        self.position_embeeding = self.position_embeeding_fx_dict if shapes[
                                                                         -2] == 8 else self.position_embeeding_fz_dict
        for idx, result in enumerate(x):
            result = self.position_embeeding[str(self.used_layers[idx])](result)
            result = result.view(*result.shape[:2], -1)  # B, C, HW
            result = result.permute(2, 0, 1).contiguous()  # HW, B, C
            cat_results.append(result)

        for i, result in enumerate(cat_results):
            if i <= 1:
                cat_results1.append(result)
            if i >= 1:
                cat_results2.append(result)

        cat_out.append(torch.cat(cat_results1, dim=0))
        cat_out.append(torch.cat(cat_results2, dim=0))

        # cat_input = torch.cat(cat_results, dim=0)
        ##############################################################
        enc_output, cat_out1 = self.encoder1(cat_out[0])

        output1 = torch.split(cat_out1, [shapes[0] ** 2, shapes[1] ** 2], dim=0)[-1]
        ##############################################################
        enc_output, cat_out2 = self.encoder2(cat_out[1])

        output2 = torch.split(cat_out2, [shapes[1] ** 2, shapes[2] ** 2], dim=0)[-2]
        ##############################################################
        cat_results3.append(output1)
        cat_results3.append(output2)
        min_out = torch.cat(cat_results3, dim=2)
        enc_output, min_out = self.encoderX(min_out)


        # output = torch.split(cat_out, [shape[1], shape[1]], dim=0)
        # out = (output[0] + output[1]) / 2

        L, B, C = enc_output.shape
        out = enc_output.view(shapes[-2], shapes[-2], B, C).permute(2, 3, 0, 1).contiguous()

        return out


@TRACK_FUSIONS.register
class Fusion(ModuleBase):
    default_hyper_params = dict(
        input_channel=48,
        feat_fx_size=[8, 8, 4],
        feat_fz_size=[19, 19, 10],
        used_layers=[6, 8, 11],
        num_heads=3,
        num_layers=2,
        prob_dropout=0.1,
    )

    def __init__(self):
        super(Fusion, self).__init__()
        self.position_embeeding_fx_dict = nn.ModuleDict()
        self.position_embeeding_fz_dict = nn.ModuleDict()
        self.used_layers = None

    def update_params(self):
        super().update_params()

        self.encoder = Encoder(mid_channels_model=self._hyper_params['input_channel'],
                               mid_channels_ffn=self._hyper_params['input_channel'] * 4,
                               num_heads=self._hyper_params['num_heads'],
                               num_layers=self._hyper_params['num_layers'],
                               prob_dropout=self._hyper_params['prob_dropout'], )
        self.feat_fx_size = self._hyper_params['feat_fx_size']
        self.feat_fz_size = self._hyper_params['feat_fz_size']
        self.used_layers = self._hyper_params['used_layers']
        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fx_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fx_size'][idx])

        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fz_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fz_size'][idx])

    def forward(self, x):
        cat_results = list()
        shapes = list()

        for idx in range(len(x)):
            shapes.append(x[idx].shape[-2])
        self.position_embeeding = self.position_embeeding_fx_dict if shapes[-2] == self.feat_fx_size[-2] else self.position_embeeding_fz_dict
        for idx, result in enumerate(x):
            result = self.position_embeeding[str(self.used_layers[idx])](result)
            result = result.view(*result.shape[:2], -1)  # B, C, HW
            result = result.permute(2, 0, 1).contiguous()  # HW, B, C
            cat_results.append(result)


        ###########################################
        # xf1 = x[1].cpu().detach().numpy()
        # xf2 = torch.zeros(8, 8).cpu().detach().numpy()
        # for x in range(48):
        #     xf2 = xf2 + xf1[0][x]
        # plt.imshow(xf2, cmap='viridis', origin='upper')
        # plt.title('Window')
        # plt.colorbar()
        # # 保存图片
        # plt.savefig('Window.png')
        # plt.show()
        #############################################

        out_dim = torch.cat(cat_results, dim=0)
        ##############################################################
        enc_output, cat_out1 = self.encoder(out_dim)

        output = torch.split(enc_output, [shapes[0] ** 2, shapes[1] ** 2,shapes[2] ** 2], dim=0)[-2]
        L, B, C = output.shape
        out = output.view(shapes[-2], shapes[-2], B, C).permute(2, 3, 0, 1).contiguous()
        # ###########################################
        # xf1 = out.cpu().detach().numpy()
        # xf2 = torch.zeros(8, 8).cpu().detach().numpy()
        # for x in range(48):
        #     xf2 = xf2 + xf1[0][x]
        # plt.imshow(xf2, cmap='viridis', origin='upper')
        # plt.title('Window1')
        # plt.colorbar()
        # # 保存图片
        # plt.savefig('Window1.png')
        # plt.show()
        # #############################################
        return out


@TRACK_FUSIONS.register
class Fusion_M(ModuleBase):
    default_hyper_params = dict(
        input_channel=48,
        feat_fx_size=[8, 8, 4],
        feat_fz_size=[19, 19, 10],
        used_layers=[6, 8, 11],
        num_heads=3,
        num_layers=2,
        prob_dropout=0.1,
    )

    def __init__(self):
        super(Fusion_M, self).__init__()
        self.position_embeeding_fx_dict = nn.ModuleDict()
        self.position_embeeding_fz_dict = nn.ModuleDict()
        self.used_layers = None

    def update_params(self):
        super().update_params()

        self.encoder = Encoder(mid_channels_model=self._hyper_params['input_channel'],
                               mid_channels_ffn=self._hyper_params['input_channel'] * 4,
                               num_heads=self._hyper_params['num_heads'],
                               num_layers=self._hyper_params['num_layers'],
                               prob_dropout=self._hyper_params['prob_dropout'], )
        self.feat_fx_size = self._hyper_params['feat_fx_size']
        self.feat_fz_size = self._hyper_params['feat_fz_size']
        self.used_layers = self._hyper_params['used_layers']
        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fx_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fx_size'][idx])

        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fz_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fz_size'][idx])

    def forward(self, x):
        cat_results = list()
        shapes = list()

        for idx in range(len(x)):
            shapes.append(x[idx].shape[-2])
        self.position_embeeding = self.position_embeeding_fx_dict if shapes[-2] == self.feat_fx_size[-2] else self.position_embeeding_fz_dict
        for idx, result in enumerate(x):
            result = self.position_embeeding[str(self.used_layers[idx])](result)
            result = result.view(*result.shape[:2], -1)  # B, C, HW
            result = result.permute(2, 0, 1).contiguous()  # HW, B, C
            cat_results.append(result)

        out_dim = torch.cat(cat_results, dim=0)
        ##############################################################
        enc_output, cat_out1 = self.encoder(out_dim)

        output = torch.split(cat_out1, [shapes[0] ** 2, shapes[1] ** 2,shapes[2] ** 2], dim=0)[-2]
        L, B, C = output.shape
        out = output.view(shapes[-2], shapes[-2], B, C).permute(2, 3, 0, 1).contiguous()

        return out

@TRACK_FUSIONS.register
class Fusion_alexnet(ModuleBase):
    default_hyper_params = dict(
        input_channel=48,
        feat_fx_size=[8, 8, 4],
        feat_fz_size=[19, 19, 10],
        used_layers=[6, 8, 11],
        num_heads=3,
        num_layers=2,
        prob_dropout=0.1,
    )

    def __init__(self):
        super(Fusion_alexnet, self).__init__()
        self.position_embeeding_fx_dict = nn.ModuleDict()
        self.position_embeeding_fz_dict = nn.ModuleDict()
        self.used_layers = None

    def update_params(self):
        super().update_params()

        self.encoder = Encoder(mid_channels_model=self._hyper_params['input_channel'],
                               mid_channels_ffn=self._hyper_params['input_channel'] * 4,
                               num_heads=self._hyper_params['num_heads'],
                               num_layers=self._hyper_params['num_layers'],
                               prob_dropout=self._hyper_params['prob_dropout'], )
        self.feat_fx_size = self._hyper_params['feat_fx_size']
        self.feat_fz_size = self._hyper_params['feat_fz_size']
        self.used_layers = self._hyper_params['used_layers']
        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fx_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fx_size'][idx])

        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fz_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fz_size'][idx])

    def forward(self, x):
        cat_results = list()
        shapes = list()

        for idx in range(len(x)):
            shapes.append(x[idx].shape[-2])
        self.position_embeeding = self.position_embeeding_fx_dict if shapes[-2] == self.feat_fx_size[-2] else self.position_embeeding_fz_dict
        for idx, result in enumerate(x):
            result = self.position_embeeding[str(self.used_layers[idx])](result)
            result = result.view(*result.shape[:2], -1)  # B, C, HW
            result = result.permute(2, 0, 1).contiguous()  # HW, B, C
            cat_results.append(result)

        out_dim = torch.cat(cat_results, dim=0)
        ##############################################################
        enc_output, cat_out1 = self.encoder(out_dim)

        output = torch.split(enc_output, [shapes[0] ** 2, shapes[1] ** 2,shapes[2] ** 2], dim=0)[-1]
        L, B, C = output.shape
        out = output.view(shapes[-1], shapes[-1], B, C).permute(2, 3, 0, 1).contiguous()

        return out



class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, mask=None):
        enc_output, enc_slf_attn = self.slf_attn(query=enc_input, key=enc_input, value=enc_input,
                                                 attn_mask=mask)
        enc_output = enc_input + enc_output
        enc_output = self.norm(enc_output)

        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

@TRACK_FUSIONS.register
class Fusion_TPN(ModuleBase):
    default_hyper_params = dict(
        input_channel=48,
        feat_fx_size=[8, 8, 4],
        feat_fz_size=[19, 19, 10],
        used_layers=[6, 8, 11],
        num_heads=3,
        num_layers=2,
        prob_dropout=0.1,
    )

    def __init__(self):
        super(Fusion_TPN, self).__init__()
        self.position_embeeding_fx_dict = nn.ModuleDict()
        self.position_embeeding_fz_dict = nn.ModuleDict()
        self.used_layers = None

    def update_params(self):
        super().update_params()
        self.encoder_X = Encoder(mid_channels_model=self._hyper_params['input_channel'] * 4,
                               mid_channels_ffn=self._hyper_params['input_channel'] * 4,
                               num_heads=self._hyper_params['num_heads'],
                               num_layers=2,
                               prob_dropout=self._hyper_params['prob_dropout'], )
        self.encoder_L = Encoder_siamtpn(mid_channels_model=self._hyper_params['input_channel'],
                               mid_channels_ffn=self._hyper_params['input_channel'] * 4,
                               num_heads=self._hyper_params['num_heads'],
                               num_layers=1,
                               prob_dropout=self._hyper_params['prob_dropout'], )
        self.encoder = Encoder_siamtpn(mid_channels_model=self._hyper_params['input_channel'],
                               mid_channels_ffn=self._hyper_params['input_channel'] * 4,
                               num_heads=self._hyper_params['num_heads'],
                               num_layers=1,
                               prob_dropout=self._hyper_params['prob_dropout'], )
        self.encoder_M = Encoder_siamtpn(mid_channels_model=self._hyper_params['input_channel'],
                               mid_channels_ffn=self._hyper_params['input_channel'] * 4,
                               num_heads=self._hyper_params['num_heads'],
                               num_layers=1,
                               prob_dropout=self._hyper_params['prob_dropout'], )
        self.feat_fx_size = self._hyper_params['feat_fx_size']
        self.feat_fz_size = self._hyper_params['feat_fz_size']
        self.used_layers = self._hyper_params['used_layers']
        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fx_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fx_size'][idx])

        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fz_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fz_size'][idx])

    def forward(self, x):
        cat_results = list()
        shapes = list()
        results = list()
        for idx in range(len(x)):
            shapes.append(x[idx].shape[-2])
        self.position_embeeding = self.position_embeeding_fx_dict if shapes[-2] == self.feat_fx_size[-2] else self.position_embeeding_fz_dict
        for idx, result in enumerate(x):
            result = self.position_embeeding[str(self.used_layers[idx])](result)
            result = result.view(*result.shape[:2], -1)  # B, C, HW
            result = result.permute(2, 0, 1).contiguous()  # HW, B, C
            cat_results.append(result)


        ##############################################################
        for idx in range(self._hyper_params['num_layers']):
            enc_output1, cat_out1 = self.encoder_L(cat_results[1], cat_results[0])
            enc_output2, cat_out2 = self.encoder(cat_results[1], cat_results[1])
            enc_output3, cat_out3 = self.encoder_M(cat_results[1], cat_results[2])
            results.append(enc_output1)
            results.append(enc_output2)
            results.append(enc_output3)
            results.append(cat_results[1])

            out = torch.cat(results,dim=2)
            out, cat_out4 = self.encoder_X(out)
        # output = torch.split(out, [shapes[1] ** 2, shapes[1] ** 2,shapes[1] ** 2], dim=0)[1]
        L, B, C = out.shape
        out = out.view(shapes[-2], shapes[-2], B, C).permute(2, 3, 0, 1).contiguous()

        return out





@TRACK_FUSIONS.register
class Fusion_TPN_X(ModuleBase):
    default_hyper_params = dict(
        input_channel=48,
        feat_fx_size=[8, 8, 4],
        feat_fz_size=[19, 19, 10],
        used_layers=[6, 8, 11],
        num_heads=3,
        num_layers=2,
        prob_dropout=0.1,
    )

    def __init__(self):
        super(Fusion_TPN_X, self).__init__()
        self.position_embeeding_fx_dict = nn.ModuleDict()
        self.position_embeeding_fz_dict = nn.ModuleDict()
        self.used_layers = None

    def update_params(self):
        super().update_params()
        self.encoder_L = Encoder_siamtpn(mid_channels_model=self._hyper_params['input_channel'] * 2,
                               mid_channels_ffn=self._hyper_params['input_channel'] * 4,
                               num_heads=self._hyper_params['num_heads'],
                               num_layers=2,
                               prob_dropout=self._hyper_params['prob_dropout'], )
        self.encoder = Encoder(mid_channels_model=self._hyper_params['input_channel'],
                               mid_channels_ffn=self._hyper_params['input_channel'] * 4,
                               num_heads=self._hyper_params['num_heads'],
                               num_layers=self._hyper_params['num_layers'],
                               prob_dropout=self._hyper_params['prob_dropout'], )
        self.encoder_self = Encoder_siamtpn(mid_channels_model=self._hyper_params['input_channel'] * 2,
                               mid_channels_ffn=self._hyper_params['input_channel'] * 4,
                               num_heads=self._hyper_params['num_heads'],
                               num_layers=2,
                               prob_dropout=self._hyper_params['prob_dropout'], )
        self.encoder_M = Encoder_siamtpn(mid_channels_model=self._hyper_params['input_channel'] * 2,
                               mid_channels_ffn=self._hyper_params['input_channel'] * 4,
                               num_heads=self._hyper_params['num_heads'],
                               num_layers=2,
                               prob_dropout=self._hyper_params['prob_dropout'], )
        self.feat_fx_size = self._hyper_params['feat_fx_size']
        self.feat_fz_size = self._hyper_params['feat_fz_size']
        self.used_layers = self._hyper_params['used_layers']
        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fx_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fx_size'][idx])

        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fz_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fz_size'][idx])

    def forward(self, x):
        cat_results = list()
        shapes = list()
        results = list()
        for idx in range(len(x)):
            shapes.append(x[idx].shape[-2])
        self.position_embeeding = self.position_embeeding_fx_dict if shapes[-2] == self.feat_fx_size[-2] else self.position_embeeding_fz_dict
        for idx, result in enumerate(x):
            result = self.position_embeeding[str(self.used_layers[idx])](result)
            result = result.view(*result.shape[:2], -1)  # B, C, HW
            result = result.permute(2, 0, 1).contiguous()  # HW, B, C
            cat_results.append(result)

        out_dim = torch.cat(cat_results, dim=0)
        ##############################################################
        enc_output, out_dim = self.encoder(out_dim)
        out1 = torch.split(out_dim, [shapes[0] ** 2, shapes[1] ** 2, shapes[2] ** 2], dim=0)[0]
        out2 = torch.split(out_dim, [shapes[0] ** 2, shapes[1] ** 2, shapes[2] ** 2], dim=0)[1]
        out3 = torch.split(out_dim, [shapes[0] ** 2, shapes[1] ** 2, shapes[2] ** 2], dim=0)[2]
        ##############################################################

        enc_output1, cat_out1 = self.encoder_L(out2, out1)
        enc_output2, cat_out2 = self.encoder_self(out2, out2)
        enc_output3, cat_out3 = self.encoder_M(out2, out3)
        results.append(enc_output1)
        results.append(enc_output2)
        results.append(enc_output3)
        results.append(out2)
        out = torch.cat(results,dim=2)
        # output = torch.split(out, [shapes[1] ** 2, shapes[1] ** 2,shapes[1] ** 2], dim=0)[1]
        L, B, C = out.shape
        out = out.view(shapes[-2], shapes[-2], B, C).permute(2, 3, 0, 1).contiguous()

        return out

@TRACK_FUSIONS.register
class FusionM(ModuleBase):
    default_hyper_params = dict(
        input_channel=48,
        feat_fx_size=[8, 8, 4],
        feat_fz_size=[19, 19, 10],
        used_layers=[6, 8, 11],
        num_heads=3,
        num_layers=2,
        prob_dropout=0.1,
    )

    def __init__(self):
        super(FusionM, self).__init__()
        self.position_embeeding_fx_dict = nn.ModuleDict()
        self.position_embeeding_fz_dict = nn.ModuleDict()
        self.used_layers = None

    def update_params(self):
        super().update_params()

        self.encoder = Encoder(mid_channels_model=self._hyper_params['input_channel'],
                               mid_channels_ffn=self._hyper_params['input_channel'] * 4,
                               num_heads=self._hyper_params['num_heads'],
                               num_layers=self._hyper_params['num_layers'],
                               prob_dropout=self._hyper_params['prob_dropout'], )
        self.feat_fx_size = self._hyper_params['feat_fx_size']
        self.feat_fz_size = self._hyper_params['feat_fz_size']
        self.used_layers = self._hyper_params['used_layers']
        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fx_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fx_size'][idx])

        for idx, layer in enumerate(self.used_layers):
            self.position_embeeding_fz_dict[str(layer)] = SpatialPositionEncodingLearned(
                self._hyper_params['input_channel'],
                self._hyper_params['feat_fz_size'][idx])

    def forward(self, x):
        cat_results = list()
        shapes = list()

        for idx in range(len(x)):
            shapes.append(x[idx].shape[-2])
        self.position_embeeding = self.position_embeeding_fx_dict if shapes[-2] == self.feat_fx_size[-2] else self.position_embeeding_fz_dict
        for idx, result in enumerate(x):
            result = self.position_embeeding[str(self.used_layers[idx])](result)
            result = result.view(*result.shape[:2], -1)  # B, C, HW
            result = result.permute(2, 0, 1).contiguous()  # HW, B, C
            cat_results.append(result)

        out_dim = torch.cat(cat_results, dim=0)
        ##############################################################
        enc_output, cat_out1 = self.encoder(out_dim)
        out = torch.cat(cat_out1,dim=2)
        output = torch.split(out, [shapes[0] ** 2, shapes[1] ** 2,shapes[2] ** 2], dim=0)[-2]
        L, B, C = output.shape
        out = output.view(shapes[-2], shapes[-2], B, C).permute(2, 3, 0, 1).contiguous()

        return out

if __name__ == "__main__":
    print(VOS_BACKBONES)
    resnet_m = Fusion()
    image = torch.rand((1, 3, 257, 257))
    print(image.shape)
    feature = resnet_m(image)
    print(feature.shape)
    print(resnet_m.state_dict().keys())
    # print(resnet_m)
