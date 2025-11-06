# -*- coding: utf-8 -*

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet.models.backbones.resnet import Bottleneck

from videoanalyst.model.common_opr.common_block import conv_bn_relu
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_head.taskhead_base import TRACK_HEADS, VOS_HEADS
from videoanalyst.model.task_head.taskhead_impl.double_conv_fc_bbox_head import get_xy_ctr_np, get_fm_center_torch

torch.set_printoptions(precision=8)


class BasicResBlock(BaseModule):
    """Basic residual block.

    This block is a little different from the block in the ResNet backbone.
    The kernel size of conv1 is 1 in this block while 3 in ResNet BasicBlock.

    Args:
        in_channels (int): Channels of the input feature map.
        out_channels (int): Channels of the output feature map.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(BasicResBlock, self).__init__(init_cfg)

        # main path
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.conv2 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        # identity path
        self.conv_identity = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        identity = self.conv_identity(identity)
        out = x + identity

        out = self.relu(out)
        return out

def get_xy_ctr(score_size, score_offset, total_stride):
    batch, fm_height, fm_width = 1, score_size, score_size

    y_list = torch.linspace(0., fm_height - 1., fm_height).reshape(
        1, fm_height, 1, 1).repeat(1, 1, fm_width,
                                   1)  # .broadcast([1, fm_height, fm_width, 1])
    x_list = torch.linspace(0., fm_width - 1., fm_width).reshape(
        1, 1, fm_width, 1).repeat(1, fm_height, 1,
                                  1)  # .broadcast([1, fm_height, fm_width, 1])
    xy_list = score_offset + torch.cat([x_list, y_list], 3) * total_stride
    xy_ctr = xy_list.repeat(batch, 1, 1, 1).reshape(
        batch, -1,
        2)  # .broadcast([batch, fm_height, fm_width, 2]).reshape(batch, -1, 2)
    xy_ctr = xy_ctr.type(torch.Tensor)
    return xy_ctr


def get_box(xy_ctr, offsets):
    offsets = offsets.permute(0, 2, 3, 1)  # (B, H, W, C), C=4
    offsets = offsets.reshape(offsets.shape[0], -1, 4)
    xy0 = (xy_ctr[:, :, :] - offsets[:, :, :2])
    xy1 = (xy_ctr[:, :, :] + offsets[:, :, 2:])
    bboxes_pred = torch.cat([xy0, xy1], 2)

    return bboxes_pred




@TRACK_HEADS.register
@VOS_HEADS.register
class DoubleConvFCBBoxHead(ModuleBase):
    default_hyper_params = dict(
        in_channels=512,
        num_convs=0,
        input_size_adapt=False,
        x_size=0,
        score_size=0,
        total_stride=0,
    )
    def __init__(self):
        super(DoubleConvFCBBoxHead, self).__init__()

        self.bi_fc = torch.nn.Parameter(torch.tensor(0.))
        self.si_fc = torch.nn.Parameter(torch.tensor(1.))
        self.bi_conv = torch.nn.Parameter(torch.tensor(0.))
        self.si_conv = torch.nn.Parameter(torch.tensor(1.))

    def offset2bbox(self, offsets, x_size):
        # bbox decoding
        if self.input_size_adapt and x_size > 0:
            score_offsets = (x_size - 1 - (offsets.size(-1) - 1) * self.total_stride) // 2
            # fm_ctr = get_xy_ctr_np(offsets.size(-1), score_offset, self.total_stride)
            # fm_ctr = fm_ctr.to(offsets.device)
            fm_ctr = get_fm_center_torch(offsets.shape[-1], score_offsets, self.total_stride, offsets.device)
        else:
            fm_ctr = self.fm_ctr
        bbox = get_box(fm_ctr, offsets)
        return bbox

    def update_params(self):

        self.in_channels = self._hyper_params['in_channels']
        self.num_convs = self._hyper_params['num_convs']
        self.total_stride = self._hyper_params["total_stride"]
        self.input_size_adapt = self._hyper_params["input_size_adapt"]

        x_size = self._hyper_params["x_size"]
        self.score_size = self._hyper_params["score_size"]
        score_offset = (x_size - 1 - (self.score_size - 1) * self.total_stride) // 2
        self._hyper_params["score_offset"] = score_offset
        self.score_offset = self._hyper_params["score_offset"]

        fm_ctr = get_fm_center_torch(self.score_size, self.score_offset, self.total_stride)
        self.register_buffer('fm_ctr', fm_ctr.clone().detach().requires_grad_(False))

        self.conv_branch_cls = ConvModule(in_channels=self.in_channels,
                                          out_channels=1,
                                          kernel_size=1,
                                          norm_cfg=dict(type='BN'),
                                          act_cfg=None)
        mid_channels = 1024
        self.fc_branch = nn.Sequential(
            nn.Conv2d(self.in_channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        convs = []
        for _ in range(self.num_convs):
            convs.append(Bottleneck(inplanes=mid_channels,
                                    planes=mid_channels // 4,
                                    conv_cfg=None,
                                    norm_cfg=dict(type='BN')))

        self.conv_branch = nn.Sequential(
            BasicResBlock(self.in_channels, mid_channels),
            *convs,
        )
        self.fc_cls = ConvModule(
            in_channels=mid_channels,
            out_channels=1,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=None
        )
        self.fc_reg = ConvModule(
            in_channels=mid_channels,
            out_channels=4,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=None
        )
        self.conv_cls = ConvModule(
            in_channels=mid_channels,
            out_channels=1,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=None
        )
        self.conv_reg = ConvModule(
            in_channels=mid_channels,
            out_channels=4,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=None
        )

    def forward(self,x,x_size=0):
        x_fc = self.fc_branch(x)
        x_fc_cls = self.fc_cls(x_fc)
        x_fc_reg = self.fc_reg(x_fc)
        x_fc_cls = x_fc_cls.permute(0,2,3,1).reshape(x_fc_cls.shape[0],-1,1)


        x_conv = self.conv_branch(x)
        x_conv_cls = self.conv_cls(x_conv)
        x_conv_reg = self.conv_reg(x_conv)
        x_conv_cls = x_conv_cls.permute(0,2,3,1).reshape(x_fc_cls.shape[0],-1,1)


        reg_fc = torch.exp(self.si_fc * x_fc_reg + self.bi_fc) * self.total_stride
        bbox_fc = self.offset2bbox(reg_fc, x_size)
        reg_conv = torch.exp(self.si_conv * x_conv_reg + self.bi_conv) * self.total_stride
        bbox_conv = self.offset2bbox(reg_conv, x_size)

        return x_fc_cls,bbox_fc,x_conv_cls,bbox_conv


@TRACK_HEADS.register
@VOS_HEADS.register
class MyDenseboxHead(ModuleBase):
    r"""
    Densebox Head for siamfcpp

    Hyper-parameter
    ---------------
    total_stride: int
        stride in backbone
    score_size: int
        final feature map
    x_size: int
        search image size
    num_conv3x3: int
        number of conv3x3 tiled in head
    head_conv_bn: list
        has_bn flag of conv3x3 in head, list with length of num_conv3x3
    head_width: int
        feature width in head structure
    conv_weight_std: float
        std for conv init
    """
    default_hyper_params = dict(
        total_stride=10,
        score_size=19,
        x_size=289,
        num_conv3x3=2,
        head_conv_bn=[False, False],
        input_size_adapt=False,
        in_channel=128
    )

    def __init__(self):
        super(MyDenseboxHead, self).__init__()


    def forward(self, x,x_size=0,raw_output=None):
        # classification head
        x_cls = self.conv_branch_cls(x)
        cls_score = x_cls.permute(0, 2, 3, 1)
        cls_score = cls_score.reshape(cls_score.shape[0], -1, 1)
        ctr_score = self.ctr_score(x)
        ctr_score = ctr_score.permute(0, 2, 3, 1)
        ctr_score = ctr_score.reshape(ctr_score.shape[0], -1, 1)


        offsets = self.conv_branch_reg(x)
        offsets = torch.exp(self.si * offsets + self.bi) * self.total_stride
        if raw_output:
            return [cls_score, ctr_score, offsets]
        # bbox decoding
        if self._hyper_params["input_size_adapt"] and x_size > 0:
            score_offset = (x_size - 1 -
                            (offsets.size(-1) - 1) * self.total_stride) // 2
            fm_ctr = get_xy_ctr_np(offsets.size(-1), score_offset,
                                   self.total_stride)
            fm_ctr = fm_ctr.to(offsets.device)
        else:
            fm_ctr = self.fm_ctr.to(offsets.device)
        bbox = get_box(fm_ctr, offsets)

        return [cls_score, ctr_score, bbox, x]

    def update_params(self):

        x_size = self._hyper_params["x_size"]
        score_size = self._hyper_params["score_size"]
        total_stride = self._hyper_params["total_stride"]
        score_offset = (x_size - 1 - (score_size - 1) * total_stride) // 2
        self._hyper_params["score_offset"] = score_offset

        self.score_size = self._hyper_params["score_size"]
        self.total_stride = self._hyper_params["total_stride"]
        self.score_offset = self._hyper_params["score_offset"]
        ctr = get_xy_ctr_np(self.score_size, self.score_offset,
                            self.total_stride)
        self.fm_ctr = ctr
        self.total_stride = self._hyper_params["total_stride"]
        self.in_channel = self._hyper_params["in_channel"]

        self.bi = torch.nn.Parameter(torch.tensor(0.).type(torch.Tensor))
        self.si = torch.nn.Parameter(torch.tensor(1.).type(torch.Tensor))
        self.conv_branch_cls = ConvModule(in_channels=self.in_channel,
                                          out_channels=1,
                                          kernel_size=1,
                                          norm_cfg=dict(type='BN'),
                                          act_cfg=None)
        self.ctr_score = ConvModule(in_channels=self.in_channel,
                                          out_channels=1,
                                          kernel_size=1,
                                          norm_cfg=dict(type='BN'),
                                          act_cfg=None)
        self.conv_branch_reg = ConvModule(in_channels=self.in_channel,
                                          out_channels=4,
                                          kernel_size=1,
                                          norm_cfg=dict(type='BN'),
                                          act_cfg=None)



    def _make_conv3x3(self):
        num_conv3x3 = self._hyper_params['num_conv3x3']
        head_conv_bn = self._hyper_params['head_conv_bn']
        head_width = self._hyper_params['head_width']
        self.cls_conv3x3_list = []
        self.bbox_conv3x3_list = []
        for i in range(num_conv3x3):
            cls_conv3x3 = conv_bn_relu(head_width,
                                       head_width,
                                       stride=1,
                                       kszie=3,
                                       pad=0,
                                       has_bn=head_conv_bn[i])

            bbox_conv3x3 = conv_bn_relu(head_width,
                                        head_width,
                                        stride=1,
                                        kszie=3,
                                        pad=0,
                                        has_bn=head_conv_bn[i])
            setattr(self, 'cls_p5_conv%d' % (i + 1), cls_conv3x3)
            setattr(self, 'bbox_p5_conv%d' % (i + 1), bbox_conv3x3)
            self.cls_conv3x3_list.append(cls_conv3x3)
            self.bbox_conv3x3_list.append(bbox_conv3x3)

    def _make_conv_output(self):
        head_width = self._hyper_params['head_width']
        self.cls_score_p5 = conv_bn_relu(head_width,
                                         1,
                                         stride=1,
                                         kszie=1,
                                         pad=0,
                                         has_relu=False)
        self.ctr_score_p5 = conv_bn_relu(head_width,
                                         1,
                                         stride=1,
                                         kszie=1,
                                         pad=0,
                                         has_relu=False)
        self.bbox_offsets_p5 = conv_bn_relu(head_width,
                                            4,
                                            stride=1,
                                            kszie=1,
                                            pad=0,
                                            has_relu=False)

    def _initialize_conv(self, ):
        num_conv3x3 = self._hyper_params['num_conv3x3']
        conv_weight_std = self._hyper_params['conv_weight_std']

        # initialze head
        conv_list = []
        for i in range(num_conv3x3):
            conv_list.append(getattr(self, 'cls_p5_conv%d' % (i + 1)).conv)
            conv_list.append(getattr(self, 'bbox_p5_conv%d' % (i + 1)).conv)

        conv_list.append(self.cls_score_p5.conv)
        conv_list.append(self.ctr_score_p5.conv)
        conv_list.append(self.bbox_offsets_p5.conv)
        conv_classifier = [self.cls_score_p5.conv]
        assert all(elem in conv_list for elem in conv_classifier)

        num_classes = 1
        pi = 0.01
        bv = -np.log((1 - pi) / pi)
        for ith in range(len(conv_list)):
            # fetch conv from list
            conv = conv_list[ith]
            # torch.nn.init.normal_(conv.weight, std=0.01) # from megdl impl.
            torch.nn.init.normal_(
                conv.weight, std=conv_weight_std)  # conv_weight_std = 0.0001
            # nn.init.kaiming_uniform_(conv.weight, a=np.sqrt(5))  # from PyTorch default implementation
            # nn.init.kaiming_uniform_(conv.weight, a=0)  # from PyTorch default implementation
            if conv in conv_classifier:
                torch.nn.init.constant_(conv.bias, torch.tensor(bv))
            else:
                # torch.nn.init.constant_(conv.bias, 0)  # from PyTorch default implementation
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(conv.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(conv.bias, -bound, bound)
