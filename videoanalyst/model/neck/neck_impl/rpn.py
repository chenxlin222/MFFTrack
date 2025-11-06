# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn

from videoanalyst.model.common_opr.common_block import xcorr_depthwise
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.neck.neck_base import TRACK_NECKS



class DepthwiseXCorr(nn.Module):

    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        # self.head = nn.Sequential(
        #     nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(hidden),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(hidden, out_channels, kernel_size=1)
        # )

    def forward(self, kernel, search):
        # kernel = self.conv_kernel(kernel)
        # search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)


        return feature


@TRACK_NECKS.register
class DepthwiseRPN(ModuleBase):
    default_hyper_params = dict(
        anchor_num=5,
        in_channels=256,
        out_channels=256
    )

    def __init__(self):
        super(DepthwiseRPN, self).__init__()

    def update_params(self):
        super(DepthwiseRPN, self).update_params()

        self.in_channels = self._hyper_params['in_channels']
        self.anchor_num = self._hyper_params['anchor_num']
        self.out_channels = self._hyper_params["out_channels"]

        self.neck = DepthwiseXCorr(self.in_channels, self.out_channels, 2 * self.anchor_num)


    def forward(self, x_f, z_f):
        feature = self.neck(z_f, x_f)

        return feature
