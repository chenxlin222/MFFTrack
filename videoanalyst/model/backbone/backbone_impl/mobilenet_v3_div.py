from typing import OrderedDict
import torch.nn as nn
import math

import torch

import sys
import os

env_path = os.path.join(os.path.dirname(__file__), '/home/xlsun/xlsun/code/SparseTT')
if env_path not in sys.path:
    sys.path.append(env_path)

from videoanalyst.model.backbone.backbone_base import (TRACK_BACKBONES,
                                                       VOS_BACKBONES)
from videoanalyst.model.common_opr.common_block import conv_bn_relu
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.neck.neck_impl.encoder_enhance import Encoder
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmdet.models.builder import BACKBONES
from videoanalyst.model.utils.transformer_layers import (SpatialPositionEncodingLearned,
                                                         MultiHeadAttention,
                                                         PositionWiseFeedForward)
from videoanalyst.model.neck.neck_impl.encoder_enhance import Encoder
from mmcv import Config
from mmdet.utils import get_root_logger

__all__ = ['mobilenetv3_large', 'mobilenetv3_small']


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


# @BACKBONES.register_module()
class MobileNetV3SmallImpl(BaseModule):
    def __init__(self, cfgs_mode, width_mult=1., used_layers=[6, 8, 11],
                 feat_channel=[40, 48, 96],
                 final_channel=48, init_cfg=None):
        super(MobileNetV3SmallImpl, self).__init__(init_cfg=init_cfg)
        # setting of inverted residual blocks
        cfgs = self.get_cfgs(cfgs_mode)
        self.cfgs = cfgs
        self.mode = cfgs_mode
        self.used_layers = used_layers
        self.feat_channel = feat_channel

        self.feat_adjuster_block_dict = nn.ModuleDict()

        for idx, layer in enumerate(self.used_layers):
            self.feat_adjuster_block_dict[str(layer)] = conv_1x1_bn(self.feat_channel[idx], final_channel)


        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]

        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # self._initialize_weights()

    def forward(self, x):

        results = list()
        if self.used_layers is not None and isinstance(self.used_layers, list):
            # for layer in self.used_layers:
            #     results.append(self.feat_adjuster_block_dict[str(layer)](self.features[:layer](x)))
            for i in range(len(self.features)):
                x = self.features[i](x)
                if i in self.used_layers:
                    results.append(self.feat_adjuster_block_dict[str(i)](x))
        else:
            results.append(self.features(x))

        return results

    def _initialize_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
        else:
            self.load_pretrain(self.init_cfg.checkpoint)

    def load_pretrain(self, pretrained_path):
        self.logger = get_root_logger()
        self.logger.info('load pretrained model from {}'.format(pretrained_path))
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path,
                                     map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'],
                                                 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')

        try:
            self.check_keys(pretrained_dict)
        except:
            self.logger.info('[Warning]: using pretrain as features.\
                    Adding "features." as prefix')
            new_dict = {}
            for k, v in pretrained_dict.items():
                k = 'features.' + k
                new_dict[k] = v
            pretrained_dict = new_dict
            self.check_keys(pretrained_dict)
        self.load_state_dict(pretrained_dict, strict=False)

    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters
        share common prefix 'module.' '''
        self.logger.info('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def check_keys(self, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(self.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        # filter 'num_batches_tracked'
        missing_keys = [x for x in missing_keys
                        if not x.endswith('num_batches_tracked')]
        if len(missing_keys) > 0:
            self.logger.info('[Warning] missing keys: {}'.format(missing_keys))
            self.logger.info('missing keys:{}'.format(len(missing_keys)))
        if len(unused_pretrained_keys) > 0:
            self.logger.info('[Warning] unused_pretrained_keys: {}'.format(
                unused_pretrained_keys))
            self.logger.info('unused checkpoint keys:{}'.format(
                len(unused_pretrained_keys)))
        self.logger.info('used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, \
            'load NONE from pretrained checkpoint'
        return True

    def get_cfgs(self, cfgs_mode):
        assert cfgs_mode in ['small', 'large'], 'cfgs_mode must be small or large'
        if cfgs_mode == 'small':
            cfgs = [  #
                # k, t, c, SE, HS, s
                [3, 1, 16, 1, 0, 2],
                [3, 4.5, 24, 0, 0, 2],
                [3, 3.67, 24, 0, 0, 1],
                [5, 4, 40, 1, 1, 2],
                [5, 6, 40, 1, 1, 1],
                [5, 6, 40, 1, 1, 1],
                [5, 3, 48, 1, 1, 1],
                [5, 3, 48, 1, 1, 1],
                [5, 6, 96, 1, 1, 2],
                [5, 6, 96, 1, 1, 1],
                [5, 6, 96, 1, 1, 1],
            ]
            return cfgs
        elif cfgs_mode == 'large':
            cfgs = [
                # k, t, c, SE, HS, s
                [3, 1, 16, 0, 0, 1],
                [3, 4, 24, 0, 0, 2],
                [3, 3, 24, 0, 0, 1],
                [5, 3, 40, 1, 0, 2],
                [5, 3, 40, 1, 0, 1],
                [5, 3, 40, 1, 0, 1],
                [3, 6, 80, 0, 1, 2],
                [3, 2.5, 80, 0, 1, 1],
                [3, 2.3, 80, 0, 1, 1],
                [3, 2.3, 80, 0, 1, 1],
                [3, 6, 112, 1, 1, 1],
                [3, 6, 112, 1, 1, 1],
                [5, 6, 160, 1, 1, 2],
                [5, 6, 160, 1, 1, 1],
                [5, 6, 160, 1, 1, 1]
            ]
            return cfgs


@TRACK_BACKBONES.register
class MobileNetV3_div(ModuleBase):
    default_hyper_params = dict(
        cfgs_mode='small',
        pretrained='',
        used_layers=[6, 8, 11],
        feat_channel=[40, 48, 96],
        final_channel=48
    )

    def __init__(self):
        super(MobileNetV3_div, self).__init__()

    def update_params(self):
        super().update_params()
        init_cfg = Config(dict(type='Pretrained', checkpoint=self._hyper_params['pretrained']))
        # init_cfg = None
        self.net = MobileNetV3SmallImpl(
            cfgs_mode=self._hyper_params['cfgs_mode'],
            used_layers=self._hyper_params['used_layers'],
            feat_channel=self._hyper_params['feat_channel'],
            final_channel=self._hyper_params['final_channel'],
            init_cfg=init_cfg
        )
        self.init_weights()

    def init_weights(self):
        self.net._initialize_weights()

    def forward(self, x):
        outs = self.net(x)
        return outs


if __name__ == "__main__":
    print(VOS_BACKBONES)
    resnet_m = MobileNetV3X()
    image = torch.rand((1, 3, 257, 257))
    print(image.shape)
    feature = resnet_m(image)
    print(feature.shape)
    print(resnet_m.state_dict().keys())
    # print(resnet_m)
