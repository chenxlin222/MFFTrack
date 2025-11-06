# -*- coding: utf-8 -*
import math

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.utils import get_root_logger

from videoanalyst.model.backbone.backbone_base import (TRACK_BACKBONES,
                                                       VOS_BACKBONES)
from videoanalyst.model.backbone.backbone_impl.mobilenet_v3_div import conv_1x1_bn

from videoanalyst.model.common_opr.common_block import conv_bn_relu
from videoanalyst.model.module_base import ModuleBase

from mmcv import Config

class AlexNetSmallImpl(BaseModule):
    def __init__(self,feat_channel,final_channel,used_layers,init_cfg=None):
        super(AlexNetSmallImpl, self).__init__(init_cfg=init_cfg)
        configs = [3, 96, 256, 384, 384, 256]

        self.layer1 = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
        )
        self.feature_size = configs[5]
        self.feat_channel = feat_channel
        self.final_channel = final_channel
        self.used_layers = used_layers
        self.feat_adjuster_block_dict = nn.ModuleDict()
        for idx, layer in enumerate(self.used_layers):
            self.feat_adjuster_block_dict[str(layer)] = conv_1x1_bn(self.feat_channel[idx], self.final_channel)

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

    def forward(self, x):
        results = list()
        output = list()
        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        results.append(x3)
        results.append(x4)
        results.append(x5)
        for idx, layer in enumerate(self.used_layers):
            out = self.feat_adjuster_block_dict[str(layer)](results[idx])
            output.append(out)
        return output

@VOS_BACKBONES.register
@TRACK_BACKBONES.register
class AlexNet(ModuleBase):
    r"""
    AlexNet

    Hyper-parameters
    ----------------
    pretrain_model_path: string
        Path to pretrained backbone parameter file,
        Parameter to be loaded in _update_params_
    """
    default_hyper_params = dict(
        pretrained='',
        used_layers=[6, 8, 11],
        feat_channel=[40, 48, 96],
        feat_fx_size=[8, 8, 4],
        feat_fz_size=[19, 19, 10],
        final_channel=48,
        num_heads=3,
        num_layers=2,
        prob_dropout=0.0
    )
    def __init__(self):
        super(AlexNet, self).__init__()




    def update_params(self):
        super().update_params()
        self.used_layers=self._hyper_params['used_layers']
        self.feat_channel=self._hyper_params['feat_channel']
        self.final_channel=self._hyper_params['final_channel']


        init_cfg = Config(dict(type='Pretrained', checkpoint=self._hyper_params['pretrained']))

        self.net = AlexNetSmallImpl(
            feat_channel=self._hyper_params['feat_channel'],
            final_channel=self._hyper_params['final_channel'],
            used_layers=self._hyper_params['used_layers'],
            init_cfg=init_cfg
        )
        self.init_weights()
    def init_weights(self):

        self.net._initialize_weights()


    def forward(self, x):
        outs = self.net(x)
        return outs


