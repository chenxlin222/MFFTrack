import math
from typing import Callable, List, Optional

import torch
from mmengine import Config, DictAction
from mmcv.runner import BaseModule
from mmdet.utils import get_root_logger
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial

from videoanalyst.model.backbone.backbone_base import TRACK_BACKBONES
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.neck.neck_impl.encoder import Encoder, SpatialPositionEncodingLearned


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer(inplace=True))


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


class InvertedResidualConfig:
    def __init__(self,
                 input_c: int,
                 kernel: int,
                 expanded_c: int,
                 out_c: int,
                 use_se: bool,
                 activation: str,
                 stride: int,
                 width_multi: float):
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"  # whether using h-swish activation
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)


class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_c != cnf.input_c:
            layers.append(ConvBNActivation(cnf.input_c,
                                           cnf.expanded_c,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))

        # depthwise
        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.expanded_c,
                                       kernel_size=cnf.kernel,
                                       stride=cnf.stride,
                                       groups=cnf.expanded_c,
                                       norm_layer=norm_layer,
                                       activation_layer=activation_layer))

        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))

        # project
        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.out_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x

        return result


class MobileNetV3_small(BaseModule):

    def __init__(self,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 use_layers=[6,8,11],
                 feat_fx_size=[19,19,10],
                 feat_fz_size=[8,8,4],
                 final_channel=48,
                 in_channels=None,
                 out_channels=None,
                 init_cfg=None):
        super(MobileNetV3_small, self).__init__(init_cfg=init_cfg)
        width_multi = 1.0
        bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
        self.feat_fx_size = feat_fx_size
        self.feat_fz_size = feat_fz_size
        self.use_layers = use_layers
        self.position_embeeding_fx_dict = nn.ModuleDict()
        self.position_embeeding_fz_dict = nn.ModuleDict()
        reduce_divider = 1
        self.conv = nn.ModuleDict()
        for i in range(len(use_layers)):
            self.conv[str(i)] = nn.Sequential(
                nn.Conv2d(in_channels=in_channels[i], out_channels=out_channels[i], kernel_size=1, stride=1, padding=0),
                nn.LayerNorm(out_channels[i]),
                nn.ReLU(inplace=False)

            )
        inverted_residual_setting = [
            # input_c, kernel, expanded_c, out_c, use_se, activation, stride
            bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
        ]

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty.")
        elif not (isinstance(inverted_residual_setting, List) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_c = inverted_residual_setting[0].input_c
        layers.append(ConvBNActivation(3,
                                       firstconv_output_c,
                                       kernel_size=3,
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        self.features = nn.Sequential(*layers)

        for idx, layer in enumerate(self.use_layers):
            self.position_embeeding_fx_dict[str(idx)] = SpatialPositionEncodingLearned(final_channel,
                                                                                         self.feat_fx_size[idx])

        for idx, layer in enumerate(self.use_layers):
            self.position_embeeding_fz_dict[str(idx)] = SpatialPositionEncodingLearned(final_channel,
                                                                                         self.feat_fz_size[idx])


        self.encoder = Encoder(
            mid_channels_models=48,
            min_channels=256,
            nheads=3,
            num_encoder_layer=2)

    def _forward_impl(self, x: Tensor) -> Tensor:
        out = []
        dim_out = list()
        self.shape = self.feat_fx_size if x.shape[-1] == 289 else self.feat_fz_size
        self.position_embeeding = self.position_embeeding_fx_dict if x.shape[
                                                                         -1] == 289 else self.position_embeeding_fz_dict
        for i in range(len(self.features)):
            x = self.features[i](x)
            if i in self.use_layers:
                out.append(x)
        for i,x in enumerate(out):
            result = self.conv[str(i)](x)
            result = self.position_embeeding[str(i)](result)
            result = result.view(*result.shape[:2], -1)  # B, C, HW
            result = result.permute(2, 0, 1).contiguous()  # HW, B, C
            dim_out.append(result)

        output = torch.cat(dim_out,dim=0)
        output = self.encoder(output)
        output = torch.split(output, [self.shape[i] ** 2 for i in range(len(self.shape))], dim=0)[-2]
        L, B, C = output.shape
        output = output.view(self.shape[-2], self.shape[-2], B, C).permute(2, 3, 0, 1).contiguous()

        return output

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

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

@TRACK_BACKBONES.register
class MobileNetV3(ModuleBase):
    default_hyper_params = dict(
        pretrained='',
        use_layers=[6, 8, 11],
        feat_channel=[40, 48, 96],
        feat_fx_size=[8, 8, 4],
        feat_fz_size=[19, 19, 10],
        final_channel=48,
        out_channels=[48,48,48]
    )

    def __init__(self):
        super(MobileNetV3, self).__init__()

    def update_params(self):
        super().update_params()
        init_cfg = Config(dict(type='Pretrained', checkpoint=self._hyper_params['pretrained']))

        # init_cfg = None
        self.net = MobileNetV3_small(
            block=None,
            norm_layer=None,
            use_layers=self._hyper_params['use_layers'],
            feat_fx_size=self._hyper_params['feat_fx_size'],
            feat_fz_size=self._hyper_params['feat_fz_size'],
            final_channel=self._hyper_params['final_channel'],
            in_channels=self._hyper_params['feat_channel'],
            out_channels=self._hyper_params['out_channels'],
            init_cfg=init_cfg
        )
        self.init_weights()

    def init_weights(self):
        self.net._initialize_weights()

    def forward(self, x):
        outs = self.net(x)
        return outs


def mobilenet_v3_large(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


def mobilenet_v3_small(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting)
