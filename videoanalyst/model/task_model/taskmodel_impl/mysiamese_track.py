# -*- coding: utf-8 -*

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from loguru import logger
from matplotlib import pyplot as plt

from videoanalyst.model.common_opr.common_block import (conv_bn_relu,
                                                        xcorr_depthwise)
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.neck.neck_impl.encoder import tensor2transformer
from videoanalyst.model.task_model.taskmodel_base import (TRACK_TASKMODELS,
                                                          VOS_TASKMODELS)

torch.set_printoptions(precision=8)

@TRACK_TASKMODELS.register
@VOS_TASKMODELS.register
class MySiamTrack_xcorr(ModuleBase):
    r"""
    SiamTrack model for tracking

    Hyper-Parameters
    ----------------
    pretrain_model_path: string
        path to parameter to be loaded into module
    head_width: int
        feature width in head structure
    """

    default_hyper_params = dict(
        pretrain_model_path="",
        in_channels=768,
        mid_channels=512,
        conv_weight_std=0.01,
        corr_feat_output=False,
        amp=False,
        in_channels_adjuster=96
    )

    def __init__(self, backbone, neck, head, loss=None):
        super(MySiamTrack_xcorr, self).__init__()
        self.basemodel = backbone
        self.neck = neck
        self.head = head
        self.loss = loss

    # def test_forward(self, f_x, enc_output, x_size):
    #     # feature matching
    #     output = self.neck.decode(f_x, enc_output)
    #     # head
    #
    #     fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(
    #         output, search_img.size(-1))
    #     # apply sigmoid
    #     fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
    #     fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)
    #     # apply centerness correction
    #     fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final
    #     # register extra output
    #     extra = dict(c_x=c_x, r_x=r_x, corr_fea=corr_fea)
    #     self.cf = c_x
    #     # output
    #     out_list = fcos_score_final, fcos_bbox_final, fcos_cls_prob_final, fcos_ctr_prob_final, extra
    #     return out_list

    def forward(self, *args, phase="train"):
        r"""
        Perform tracking process for different phases (e.g. train / init / track)

        Arguments
        ---------
        target_img: torch.Tensor
            target template image patch
        search_img: torch.Tensor
            search region image patch

        Returns
        -------
        fcos_score_final: torch.Tensor
            predicted score for bboxes, shape=(B, HW, 1)
        fcos_bbox_final: torch.Tensor
            predicted bbox in the crop, shape=(B, HW, 4)
        fcos_cls_prob_final: torch.Tensor
            classification score, shape=(B, HW, 1)
        fcos_ctr_prob_final: torch.Tensor
            center-ness score, shape=(B, HW, 1)
        """
        if phase == 'train':
            # resolve training data
            training_data = args[0]
            target_img = training_data["im_z"]
            search_img = training_data["im_x"]
            # backbone feature
            f_z = self.basemodel(target_img)
            f_x = self.basemodel(search_img)
            # feature adjustment
            f_z = self.feat_adjuster_z(f_z)
            f_x = self.feat_adjuster_x(f_x)
            N_out = self.neck(f_x, f_z)

            # head
            cls_fc, bbox_fc, cls_conv, bbox_conv = self.head(N_out)
            predict_data = dict(
                cls_fc=cls_fc,
                bbox_fc=bbox_fc,
                cls_conv=cls_conv,
                bbox_conv=bbox_conv
            )
            if self._hyper_params["corr_feat_output"]:
                predict_data["corr_feat"] = N_out
            return predict_data

        elif phase == 'feature':
            target_img, = args
            # backbone feature
            f_z = self.basemodel(target_img)
            # template as kernel
            enc_output = self.feat_adjuster_z(f_z)



            # output
            out_list = [enc_output]


        elif phase == 'track':
            if len(args) == 2:
                search_img, enc_output = args
                # backbone feature
                f_x = self.basemodel(search_img)

                # feature adjustment
                f_x = self.feat_adjuster_x(f_x)

                out = self.neck(f_x, enc_output)

                # head
                cls_fc, bbox_fc, cls_conv, bbox_conv = self.head(out, search_img.size(-1))
                # apply sigmoid
                cls_fc = torch.sigmoid(cls_fc)
                cls_conv = torch.sigmoid(cls_conv)
                # merge two cls socres
                cls_score_final = cls_fc + cls_conv * (1 - cls_fc)
                # register extra output
                extra = dict()  # for faster inference
                # extra = {"f_x": f_x, "encoder_output": enc_output, "decoder_output": output}
                # output
                out_list = cls_score_final, bbox_conv, extra
                out_list[-1]

            elif len(args) == 4:
                # c_x, r_x already computed
                c_z_k, r_z_k, c_x, r_x = args
            else:
                raise ValueError("Illegal args length: %d" % len(args))
        else:
            raise ValueError("Phase non-implemented.")
        return out_list

    def update_params(self):
        r"""
        Load model parameters
        """
        self._make_convs()
        # self._initialize_conv()
        super().update_params()

    def _make_convs(self):

        in_channels = self._hyper_params['in_channels']
        mid_channels = self._hyper_params['mid_channels']

        # feature adjustment
        self.feat_adjuster_z = conv_bn_relu(in_channels, mid_channels, kszie=1, has_relu=False, bn_eps=1e-3)
        self.feat_adjuster_x = conv_bn_relu(in_channels, mid_channels, kszie=1, has_relu=False, bn_eps=1e-3)

    def _initialize_conv(self, ):
        conv_weight_std = self._hyper_params['conv_weight_std']
        conv_list = [
            self.r_z_k.conv, self.c_z_k.conv, self.r_x.conv, self.c_x.conv
        ]
        for ith in range(len(conv_list)):
            conv = conv_list[ith]
            torch.nn.init.normal_(conv.weight,
                                  std=conv_weight_std)  # conv_weight_std=0.01

    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)

@TRACK_TASKMODELS.register
@VOS_TASKMODELS.register
class MySiamTrack(ModuleBase):
    r"""
    SiamTrack model for tracking

    Hyper-Parameters
    ----------------
    pretrain_model_path: string
        path to parameter to be loaded into module
    head_width: int
        feature width in head structure
    """

    default_hyper_params = dict(
        pretrain_model_path="",
        in_channels=768,
        mid_channels=512,
        conv_weight_std=0.01,
        corr_feat_output=False,
        amp=False,
        in_channels_adjuster=96
    )

    def __init__(self, backbone, fusion,neck, head, loss=None):
        super(MySiamTrack, self).__init__()
        self.basemodel = backbone
        self.fusion = fusion
        self.neck = neck
        self.head = head
        self.loss = loss

    # def test_forward(self, f_x, enc_output, x_size):
    #     # feature matching
    #     output = self.neck.decode(f_x, enc_output)
    #     # head
    #
    #     fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(
    #         output, search_img.size(-1))
    #     # apply sigmoid
    #     fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
    #     fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)
    #     # apply centerness correction
    #     fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final
    #     # register extra output
    #     # extra = dict(c_x=c_x, r_x=r_x, corr_fea=corr_fea)
    #     # self.cf = c_x
    #     # output
    #     out_list = fcos_score_final, fcos_bbox_final, fcos_cls_prob_final, fcos_ctr_prob_final, extra
    #     return out_list

    def forward(self, *args, phase="train"):
        r"""
        Perform tracking process for different phases (e.g. train / init / track)

        Arguments
        ---------
        target_img: torch.Tensor
            target template image patch
        search_img: torch.Tensor
            search region image patch

        Returns
        -------
        fcos_score_final: torch.Tensor
            predicted score for bboxes, shape=(B, HW, 1)
        fcos_bbox_final: torch.Tensor
            predicted bbox in the crop, shape=(B, HW, 4)
        fcos_cls_prob_final: torch.Tensor
            classification score, shape=(B, HW, 1)
        fcos_ctr_prob_final: torch.Tensor
            center-ness score, shape=(B, HW, 1)
        """
        if phase == 'train':
            # resolve training data
            training_data = args[0]
            target_img = training_data["im_z"]
            search_img = training_data["im_x"]
            # backbone feature
            f_z = self.basemodel(target_img)
            f_x = self.basemodel(search_img)
            f_z = self.fusion(f_z)
            f_x = self.fusion(f_x)
            # feature adjustment
            f_z = self.feat_adjuster_z(f_z)
            f_x = self.feat_adjuster_x(f_x)

            N_out = self.neck(f_x, f_z)

            # head
            cls_fc, bbox_fc, cls_conv, bbox_conv = self.head(N_out)
            predict_data = dict(
                cls_fc=cls_fc,
                bbox_fc=bbox_fc,
                cls_conv=cls_conv,
                bbox_conv=bbox_conv
            )
            if self._hyper_params["corr_feat_output"]:
                predict_data["corr_feat"] = N_out
            return predict_data

        elif phase == 'feature':
            target_img, = args
            # backbone feature
            f_z = self.basemodel(target_img)
            f_z = self.fusion(f_z)

            # template as kernel
            enc_output = self.feat_adjuster_z(f_z)

            # output
            out_list = [enc_output]


        elif phase == 'track':
            if len(args) == 2:
                search_img, enc_output = args
                # backbone feature
                f_x = self.basemodel(search_img)
                f_x = self.fusion(f_x)
                # feature adjustment
                f_x = self.feat_adjuster_x(f_x)
                # output = self.neck.decoder(f_x, enc_output)
                out = self.neck(f_x, enc_output)
                # head
                cls_fc, bbox_fc, cls_conv, bbox_conv = self.head(out, search_img.size(-1))
                # apply sigmoid
                cls_fc = torch.sigmoid(cls_fc)
                cls_conv = torch.sigmoid(cls_conv)
                # merge two cls socres
                cls_score_final = cls_fc + cls_conv * (1 - cls_fc)
                # register extra output
                extra = dict()  # for faster inference
                # extra = {"f_x": f_x, "encoder_output": enc_output, "decoder_output": output}
                # output
                out_list = cls_score_final, bbox_conv, extra
                out_list[-1]

            elif len(args) == 4:
                # c_x, r_x already computed
                c_z_k, r_z_k, c_x, r_x = args
            else:
                raise ValueError("Illegal args length: %d" % len(args))
        else:
            raise ValueError("Phase non-implemented.")
        return out_list

    def update_params(self):
        r"""
        Load model parameters
        """
        self._make_convs()
        # self._initialize_conv()
        super().update_params()

    def _make_convs(self):

        in_channels = self._hyper_params['in_channels']
        mid_channels = self._hyper_params['mid_channels']

        # feature adjustment
        self.feat_adjuster_z = conv_bn_relu(in_channels, mid_channels, kszie=1, has_relu=False, bn_eps=1e-3)
        self.feat_adjuster_x = conv_bn_relu(in_channels, mid_channels, kszie=1, has_relu=False, bn_eps=1e-3)

    def _initialize_conv(self, ):
        conv_weight_std = self._hyper_params['conv_weight_std']
        conv_list = [
            self.r_z_k.conv, self.c_z_k.conv, self.r_x.conv, self.c_x.conv
        ]
        for ith in range(len(conv_list)):
            conv = conv_list[ith]
            torch.nn.init.normal_(conv.weight,
                                  std=conv_weight_std)  # conv_weight_std=0.01

    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)


@TRACK_TASKMODELS.register
@VOS_TASKMODELS.register
class MySiamTrack_fusion(ModuleBase):
    r"""
    SiamTrack model for tracking

    Hyper-Parameters
    ----------------
    pretrain_model_path: string
        path to parameter to be loaded into module
    head_width: int
        feature width in head structure
    """

    default_hyper_params = dict(
        pretrain_model_path="",
        in_channels=768,
        mid_channels=512,
        conv_weight_std=0.01,
        corr_feat_output=False,
        amp=False,
        in_channels_adjuster=96
    )

    def __init__(self, backbone, fusion,neck, head, loss=None):
        super(MySiamTrack_fusion, self).__init__()
        self.basemodel = backbone
        self.fusion = fusion
        self.neck = neck
        self.head = head
        self.loss = loss



    def forward(self, *args, phase="train"):
        r"""
        Perform tracking process for different phases (e.g. train / init / track)

        Arguments
        ---------
        target_img: torch.Tensor
            target template image patch
        search_img: torch.Tensor
            search region image patch

        Returns
        -------
        fcos_score_final: torch.Tensor
            predicted score for bboxes, shape=(B, HW, 1)
        fcos_bbox_final: torch.Tensor
            predicted bbox in the crop, shape=(B, HW, 4)
        fcos_cls_prob_final: torch.Tensor
            classification score, shape=(B, HW, 1)
        fcos_ctr_prob_final: torch.Tensor
            center-ness score, shape=(B, HW, 1)
        """
        if phase == 'train':
            # resolve training data
            training_data = args[0]
            target_img = training_data["im_z"]
            search_img = training_data["im_x"]
            # backbone feature
            f_z = self.basemodel(target_img)
            f_x = self.basemodel(search_img)
            f_z = self.fusion(f_z)
            f_x = self.fusion(f_x)
            # feature adjustment
            f_z = self.feat_adjuster_z(f_z)
            f_x = self.feat_adjuster_x(f_x)
            N_out = self.neck(f_x, f_z)

            # head
            cls_fc, bbox_fc, cls_conv, bbox_conv = self.head(N_out)
            predict_data = dict(
                cls_fc=cls_fc,
                bbox_fc=bbox_fc,
                cls_conv=cls_conv,
                bbox_conv=bbox_conv
            )
            if self._hyper_params["corr_feat_output"]:
                predict_data["corr_feat"] = N_out
            return predict_data

        elif phase == 'feature':
            target_img, = args
            # backbone feature
            f_z = self.basemodel(target_img)
            f_z = self.fusion(f_z)
            # template as kernel
            f_z = self.feat_adjuster_z(f_z)



            # output
            out_list = [f_z]


        elif phase == 'track':
            if len(args) == 2:
                search_img,f_z = args
                # backbone feature
                f_x = self.basemodel(search_img)
                f_x = self.fusion(f_x)
                # feature adjustment
                f_x = self.feat_adjuster_x(f_x)
                # output = self.neck.decoder(f_x, enc_output)
                out = self.neck(f_x, f_z)
                # head
                cls_fc, bbox_fc, cls_conv, bbox_conv = self.head(out, search_img.size(-1))
                # apply sigmoid
                cls_fc = torch.sigmoid(cls_fc)
                cls_conv = torch.sigmoid(cls_conv)
                # merge two cls socres
                cls_score_final = cls_fc + cls_conv * (1 - cls_fc)
                # register extra output
                extra = dict()  # for faster inference
                # extra = {"f_x": f_x, "encoder_output": enc_output, "decoder_output": output}
                # output
                # cls_score_final = cls_score_final.view(19,19)
                # plt.imshow(cls_score_final.cpu().detach().numpy())
                # plt.tight_layout()
                # plt.show()
                out_list = cls_score_final, bbox_conv, extra
                out_list[-1]

            elif len(args) == 4:
                # c_x, r_x already computed
                c_z_k, r_z_k, c_x, r_x = args
            else:
                raise ValueError("Illegal args length: %d" % len(args))
        else:
            raise ValueError("Phase non-implemented.")
        return out_list

    def update_params(self):
        r"""
        Load model parameters
        """
        self._make_convs()
        # self._initialize_conv()
        super().update_params()

    def _make_convs(self):

        in_channels = self._hyper_params['in_channels']
        mid_channels = self._hyper_params['mid_channels']

        # feature adjustment
        self.feat_adjuster_z = conv_bn_relu(in_channels, mid_channels, kszie=1, has_relu=False, bn_eps=1e-3)
        self.feat_adjuster_x = conv_bn_relu(in_channels, mid_channels, kszie=1, has_relu=False, bn_eps=1e-3)

    def _initialize_conv(self, ):
        conv_weight_std = self._hyper_params['conv_weight_std']
        conv_list = [
            self.r_z_k.conv, self.c_z_k.conv, self.r_x.conv, self.c_x.conv
        ]
        for ith in range(len(conv_list)):
            conv = conv_list[ith]
            torch.nn.init.normal_(conv.weight,
                                  std=conv_weight_std)  # conv_weight_std=0.01

    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)
