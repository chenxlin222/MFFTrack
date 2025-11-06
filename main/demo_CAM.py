from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

from PIL import Image

sys.path.append('../')

import argparse
import cv2
import torch
from glob import glob
import torchvision.utils as vutils

from CAM import get_CAM

import numpy as np

import torch.nn as nn
torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='SiamCAR CAM demo')
parser.add_argument('--config', type=str, default='D:\my_code\\SiamCAR-CAM-master\\experiments\\siamcar_r50\\config.yaml', help='config file')
parser.add_argument('--dataset_dir', type=str,# '',
        help='datasets')
parser.add_argument('--dataset', type=str, default='GOT10k',
        help='datasets')
parser.add_argument('--snapshot', type=str, default='D:\my_code\\SiamCAR-CAM-master\\snapshot\\got10k_model.pth', help='model name')
parser.add_argument('--CAM_name', type=str, default='GradCAM', help='model name')
# softmax, head.cls_logits, head.centerness, head.bbox_tower.0, backbone.layer4.2, ....
parser.add_argument('--register_layer', default='softmax', type=str, help=' View the parameter names of each layer corresponding to the printed model')
parser.add_argument('--video_name', default='', type=str, help='videos or image files')
parser.add_argument('--format', default='bmp', type=str, help='png, pdf, jpg, bmp')
parser.add_argument('--save_dir', default='./test', type=str, help='Save path')

args = parser.parse_args()

def show_cam(img, mask, title=None, title2=None):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        img (Tensor): shape (1, 3, H, W)
        mask (Tensor): shape (1, 1, H, W)
    Return:
        heatmap (Tensor): shape (3, H, W)
        cam (Tensor): synthesized GradCAM cam of the same shape with heatmap.
        :param title:
    """
    mask = (mask - mask.min()).div(mask.max() - mask.min()).data
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze().float()), cv2.COLORMAP_JET)  # [H, W, 3]
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)

    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # cam = heatmap + img.cpu()
    img = torch.from_numpy(img)
    # cam = 1 * (1 - mask ** 0.8) * img + (mask ** 0.8) * heatmap
    cam = 0.9 * img + 0.1 * heatmap
    im = out_image(cam)

    return im

def out_image(
    tensor,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = vutils.make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # im = Image.fromarray(ndarr)
    return ndarr

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)

        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = sorted(glob(os.path.join(video_name, '*.jp*')))
        for img in images:
            frame = cv2.imread(img)
            yield frame

class Model(nn.Module):
    def __init__(self, model):
        super(Model,self).__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def template(self, z):
        self.model.template(z)

    def track(self, x):
        outputs = self.model.track(x)
        outputs['cls'] = self.softmax(outputs['cls'])
        return outputs

def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).to(device)
    print(model)
    model = Model(model)
    # build tracker
    tracker = SiamCARTracker(model, cfg.TRACK)
    CAM = get_CAM(args.CAM_name, model=tracker, target_layer=args.register_layer)
    if args.dataset == "GOT10k":
        dataset = "GOT10k"
    else:
        dataset = args.dataset
    params = getattr(cfg.HP_SEARCH, dataset)
    hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2]}
    dataset_root = os.path.join('D:\my_code\\testing_datasets', args.dataset)
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)


    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        for v_idx, video in enumerate(dataset):
            if args.video_name != '':
                # test one special video
                if video.name != args.video_name:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                               gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    CAM.model.init(img, gt_bbox_)
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs, img_crop, pred_bbox = CAM(img, hp)
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5  # skip 5 frames
                        lost_number += 1
                    if outputs is not None:
                        heatmap_path = os.path.join(args.save_dir, args.CAM_name, "heatmap", video.name)
                        if os.path.exists(heatmap_path) is False:
                            os.makedirs(heatmap_path)
                        fusion_path = os.path.join(args.save_dir, args.CAM_name, "fusion", video.name)
                        if os.path.exists(fusion_path) is False:
                            os.makedirs(fusion_path)
                        show_cam(img_crop, outputs, fusion_path + "/{:06d}.{}".format(idx, args.format),
                                 heatmap_path + "/{:06d}.{}".format(idx, args.format))
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
            toc /= cv2.getTickFrequency()
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                v_idx + 1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("total lost: {:d}".format(total_lost))
    else:
        first_frame = True
        if args.video_name:
            video_name = args.video_name.split('/')[-1].split('.')[0]
        else:
            video_name = 'webcam'
        cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
        for frame in get_frames(args.video_name):
            if first_frame:
                try:
                    init_rect = cv2.selectROI(video_name, frame, False, False)
                except:
                    exit()
                cx, cy, w, h = get_axis_aligned_bbox(np.array(init_rect))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                CAM.model.init(frame, gt_bbox_)
                # tracker.init(frame, init_rect)
                first_frame = False
                idx = 0
            else:
                idx= idx+1
                outputs, img_crop, bbox = CAM(frame, hp)
                if outputs is not None:
                    heatmap_path = os.path.join(args.save_dir, args.CAM_name, "heatmap")
                    if os.path.exists(heatmap_path) is False:
                        os.makedirs(heatmap_path)
                    fusion_path = os.path.join(args.save_dir, args.CAM_name, "fusion")
                    if os.path.exists(fusion_path) is False:
                        os.makedirs(fusion_path)
                    im = show_cam(img_crop, outputs, fusion_path + "/{:06d}.{}".format(idx, args.format),
                             heatmap_path + "/{:06d}.{}".format(idx, args.format))
                # outputs = tracker.track_cam(frame, hp)
                # bbox = list(map(int, outputs['bbox']))
                # cv2.rectangle(im, (bbox[0], bbox[1]),
                #               (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                #               (0, 255, 0), 3)
                cv2.imshow(video_name, im)
                cv2.waitKey(40)
        # for v_idx, video in enumerate(dataset):
        #     if args.video_name != '':
        #         # test one special video
        #         if video.name != args.video_name:
        #             continue
        #     toc = 0
        #     track_times = []
        #     for idx, (img, gt_bbox) in enumerate(video):
        #         tic = cv2.getTickCount()
        #         if idx == 0:
        #             cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
        #             gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
        #             CAM.model.init(img, gt_bbox_)
        #         else:
        #             outputs, img_crop, bbox = CAM(img, hp)
        #             if outputs is not None:
        #                 heatmap_path = os.path.join(args.save_dir, args.CAM_name, "heatmap", video.name)
        #                 if os.path.exists(heatmap_path) is False:
        #                     os.makedirs(heatmap_path)
        #                 fusion_path = os.path.join(args.save_dir,args.CAM_name, "fusion", video.name)
        #                 if os.path.exists(fusion_path) is False:
        #                     os.makedirs(fusion_path)
        #                 show_cam(img_crop, outputs, fusion_path + "/{:06d}.{}".format(idx, args.format),
        #                          heatmap_path + "/{:06d}.{}".format(idx, args.format))
        #                 # scores.append(outputs['best_score'])
        #
        #         toc += cv2.getTickCount() - tic
        #         track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
        #         if idx == 0:
        #             cv2.destroyAllWindows()
        #     toc /= cv2.getTickFrequency()
        #     # save results
        #     print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
        #         v_idx + 1, video.name, toc, idx / toc))
if __name__ == '__main__':
    main()
