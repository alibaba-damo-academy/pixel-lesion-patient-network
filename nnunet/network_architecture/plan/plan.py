# Copyright (c) Medical AI Lab, Alibaba DAMO Academy

import SimpleITK as sitk
import torch
from nnunet.network_architecture.attn_unet.my_generic_UNet import MTUNet
from .cls_head import ClsHead
from .det_head import DetHead
from .seg_head import SegHead

from torch import nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from configs.config_utils import cfg as cfg_all


class PLAN(MTUNet):
    """Pixel-Lesion-pAtient Network"""
    def __init__(self, nnunet, cfg, nnunet_loss):
        super(MTUNet, self).__init__()
        self.unet = nnunet
        self.conv_op = self.unet.conv_op  # for validation
        self.unet_fixed = False
        self.cfg = cfg
        self.num_classes = self.cfg.num_all_classes
        self.num_lesion_classes = self.cfg.num_lesion_classes  # classes 1~num_lesion_classes are lesions, num_lesion_classes+1~num_classes are organs
        self.do_ds = self.cfg.nnunet_deep_supervision
        self.pred_scale_factor = {1: 1, 2: [1, 2, 2]}[cfg.prediction_stride]
        self.add_anchor_query = cfg.add_anchor_query

        if 'seg' in cfg.components:
            self.seg_head = SegHead(self.num_classes, nnunet_loss, cfg.seg)

        if 'det' in cfg.components:
            num_cls_det = self.num_classes
            cfg.det.num_all_classes = cfg.num_all_classes
            cfg.det.num_lesion_classes = cfg.num_lesion_classes
            cfg.det.ignore_class_label = cfg.ignore_class_label
            self.det_head = DetHead(num_cls_det, self.add_anchor_query, cfg.det)

        if 'cls' in cfg.components:
            self.cls_head = ClsHead(cfg)
            self.num_classes = self.cfg.num_all_classes + self.cls_head.output_num
        if cfg.get('consist_loss', None) is not None:
            self.consist_loss = cfg.consist_loss

    def forward(self, x, online_val=False):
        """Will be called during validation.
        return logits of each class, to be compatible to nnUNet's output format"""
        outputs = self.get_all_outputs(x)
        pred_mask = self.inference(outputs)
        return pred_mask

    def get_all_outputs(self, x):
        pixel_embs = self.unet(x)
        outputs = {}
        if 'seg' in self.cfg.components:
            outputs['seg'] = self.seg_head(pixel_embs[0])
        if 'det' in self.cfg.components and ('train' not in cfg_all.mode or cfg_all.cur_epoch >= self.cfg.det.train_start_ep):
            seg_pred = outputs['seg'] if self.add_anchor_query else None
            outputs['det'] = self.det_head(self.unet, seg_pred)
        if 'cls' in self.cfg.components and ('train' not in cfg_all.mode or cfg_all.cur_epoch >= self.cfg.cls.train_start_ep):
            outputs['cls'] = self.cls_head(self.unet)

        return outputs

    def loss(self, outputs, targets, labels, properties):
        """
        Numbers in `targets` are instance IDs;
        `labels` stores the mapping from each instance ID to class ID.
        """
        if not self.cfg.nnunet_deep_supervision:
            targets = [targets[int(np.log2(self.cfg.prediction_stride))]]
        # labels may contain mask ids that has been cropped in targets
        labels = [np.vstack([(i, l) for i, l in label if (mask==i).sum() > 0])
                      for mask, label in zip(targets[0], labels)]
        losses = {}

        if 'seg' in self.cfg.components:
            seg_losses = self.seg_head.loss(outputs['seg'], targets, labels)
            losses.update(seg_losses)

        if 'det' in self.cfg.components and cfg_all.cur_epoch >= self.cfg.det.train_start_ep:
            det_losses = self.det_head.loss(outputs['det'], targets, labels)
            losses.update(det_losses)

        if 'cls' in self.cfg.components and cfg_all.cur_epoch >= self.cfg.cls.train_start_ep:
            cls_losses = self.cls_head.loss(outputs['cls'], targets, labels)
            losses.update(cls_losses)

        if hasattr(self, 'consist_loss') and cfg_all.cur_epoch >= self.cfg.consist_loss.train_start_ep:
            losses.update(self.compute_consist_loss(outputs))
        l = sum(losses.values())
        for k in list(losses.keys()):  # remove loss items with _x to reduce printing
            if k[-2] == '_' and k[-1].isdigit():
                losses.pop(k)

        return l, losses

    def compute_consist_loss(self, outputs):
        if self.consist_loss.type == 'det_cls':
            cls = outputs['cls'].sigmoid()
            det = torch.vstack([l.softmax(1).max(0)[0] for l in outputs['det']['pred_logits']])
            det = det[:, 1:self.num_lesion_classes+1]
            cls = cls[:, -self.num_lesion_classes:]
            loss = ((det-cls.detach())**2).sum()
            loss += ((det.detach()-cls)**2).sum()
            return {'consist': loss * self.consist_loss.weight / 2}

    def inference(self, outputs):
        if 'det' in self.cfg.components and ('train' in cfg_all.mode and cfg_all.cur_epoch < self.cfg.det.train_start_ep):
            result = outputs['seg'].softmax(1)
            return result
        # if det branch exists, use segmentation result from det branch. Otherwise, use that from seg branch
        if 'det' in self.cfg.components:
            result = self.det_head.inference(outputs['det'])
        elif 'seg' in self.cfg.components:
            result = outputs['seg'].softmax(1)

        if 'cls' in self.cfg.components and not ('train' in cfg_all.mode and cfg_all.cur_epoch < self.cfg.cls.train_start_ep):
            # cat cls result in last few channels of the segmentation result, following Yingda
            class_prob = outputs['cls'].sigmoid()
            b, _, w, h, d = result.shape
            classprob2concat = class_prob[:, :, None, None, None].expand(b, self.cls_head.output_num, w, h, d)
            result = torch.cat([result, classprob2concat], dim=1)
        return result

    def _internal_predict_3D_3Dconv_tiled(self, *args, **kwargs):
        return super(MTUNet, self)._internal_predict_3D_3Dconv_tiled(*args, **kwargs)

    def _internal_maybe_mirror_and_pred_3D(self, *args, **kwargs):
        return super(MTUNet, self)._internal_maybe_mirror_and_pred_3D(*args, **kwargs)
