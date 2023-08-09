# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F, Conv3d

from nnunet.network_architecture.plan.losses import BDC_and_CE_loss, BalancedFgLoss
from utils.my_utils import instance_to_semantic
from configs.config_utils import cfg as cfg_all


class SegHead(nn.Module):
    def __init__(self, num_classes, nnunet_loss, cfg):
        super().__init__()
        self.num_classes = num_classes
        self.emb_norm = nn.LayerNorm(cfg.ft_dim)
        self.cfg = cfg
        self.pred = nn.Sequential()
        for i in range(self.cfg.conv_layers):
            conv = nn.Conv3d(cfg.ft_dim, cfg.ft_dim, 1)
            self.pred.append(conv)
            self.pred.append(nn.ReLU())
        self.pred.append(nn.Conv3d(cfg.ft_dim, num_classes, 1))

        self.loss_type = cfg.loss.get('type', 'nnunet')
        if self.loss_type == 'nnunet':
            nnunet_loss.return_sep = True  # to observe DC and CE loss separately
            self.criterion = nnunet_loss
        elif self.loss_type == 'Tversky':
            BDC_args = {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False}
            BDC_args.update(cfg.loss)
            self.criterion = BDC_and_CE_loss(BDC_args, {'reduction': 'none'},
                                   ignore_label=100, lesion_channels=list(range(1, cfg_all.num_lesion_classes+1)))
        elif self.loss_type == 'fg':
            nnunet_loss.return_sep = True  # to observe DC and CE loss separately
            self.criterion_nnunet = nnunet_loss
            self.criterion_fg = BalancedFgLoss(cfg_all.num_lesion_classes)

    def forward(self, mask_features):
        output_mask = self.pred(mask_features)
        return output_mask

    def loss(self, outputs, targets, labels):
        if self.loss_type == 'nnunet':
            semantic_target = instance_to_semantic(targets[0], labels)
            CE_loss, DC_loss = self.criterion(outputs, semantic_target)
            losses = {'seg_CE': CE_loss * self.cfg.loss.weight, 'seg_DC': DC_loss * self.cfg.loss.weight}
        elif self.loss_type == 'Tversky':
            CE_loss, DC_loss = self.criterion(outputs, targets[0].int(), labels)
            losses = {'seg_CE': CE_loss * self.cfg.loss.weight, 'seg_DC': DC_loss * self.cfg.loss.weight}
        elif self.loss_type == 'fg':
            semantic_target = instance_to_semantic(targets[0], labels)
            CE_loss, DC_loss = self.criterion_nnunet(outputs, semantic_target)
            fg_loss = self.criterion_fg(outputs, targets[0].int(), labels, semantic_target)
            losses = {'seg_CE': CE_loss * self.cfg.loss.weight, 'seg_DC': DC_loss * self.cfg.loss.weight,
                      'seg_fg': fg_loss * self.cfg.loss.weight * self.cfg.loss.fg_weight}
        return losses
