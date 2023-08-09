# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import SimpleITK as sitk
import torch
from skimage import measure

from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss3
from nnunet.network_architecture.attn_unet.my_generic_UNet import MTUNet
from utils.my_utils import get_ROI

from torch import nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ..attn_unet.my_attn_unet import DualPathBlock
from configs.config_utils import cfg


class ClsHead(nn.Module):
    """Global classification branch like Yingda's transformer"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # from Yingda's
        self.embed_dim = cfg.cls.embed_dim
        self.depth = 4
        self.num_heads = 8
        self.num_patches = cfg.cls.num_patches  # self.patches[0] * self.patches[1] * self.patches[2]
        self.input_dims = cfg.cls.input_dims
        self.memory = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        self.pixel_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        nn.init.normal_(self.memory)
        nn.init.normal_(self.pixel_pos_embed)
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))s
        self.patch_embeds = nn.ModuleList([
            nn.Conv3d(self.input_dims[0], self.embed_dim, kernel_size=(8, 8, 8), stride=(8, 8, 8)),
            nn.Conv3d(self.input_dims[1], self.embed_dim, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(self.input_dims[2], self.embed_dim, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(self.input_dims[3], self.embed_dim, kernel_size=(8, 8, 8), stride=(8, 8, 8))
        ])
        self.patch_norm = nn.ModuleList([
            nn.InstanceNorm3d(self.embed_dim, affine=True),
            nn.InstanceNorm3d(self.embed_dim, affine=True),
            nn.InstanceNorm3d(self.embed_dim, affine=True),
            nn.InstanceNorm3d(self.embed_dim, affine=True),
        ])
        self.blocks = nn.Sequential(*[
            DualPathBlock(dim=self.embed_dim, num_heads=self.num_heads, drop_path=0.2, drop=0.2)
            for _ in range(self.depth)])
        self.norm = nn.LayerNorm(self.embed_dim)
        self.merge_classes = cfg.cls.merge_classes
        self.output_num = len(self.merge_classes)
        self.classifier = nn.Linear(self.embed_dim, self.output_num)
        size_th = cfg.cls.get('size_th', 100)
        self.criterion = ClassLoss(cfg.num_lesion_classes, self.merge_classes, cfg.ignore_class_label, size_th)
        self.cfg = cfg
        self.weight = cfg.cls.get('weight', 1)

    def forward(self, backbone):
        """return logits of each class"""
        pixel_features = [backbone.encoder_fts[f] if f > 0 else backbone.decoder_fts[f]
                       for f in self.cfg.cls.transformer_in_feature]
        # pixel_features = [backbone.encoder_fts[2], backbone.encoder_fts[4],
        #                   backbone.decoder_fts[2], backbone.decoder_fts[4]]
        bs = len(pixel_features[0])
        memory_input = self.memory.expand(bs, -1, -1)
        for i in range(self.depth):
            pixel_embed = self.patch_norm[i](self.patch_embeds[i](pixel_features[i]))
            pixel_embed = pixel_embed.flatten(2).transpose(1, 2)
            if self.cfg.cls.use_pos_emb:
                pixel_embed = pixel_embed + self.pixel_pos_embed.expand(bs, -1, -1)
            memory_input = self.blocks[i](memory_input, pixel_embed)
        final_feature = memory_input.mean(1)
        class_logits = self.classifier(self.norm(final_feature))
        return class_logits

    def loss(self, outputs, targets, labels):
        loss = self.criterion(outputs, targets[0], labels)
        return {'cls': loss * self.weight}


class ClassLoss(nn.Module):
    def __init__(self, num_lesion_cls=8, merge_classes=[[0]], ignore_class_label=100, size_th=100):
        super(ClassLoss, self).__init__()
        self.num_lesion_cls = num_lesion_cls
        self.merge_classes = [np.array(cls)-1 for cls in merge_classes]
        self.num_out_cls = len(self.merge_classes)
        self.ignore_class_label = ignore_class_label
        self.criterion = nn.BCEWithLogitsLoss()
        self.size_th = size_th
        # self.smooth_loss = LabelSmoothingLoss(3, smoothing=smoothing)

    def forward(self, pred_class, mask_gt, label_gt):
        cnts = torch.zeros(len(pred_class), self.num_lesion_cls)
        for mask_gt1, label_gt1, cnt1 in zip(mask_gt, label_gt, cnts):
            for i, label in label_gt1:
                if 0 < label <= self.num_lesion_cls:
                    cnt1[label-1] += (mask_gt1==i).sum().item()

        all_class_label = cnts.float().to(pred_class.device)
        class_label = torch.hstack([all_class_label[:, cls].sum(dim=1, keepdim=True) for cls in self.merge_classes])
        class_label = (class_label / self.size_th).clamp_max(1)  # continuous label
        loss = self.criterion(pred_class, class_label)
        return loss