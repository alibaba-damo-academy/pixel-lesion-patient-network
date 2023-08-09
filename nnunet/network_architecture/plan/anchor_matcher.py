# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional

import numpy as np
import torch
from skimage import measure
from torch import nn, Tensor
from torch.nn import functional as F, Conv3d

from utils.my_utils import instance_to_semantic


class AnchorMatcher(nn.Module):
    def __init__(self, num_classes, cfg):
        super().__init__()
        self.num_classes = num_classes
        self.min_CC_size = cfg.anchor_matcher.min_CC_size
        self.max_CC_num = cfg.anchor_matcher.max_CC_num
        self.lesion_label_groups = cfg.anchor_matcher.lesion_label_groups  # cls 1~8 is lesion, cls 7 is cyst
        self.all_lesion_labels = list(range(1, cfg.num_lesion_classes+1))
        self.num_lesion_classes = cfg.num_lesion_classes
        self.organ_labels = [0]+list(range(cfg.num_lesion_classes+1, cfg.num_all_classes))
        self.CC_th = cfg.anchor_matcher.CC_th
        self.query_do_L2 = False
        self.must_have_anchor = True
        self.cfg = cfg

    def gen_anchor_query(self, pred_logits, pixel_embs):
        with torch.no_grad():
            CCs = self.get_lesion_CCs(pred_logits)
        pixel_embs = [p.detach() for p in pixel_embs]  # if don't detach, very slow
        anchor_queries, anchor_labels, CCs = self.extract_CC_fts(pixel_embs, CCs)
        return anchor_queries, anchor_labels, CCs

    def get_lesion_CCs(self, seg_outputs):
        batch_size, c, d, h, w = seg_outputs.shape
        final_CCs = [[] for _ in range(batch_size)]
        # organ_CCs = [[] for _ in range(batch_size)]
        softmax = seg_outputs.softmax(1)

        for b in range(batch_size):
            for label_group in self.lesion_label_groups:
                mask = softmax[b, label_group].sum(0) > self.CC_th
                mask = mask.cpu().numpy()
                CCs, num_CC = measure.label(mask, connectivity=2, return_num=True)
                CCs = torch.from_numpy(CCs).to(seg_outputs.device)
                sizes = torch.tensor([(CCs==i+1).sum() for i in range(num_CC)])
                keep = torch.where(sizes > self.min_CC_size)[0]
                if len(keep) > self.max_CC_num:# and self.training:
                    ord = torch.argsort(sizes[keep], descending=True)
                    # ord = torch.randperm(len(sizes[keep])).to(keep.device)
                    keep = keep[ord[:self.max_CC_num]]
                final_CCs[b].extend([(-1, CCs==i+1) for i in keep])

            seg_pred = softmax[b].argmax(0)
            # don't extract CCs for organs, treat each organ as one component
            for organ_label in self.organ_labels:
                mask = seg_pred == organ_label
                if torch.any(mask):
                    final_CCs[b].append((organ_label, mask))

            if len(final_CCs[b]) == 0 and self.must_have_anchor:
                mask = softmax[b, self.all_lesion_labels].sum(0)
                th = torch.max(mask.ravel())
                final_CCs[b].extend([(-1, mask == th)])

        return final_CCs

    def extract_CC_fts(self, ftmaps, CCs):
        fts = []
        assert np.all(np.diff(np.vstack([ft.shape[2:] for ft in ftmaps]), axis=0) <= 0), \
            'shape of ftmaps should be from large to small'
        ft_dim = sum([ft.shape[1] for ft in ftmaps])
        for b in range(len(CCs)):
            fts_single_im = []
            CC = CCs[b]
            num_CCs = len(CC)
            if num_CCs == 0:
                fts.append(torch.empty(0, ft_dim, device=ftmaps[0].device, dtype=ftmaps[0].dtype))
            else:
                for l in range(len(ftmaps)):
                    ftmap = ftmaps[l][b]
                    # if CC.shape[1:] != ftmap.shape[1:]:
                    #     CC = F.interpolate(CC[None], size=ftmap.shape[1:], mode="nearest")[0, 0]
                    fts_single_level = []
                    for i in range(num_CCs):
                        CC_label, this_CC = CC[i]
                        if this_CC.sum() > 0:
                            ft_CC = ftmap[:, this_CC].mean(1)
                        else:  # maybe this CC is too small, it is lost during downsampling of CC
                            center = torch.tensor(
                                [coord.to(torch.float).mean() for coord in torch.where(CCs[b] == i + 1)])
                            scale = torch.tensor(CCs[b].shape) / torch.tensor(ftmap.shape[1:])
                            center = (center / scale).int()
                            ft_CC = ftmap[:, center[0], center[1], center[2]]
                        if self.query_do_L2:
                            ft_CC = F.normalize(ft_CC, p=2, dim=0)
                        fts_single_level.append(ft_CC)
                    fts_single_im.append(torch.vstack(fts_single_level))
                fts.append(torch.hstack(fts_single_im))
        anchor_labels = [[c[0] for c in CC] for CC in CCs]
        CCs = [[c[1] for c in CC] for CC in CCs]
        return fts, anchor_labels, CCs

    def prepare_targets_w_anchor(self, anchor_CCs, anchor_labels, target, index_to_labels):
        """Match each pred anchor CC to a gt CC"""
        labels, masks = [], []
        shape = target.shape[2:]
        device = target.device
        prec, rec, nprop = [], [], []
        # empty_label = self.num_classes if self.pred_organ else 0
        # assert self.pred_organ
        for b in range(len(target)):
            num_anchor_lesion_CCs = (np.array(anchor_labels[b]) < 0).sum()
            lesion_index_to_label = {i: l for i, l in index_to_labels[b]
                                     if (l > 0 and l <= self.num_lesion_classes) or l == self.cfg.ignore_class_label}
            index_to_label = {i: l for i, l in index_to_labels[b]}
            num_gt_CCs = len(lesion_index_to_label)
            hit = np.zeros((num_gt_CCs,))
            nprop.append(num_anchor_lesion_CCs)

            gt_mask = target[b, 0].int()
            label, mask = [], []
            for i, anchor_label in enumerate(anchor_labels[b]):
                if anchor_label >= 0:
                    if not self.pred_organ:
                        continue
                    label1 = anchor_label
                    mask1 = anchor_CCs[b][i]
                else:
                    gt_idxs = gt_mask[anchor_CCs[b][i]]
                    gt_idx_sizes = torch.hstack([(gt_idxs == i).sum() for i, l in index_to_labels[b]])
                    hit_idx = torch.argmax(gt_idx_sizes)
                    label1 = index_to_label[index_to_labels[b][hit_idx, 0]]
                    if num_gt_CCs == 0:
                        label1 = -label1-1  # to be considered in cls loss but ignored in seg loss
                        mask1 = anchor_CCs[b][i] * 0
                    else:
                        gt_idxs = gt_mask[anchor_CCs[b][i]]
                        # CC_size = this_CC.sum()
                        gt_idx_sizes = torch.hstack([(gt_idxs == i).sum() for i in lesion_index_to_label.keys()])
                        if gt_idx_sizes.max() == 0:
                            label1 = -label1-1
                            mask1 = anchor_CCs[b][i] * 0
                        else:
                            hit_idx = torch.argmax(gt_idx_sizes)
                            gt_idx = list(lesion_index_to_label.keys())[hit_idx]
                            label1 = lesion_index_to_label[gt_idx]
                            mask1 = gt_mask == gt_idx
                            hit[hit_idx] = 1
                label.append(label1)
                mask.append(mask1)
            label = torch.tensor(label, device=device).long()
            mask = torch.stack(mask)
            prec += [hit.sum() / num_anchor_lesion_CCs if num_anchor_lesion_CCs > 0 else np.nan]
            rec += [hit.sum() / num_gt_CCs if num_gt_CCs > 0 else np.nan]
            labels.append(label)
            masks.append(mask)
        self.precision = torch.tensor(np.nanmean(prec) if not np.all(np.isnan(prec)) else np.nan)
        self.recall = torch.tensor(np.nanmean(rec) if not np.all(np.isnan(rec)) else np.nan)
        self.nprop = torch.tensor(np.mean(nprop))
        return labels, masks
