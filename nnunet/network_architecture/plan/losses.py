# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import SimpleITK as sitk
import torch
from skimage import measure

from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2

from torch import nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from configs.config_utils import cfg as cfg_all

from utils.my_utils import instance_to_semantic, get_ROI
from ...training.loss_functions.crossentropy import RobustCrossEntropyLoss
from ...training.loss_functions.dice_loss import SoftDiceLoss, get_tp_fp_fn_tn


class BDC_and_CE_loss(nn.Module):
    """Balanced DC: use Tversky loss to enhance FG and also balance lesions of different sizes"""
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1,
                 lesion_channels=[], ignore_label=None, return_sep=True):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(BDC_and_CE_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label
        self.return_sep = return_sep
        self.lesion_channels = lesion_channels

        self.dc = Balanced_Tversky(lesion_channels, **soft_dice_kwargs)

    def forward(self, net_output, instance_target, labels):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param instance_target:
        :return:
        """
        semantic_target = instance_to_semantic(instance_target, labels)
        if self.ignore_label is not None:
            assert semantic_target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = semantic_target != self.ignore_label
            semantic_target[~mask] = 0
            mask = mask.float()
            # self.ignored_pxs += (mask == 0).sum()
        else:
            mask = None

        dc_loss = self.dc(net_output, instance_target, semantic_target, labels, loss_mask=mask) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, semantic_target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.return_sep:
            return self.weight_ce * ce_loss, self.weight_dice * dc_loss
        else:
            return self.weight_ce * ce_loss + self.weight_dice * dc_loss


class Balanced_Tversky(nn.Module):
    def __init__(self, lesion_channels=[], batch_dice=False, do_bg=True, smooth=1,
                 delta=.7, **kwargs):
        """
        """
        super(Balanced_Tversky, self).__init__()
        self.lesion_channels = lesion_channels
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.smooth = smooth
        self.delta = delta

    def forward(self, x, instance_target, semantic_target, labels, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        semantic_target = semantic_target.long()
        y_onehot = torch.zeros(x.shape, device=x.device)
        y_onehot.scatter_(1, semantic_target, 1)
        x = x.softmax(1)[:, self.lesion_channels]
        y_onehot = y_onehot[:, self.lesion_channels]

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y_onehot, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + 2*(1-self.delta) * fp + 2*self.delta * fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        # if not self.do_bg:
        #     if self.batch_dice:
        #         dc = dc[1:]
        #     else:
        #         dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class BalancedFgLoss(nn.Module):
    """Compute fg loss for each lesion gt area and compute mean loss to improve recall"""
    def __init__(self, num_lesion_cls):
        super(BalancedFgLoss, self).__init__()
        self.num_lesion_cls = num_lesion_cls
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, instance_gt, instance_labels, semantic_gt):
        losses_CE, sizes = [], []
        for b, instance_label in enumerate(instance_labels):
            sel = (instance_label[:, 1] > 0) & (instance_label[:, 1] <= self.num_lesion_cls)
            lesion_idxs = instance_label[sel, 0]
            losses1 = []
            for li in lesion_idxs:
                m1 = instance_gt[b, 0] == li
                size = m1.sum()
                if size < 10:
                    continue
                logits = pred[b, :, m1].T
                gt = semantic_gt[b, :, m1][0].long()
                losses1.append(self.criterion(logits, gt))
                sizes.append(size)
            wt = torch.tensor(sizes)
            wt = wt**.5
            wt = wt/wt.mean()
            losses1 = [l*w for l, w in zip(losses1, wt)]
            if len(losses1) > 0:
                losses_CE.append(sum(losses1)/len(losses1))
        if len(losses_CE) == 0:
            return torch.tensor(0.)
        return sum(losses_CE)/len(losses_CE)


class LesionDiceLoss(nn.Module):
    """Crop each lesion gt area and compute mean loss to improve recall"""
    def __init__(self, cfg, num_lesion_cls, criterion):
        super(LesionDiceLoss, self).__init__()
        self.CE_weight = cfg.CE_weight
        self.DC_weight = cfg.DC_weight
        self.pad = cfg.ROI_pad_px
        self.num_lesion_cls = num_lesion_cls
        self.criterion = criterion

    def forward(self, pred, gt):
        min_CC_size = 10
        gt_numpy = gt.cpu().numpy()
        gt_numpy[gt_numpy > self.num_lesion_cls] = 0
        losses_CE, losses_DC = [], []
        for b in range(len(gt)):
            CCs, num_CC = measure.label(gt_numpy[b, 0], connectivity=2, return_num=True)
            for c in range(num_CC):
                roi = CCs == c+1
                if roi.sum() < min_CC_size:
                    continue
                r, (pred1, gt1) = get_ROI(roi, self.pad, gt.shape[2:], (pred[b], gt[b]))
                loss_CE, loss_DC = self.criterion(pred1[None], gt1[None])
                losses_CE.append(loss_CE*self.CE_weight)
                losses_DC.append(loss_DC*self.DC_weight)
        if len(losses_CE) == 0:
            return torch.tensor(0.), torch.tensor(0.)
        return sum(losses_CE)/len(losses_CE), sum(losses_DC)/len(losses_DC)


class InstanceDiscLoss(nn.Module):
    def __init__(self, temp=0.3, eps=1e-5, weight=1, labels=[], sampling=1000):
        super(InstanceDiscLoss, self).__init__()
        self.temp = temp
        self.eps = eps
        self.labels = labels
        self.weight = weight
        self.sampling = sampling

    def forward(self, mask_features, target_mask, instance_labels):
        """
        Make different lesions has different features, each px in one lesion (liver) should be closer to ft of this lesion
        than others
        """
        losses = []
        # mask_features = F.normalize(mask_features, dim=1)  # q c
        bs, _, d, h, w = mask_features.shape
        for b in range(bs):
            lesion_indices = [i for i, l in instance_labels[b] if l in self.labels]
            num_CC = len(lesion_indices)
            if num_CC <= 1:
                continue
            # CCs = torch.stack([target_mask[b, 0]==i for i in lesion_indices])  # q d h w
            CCs = torch.zeros(num_CC, d, h, w).to(torch.bool).to(mask_features.device)
            for q in range(num_CC):
                idxs = torch.where(target_mask[b, 0]==lesion_indices[q])
                if len(idxs[0]) > self.sampling:
                    sel = torch.randperm(len(idxs[0])).to(CCs.device)[:self.sampling]
                    idxs = [idx[sel] for idx in idxs]
                CCs[q, idxs[0], idxs[1], idxs[2]] = 1
            # eqn 16
            ft = mask_features[b]  # c d h w
            t = torch.einsum('cdhw,qdhw->cq', ft.float(), CCs.float()/CCs.sum()).T
            t = F.normalize(t, dim=-1)  # q c
            logits = torch.einsum('qc,cdhw->qdhw', t.half(), ft) / self.temp  # q d h w
            denominator = torch.logsumexp(logits, dim=0).half()  # d h w

            nominator = torch.zeros_like(denominator)#.requires_grad_()
            for q in range(num_CC):
                nominator[CCs[q]] = logits[q][CCs[q]]
            loss_px = -nominator+denominator
            loss = loss_px[CCs.max(0)[0]].mean()
            losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0).to(mask_features.device).to(torch.float)
        return sum(losses)/len(losses) * self.weight
