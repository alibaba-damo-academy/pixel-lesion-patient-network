# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Copyright (c) Medical AI Lab, Alibaba DAMO Academy

"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from ..mask2former.point_sample import (
    get_uncertain_point_coords_with_randomness,
    point_sample, calculate_uncertainty
)
from ..mask2former.criterion import multi_cls_focal_loss


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        # num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        # num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")

    return loss


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class MyCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    YK: try not use matcher if not set
    """

    def __init__(self, num_classes, weight_dict, eos_coef,
                 num_points, oversample_ratio, importance_sample_ratio, cfg):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.hard_sample_ratio = cfg.loss.hard_sample_ratio
        self.fg_sample_ratio = cfg.loss.fg_sample_ratio
        self.hard_neg_sample_ratio = cfg.loss.hard_neg_sample_ratio
        self.deep_supervision = cfg.transformer_predictor.deep_supervision
        self.eos_coef = eos_coef
        self.cfg = cfg
        # self.empty_label = self.num_classes
        # assert self.num_classes == self.cfg.num_all_classes

        # empty_weight = torch.ones(self.num_classes + 1)
        # empty_weight[self.empty_label] = eos_coef
        # self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = torch.cat(outputs, dim=0).float()
        if len(src_logits) == 0:
            return 0

        target_classes = torch.hstack(targets).long()
        target_classes[target_classes < 0] = -target_classes[target_classes<0]-1
        keep = target_classes != self.cfg.ignore_class_label
        src_logits = src_logits[keep]
        target_classes = target_classes[keep]
        if self.cfg.loss.cls_type == 'focal':
            loss_focal = multi_cls_focal_loss(src_logits.transpose, target_classes)
            losses = {"clsf_focal": loss_focal}
        else:
            # empty_weight = torch.ones(self.num_classes + 1).to(src_logits.device)
            # empty_weight[self.empty_label] = self.eos_coef
            # loss_ce = F.cross_entropy(src_logits, target_classes, empty_weight)
            loss_ce = F.cross_entropy(src_logits, target_classes)
            losses = {"clsf_ce": loss_ce}
        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        # losses = {"clsf_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, labels):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        labels = torch.hstack(labels).long()
        # keep = (labels != self.empty_label) & (labels != self.cfg.ignore_class_label)
        keep = (labels >= 0) & (labels != self.cfg.ignore_class_label)
        src_masks = torch.cat(outputs, dim=0).float()[keep]
        target_masks = torch.cat(targets, dim=0).float()[keep]
        if len(src_masks) == 0:
            return {"mask_ce": torch.tensor(0).to(keep.device).to(torch.float),
                    "dice": torch.tensor(0).to(keep.device).to(torch.float),}

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                target_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
                self.hard_sample_ratio,
                self.fg_sample_ratio,
                self.hard_neg_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "mask_ce": sigmoid_ce_loss_jit(point_logits, point_labels),
            "dice": dice_loss_jit(point_logits, point_labels),
        }

        del src_masks
        del target_masks
        return losses

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num_masks = sum(len(t["labels"]) for t in targets[0])
        # num_masks = torch.as_tensor(
        #     [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        # )
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_masks)
        # num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        losses.update(self.loss_labels(outputs['pred_logits'], targets[0]))
        losses.update(self.loss_masks(outputs['pred_masks'], targets[1], targets[0]))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs and self.deep_supervision:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                losses1 = {}
                losses1.update(self.loss_labels(aux_outputs['pred_logits'], targets[0]))
                losses1.update(self.loss_masks(aux_outputs['pred_masks'], targets[1], targets[0]))
                losses1 = {k + f"_{i}": v for k, v in losses1.items()}
                losses.update(losses1)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            # "matcher: {}".format(self.matcher.__repr__(_repr_indent=8) if self.matcher is not None else 'None'),
            # "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            # "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
