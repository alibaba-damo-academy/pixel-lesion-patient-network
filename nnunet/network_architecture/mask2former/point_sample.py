# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
# Copyright (c) Medical AI Lab, Alibaba DAMO Academy

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

# from detectron2.projects.point_rend.point_features import point_sample
from utils.my_utils import unravel_index


def point_sample(input, point_coords, **kwargs):
    """
    Modified by YK to support real 3D coordinates.
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2).unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3).squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(
    coarse_logits, target_masks, uncertainty_func, num_points,
        oversample_ratio, importance_sample_ratio, hard_sample_ratio, fg_sample_ratio, hard_neg_sample_ratio
):
    """
       Modified by YK to support real 3D coordinates.
       YK also added fg_sample_ratio, hard_sample_ratio, and hard_neg_sample_ratio to enhance foreground and
       hard sampling.

 Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 3, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_hard_points = int(hard_sample_ratio * num_points)
    num_fg_points = int(fg_sample_ratio * num_points)
    num_hard_neg_points = int(hard_neg_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points - num_hard_points - num_fg_points - num_hard_neg_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 3)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 3
    )

    if hard_sample_ratio > 0:
        pixel_loss = F.binary_cross_entropy_with_logits(coarse_logits.detach(), target_masks, reduction="none")
        # pixel_loss = (coarse_logits.detach() - target_masks).abs()
        shape = np.array(pixel_loss.shape[2:])
        num_rand_hard = max(shape.prod()*.02, num_hard_points*oversample_ratio)
        loss_hard, idx = torch.topk(pixel_loss.view(num_boxes, -1), k=int(num_rand_hard), dim=1)
        idx = idx.cpu().numpy()
        idx_sel = np.stack([np.vstack(np.unravel_index(np.random.choice(idx1, num_hard_points, replace=False), shape)).T
                            / (shape-1) for idx1 in idx])[:, :, ::-1].copy()
        point_coords = torch.cat((torch.from_numpy(idx_sel).to(point_coords.device).to(point_coords.dtype),
                                  point_coords), dim=1)

    if hard_neg_sample_ratio > 0:
        pixel_loss = F.binary_cross_entropy_with_logits(coarse_logits.detach(), target_masks, reduction="none")
        # pixel_loss = (coarse_logits.detach() - target_masks).abs()
        pixel_loss *= 1-target_masks
        shape = torch.tensor(target_masks.shape[2:], device=target_masks.device)
        num_rand_hard_neg = max(shape.prod()*.005, num_hard_neg_points*oversample_ratio)
        loss_hard_neg, idx = torch.topk(pixel_loss.view(num_boxes, -1), k=int(num_rand_hard_neg), dim=1)
        idx_sel = [idx1[torch.randperm(idx1.shape[0], device=idx1.device)][:num_hard_neg_points] for idx1 in idx]
        idx_sel = [unravel_index(idx1, shape) / (shape-1) for idx1 in idx_sel]
        idx_sel = torch.flip(torch.stack(idx_sel), dims=(2,))
        point_coords = torch.cat((idx_sel, point_coords), dim=1)

    if fg_sample_ratio > 0:
        shape = torch.tensor(target_masks.shape[2:], device=target_masks.device)
        idx = []
        for mask in target_masks:
            idx1 = torch.stack(torch.where(mask[0])).T
            if len(idx1) < num_fg_points:
                idx1 = torch.cat((idx1, torch.rand(num_fg_points-len(idx1), 3, device=idx1.device)
                                  *shape[None]), dim=0).to(torch.int)
            idx.append(idx1)

        idx_sel = torch.flip(torch.stack([idx1[torch.randperm(idx1.shape[0], device=idx[0].device)][:num_fg_points]
                            / (shape-1) for idx1 in idx]), dims=(2,))
        point_coords = torch.cat((idx_sel, point_coords), dim=1)

    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 3, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))
