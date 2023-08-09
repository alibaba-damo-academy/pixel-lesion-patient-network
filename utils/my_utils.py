# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
from typing import Tuple

import GPUtil as GPUtil
import numpy as np
import torch


def assign_idle_GPU(cfg):
    if cfg.gpu == '':
        deviceIDs = GPUtil.getAvailable(order='lowest', limit=1, maxMemory=.2)
        if len(deviceIDs) == 0:
            deviceIDs = GPUtil.getAvailable(order='lowest', limit=1, maxMemory=.9, maxLoad=1)
        gpu = deviceIDs[0]
        cfg.gpu = gpu
    torch.cuda.set_device(torch.device(int(cfg.gpu)))

    return cfg.gpu


def instance_to_semantic(masks_ins, labels, label_maps={}):
    masks_sem = masks_ins * 0
    for b in range(len(labels)):
        for instance_id, class_label in labels[b]:
            new_label = label_maps[class_label] if class_label in label_maps else class_label
            masks_sem[b, masks_ins[b] == instance_id] = new_label
    return masks_sem


def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """
    coord = []
    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim
    coord = torch.stack(coord[::-1], dim=-1)
    return coord


def get_ROI(mask, pad=(0,)*3, max_shape=(np.inf,)*3, tensors=()):
    r = []
    if isinstance(mask, torch.Tensor):
        for axis in range(3):
            reduced = mask.max(1)[0].max(1)[0] if axis == 0 else mask.max(2)[0].max(0)[0] if axis == 1 else mask.max(0)[0].max(0)[0]
            idxs = torch.where(reduced > 0)[0]
            r.extend([max(0, idxs.min().item() - pad[axis]), min(idxs.max().item() + pad[axis], max_shape[axis] - 1)])
    else:
        for axis in range(3):
            reduced = mask.max(1).max(1) if axis==0 else mask.max(2).max(0) if axis==1 else mask.max(0).max(0)
            idxs = np.where(reduced > 0)[0]
            r.extend([max(0, idxs.min()-pad[axis]), min(idxs.max()+pad[axis], max_shape[axis]-1)])
    if len(tensors) > 0:
        rois = [t[:, r[0]: r[1] + 1, r[2]: r[3] + 1, r[4]: r[5] + 1] for t in tensors]
        return r, rois
    else:
        return r
