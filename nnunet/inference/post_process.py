# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import sys
from copy import deepcopy
from typing import Union, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.spatial.distance import cdist
from skimage import measure

from nnunet.preprocessing.preprocessing import get_lowres_axis, get_do_separate_z, resample_data_or_seg
from batchgenerators.utilities.file_and_folder_operations import *
from torch.nn import functional as F

from configs.config_utils import cfg


def process_pred_mask(mask, spacing):
    num_lesion_cls = cfg.num_lesion_classes
    combine_label_idx = cfg.validation.combine_label_idx

    componentFilter = sitk.ConnectedComponentImageFilter()
    componentFilter.FullyConnectedOn()
    relabelFilter = sitk.RelabelComponentImageFilter()
    relabelFilter.SortByObjectSizeOn()

    liver_dist_th = cfg.validation.remove_lesion_outside_liver
    if liver_dist_th > 0:
        # find the largest CC with liver label as liver
        liver_label = num_lesion_cls+1  # assume the liver label is this
        liver_mask = (0 < mask) & (mask <= liver_label)
        mask_sitk = sitk.GetImageFromArray(liver_mask.astype('int16'))
        obj_label = componentFilter.Execute(mask_sitk)
        obj_relabel = relabelFilter.Execute(obj_label)
        obj_count = relabelFilter.GetNumberOfObjects()
        obj_relabel = sitk.GetArrayFromImage(obj_relabel)
        liver_mask = None
        for l in range(obj_count):
            if np.any(mask[obj_relabel == l+1] == liver_label):
                liver_mask = obj_relabel == l + 1
                break
        if liver_mask is None:
            liver_mask = obj_relabel == 1
        liver_idxs = np.vstack(np.where(liver_mask)).T * np.array(spacing[::-1])
    mask[mask > num_lesion_cls] = 0

    pixel_volume = np.prod(spacing)
    mask_sitk = sitk.GetImageFromArray((mask > 0).astype('int16'))

    obj_label = componentFilter.Execute(mask_sitk)
    obj_relabel = relabelFilter.Execute(obj_label)
    obj_count = relabelFilter.GetNumberOfObjects()
    obj_relabel = sitk.GetArrayFromImage(obj_relabel)
    mask_new = mask * 0

    for i in range(obj_count):
        m1 = obj_relabel == i + 1
        if liver_dist_th > 0:
            # if this CC is far from liver, discard
            center = np.mean(np.vstack(np.where(m1)).T * np.array(spacing[::-1]), axis=0)
            d = cdist(center[None], liver_idxs).min()
            if d > liver_dist_th:
                continue

        volume = m1.sum()
        # density = im[m1].mean() if im is not None else density_th+1  # this is designed to remove lesion predictions
        # in the air, which is rare if we crop liver patches as inputs now
        if volume * pixel_volume >= cfg.validation.lesion_volume_th:# and density >= density_th:
            labels = mask[m1]
            lb_cnt = np.array([np.sum(labels==i) for i in range(num_lesion_cls+1)])
            lb_cnt[np.setdiff1d(np.arange(num_lesion_cls+1), combine_label_idx)] = 0
            # if lb_cnt.sum()-lb_cnt[7] > 0 and lb_cnt[7] > 0:
            #     lb_cnt[7] = 0
            dominant_lb = lb_cnt.argmax()
            for idx in np.where(lb_cnt > 0)[0]:
                if idx in combine_label_idx:
                    labels[labels == idx] = dominant_lb
            mask_new[m1] = labels
    return mask_new


def resample_softmax(segmentation_softmax, properties_dict, force_separate_z, verbose,
                     interpolation_order_z, order, resampled_npz_fname, region_class_order):
    current_shape = segmentation_softmax.shape
    shape_original_after_cropping = properties_dict.get('size_after_cropping')
    # current_spacing = dct.get('spacing_after_resampling')
    # original_spacing = dct.get('original_spacing')

    if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(shape_original_after_cropping))]):
        if force_separate_z is None:
            if get_do_separate_z(properties_dict.get('original_spacing')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            elif get_do_separate_z(properties_dict.get('spacing_after_resampling')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('spacing_after_resampling'))
            else:
                do_separate_z = False
                lowres_axis = None
        else:
            do_separate_z = force_separate_z
            if do_separate_z:
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            else:
                lowres_axis = None

        if lowres_axis is not None and len(lowres_axis) != 1:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False

        if verbose: print("separate z:", do_separate_z, "lowres axis", lowres_axis)
        # from time import time
        # t = time()
        if not cfg.validation.torch_interpolate:
            seg_old_spacing = resample_data_or_seg(segmentation_softmax, shape_original_after_cropping, is_seg=False,
                                                   axis=lowres_axis, order=order, do_separate_z=do_separate_z,
                                                   order_z=interpolation_order_z)
        else:
            seg_old_spacing = \
            F.interpolate(torch.from_numpy(segmentation_softmax[None]),  # much faster, but small lesion worse
                          size=shape_original_after_cropping, mode='trilinear').numpy()[0]
        # print(time()-t)
        # seg_old_spacing = resize_softmax_output(segmentation_softmax, shape_original_after_cropping, order=order)
    else:
        if verbose: print("no resampling necessary")
        seg_old_spacing = segmentation_softmax

    if resampled_npz_fname is not None:
        np.savez_compressed(resampled_npz_fname, softmax=seg_old_spacing.astype(np.float16))
        # this is needed for ensembling if the nonlinearity is sigmoid
        if region_class_order is not None:
            properties_dict['regions_class_order'] = region_class_order
        save_pickle(properties_dict, resampled_npz_fname[:-4] + ".pkl")
    return seg_old_spacing


def post_proc_softmax(seg_old_spacing, region_class_order, properties_dict):
    """Extract and save patient wise score (Yingda's) if exists; convert softmax to seg map;
    put result into bbox of cropping if exists"""
    if 'postproc_class_remap' in cfg:
        remap = {int(k): v for k, v in cfg.postproc_class_remap.items()}
        softmax_new = np.zeros((max(remap.keys())+1, *(seg_old_spacing.shape[1:])))
        for new_lb, ori_lbs in remap.items():
            for ori_lb in ori_lbs:
                softmax_new[new_lb] += seg_old_spacing[ori_lb]
        seg_old_spacing = softmax_new
    if seg_old_spacing.shape[0] > cfg.num_all_classes:
        # this means patient-wise classification score is concatenated after softmax
        seg_old_spacing, pat_cls_score = get_patient_wise_class_score(seg_old_spacing)
    else:
        pat_cls_score = None
    softmax = seg_old_spacing
    if region_class_order is None:
        seg_old_spacing = seg_old_spacing.argmax(0)  # traditional method
    else:
        seg_old_spacing_final = np.zeros(seg_old_spacing.shape[1:])
        for i, c in enumerate(region_class_order):
            seg_old_spacing_final[seg_old_spacing[i] > 0.5] = c
        seg_old_spacing = seg_old_spacing_final

    bbox = properties_dict.get('crop_bbox')
    if bbox is not None:
        shape_original_before_cropping = properties_dict.get('original_size_of_raw_data')
        seg_old_size = np.zeros(shape_original_before_cropping, dtype=np.uint8)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing
    return seg_old_size, softmax, pat_cls_score


def get_patient_wise_class_score(seg_old_spacing):
    num_seg_cls = cfg.num_all_classes
    lesion_cls_num = cfg.num_lesion_classes
    cls_pred = seg_old_spacing[num_seg_cls:]
    seg_old_spacing = seg_old_spacing[:num_seg_cls]
    # im_mean = seg_old_spacing.mean((1, 2, 3))
    seg_old_spacing_cls = seg_old_spacing.argmax(0)
    lesion_mean = cls_pred[:, (seg_old_spacing_cls > 0) & (seg_old_spacing_cls <= lesion_cls_num)]
    if lesion_mean.shape[1] > 0:
        lesion_mean = lesion_mean.mean(1)
    else:
        lesion_mean = np.zeros((cls_pred.shape[0],))
    return seg_old_spacing, lesion_mean.tolist()


def get_patient_wise_lesion_info(softmax, seg, properties_dict):
    pixel_mm3 = np.prod(properties_dict.get('original_spacing'))
    num_seg_cls = cfg.num_all_classes
    lesion_cls_num = cfg.num_lesion_classes
    lesion_pred = (seg > 0) & (seg <= lesion_cls_num)
    CCs, num_CC = measure.label(lesion_pred, connectivity=2, return_num=True)
    lesion_size_mm3 = [(CCs == c + 1).sum()*pixel_mm3 for c in range(num_CC)]
    class_size_mm3 = [(seg == c + 1).sum()*pixel_mm3 for c in range(lesion_cls_num)]
    lesion_info = dict(
        class_size_mm3=class_size_mm3,
        lesion_size_mm3=lesion_size_mm3,
    )
    return lesion_info
