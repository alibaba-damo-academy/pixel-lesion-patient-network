# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import os
import os.path as osp
import shutil
from bisect import bisect

import torch
from skimage import measure
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
from sklearn.metrics import classification_report, confusion_matrix

from nnunet.evaluation.evaluation_utils import gt_ignore_label, class_names, malig_benign_cyst_groups, num_lesion_cls

radius_points = np.array([5, 10, 20])
volume_points = radius_points**3*np.pi*4/3
ovlp_metric = 'dice'
ovlp_th = .2
print_fun = print


def draw_conf_mat(labels_gt, labels_pred, labels_annotated):
    from sklearn import metrics
    import matplotlib.pyplot as plt
    import seaborn as sn
    conf = confusion_matrix(labels_gt, labels_pred)
    conf_norm = confusion_matrix(labels_gt, labels_pred, normalize='true')
    df_cm = pd.DataFrame(conf, index=labels_annotated[:-1], columns=labels_annotated[:-1])
    df_cmn = pd.DataFrame(conf_norm, index=labels_annotated[:-1], columns=labels_annotated[:-1])
    plt.figure(figsize=(7, 3))
    sn.heatmap(df_cmn, annot=df_cm, cmap='jet', cbar=False, fmt='.0f')
    # sn.heatmap(df_cmn, annot=True, cmap='jet', cbar=False, fmt='.2f')
    plt.savefig('confmat_num.png')


def visualize_det_acc(data):
    # ovlps, pred_sizes, pred_classes, pred_scores, gt_sizes, gt_classes = stats
    all_ovlps = [stat[0] for stat in data]
    all_scores = [stat[3] for stat in data]
    all_pred_classes = [stat[2] for stat in data]
    all_gt_classes = [stat[5] for stat in data]
    all_gt_volumes = [np.searchsorted(volume_points, stat[4].astype(int)) for stat in data]

    stats = np.vstack((np.hstack(all_gt_classes), np.hstack(all_gt_volumes)))
    classes = sorted(list(np.unique(stats[0])))
    # classes = [1,2,3]
    cnt = np.zeros((len(classes), len(volume_points)+1), dtype=int)
    for i in range(stats.shape[1]):
        cnt[classes.index(stats[0, i]), stats[1, i]] += 1
    print_fun('class and size count', class_names+['ignore'], '\n', radius_points)
    print_fun(cnt.T, '\n', cnt[:-1].sum(0), '\n', cnt.sum(1))

    summary = (f"DET: {len(data)} images, {sum([len(b) for b in all_ovlps])} predictions, "
                f"{sum([len(b) for b in all_gt_classes])} gts\n")
    ignore_masks = [cls == gt_ignore_label for cls in all_gt_classes]
    sens, fp_per_vol, precisions, sens_per_class = \
        FROC_mask(all_ovlps, all_scores, ovlp_th, ignore_masks, classes, all_gt_classes)
    summary += f"sens {sens[-1]:.4f} at FP {fp_per_vol[-1]:.4f} and precision {precisions[-1]:.4f}, "
    summary += f"mean sens of class 1~{num_lesion_cls} {sens_per_class[-1, :num_lesion_cls].mean():.4f}\n"
    summary += f"sens of each class at this point {sens_per_class[-1]}\n"

    num_sizes = len(volume_points)+1
    size_classes = list(range(num_sizes))
    sens, fp_per_vol, precisions, sens_per_size = \
        FROC_mask(all_ovlps, all_scores, ovlp_th, ignore_masks, size_classes, all_gt_volumes)
    summary += f"sens of radius {radius_points} at this point {sens_per_size[-1]}\n"

    # sens of each cls and each size
    both_classes = list(range(num_sizes * len(classes)))
    all_class_mapped = [np.array([classes.index(c) for c in c1]) for c1 in all_gt_classes]
    all_gts_both = [c*num_sizes+s for c, s in zip(all_class_mapped, all_gt_volumes)]
    sens, fp_per_vol, precisions, sens_per_both = \
        FROC_mask(all_ovlps, all_scores, ovlp_th, ignore_masks, both_classes, all_gts_both)
    print_fun('sens of each class at each size')
    print_fun(sens_per_both[-1].reshape((-1, num_sizes)).T)

    print_fun(summary)

    print_fun('>>>>>> lesion-wise classification acc')
    summary1 = evaluate_clsf(all_gt_classes, all_pred_classes, all_ovlps, ovlp_th, ignore_masks)
    summary += summary1
    return summary


def evaluate_clsf(gt_classes, pred_classes, ovlps, ovlp_th, ignore_masks):
    labels_gt, labels_pred = [], []
    for i in range(len(gt_classes)):
        if len(gt_classes[i]) == 0 or len(pred_classes[i]) == 0:
            continue
        keep = (ovlps[i].max(axis=1) >= ovlp_th)
        matched_gt_idx = ovlps[i].argmax(axis=1)
        label_gt = gt_classes[i][matched_gt_idx][keep]
        label_pred = pred_classes[i][keep]
        keep = (~ignore_masks[i][matched_gt_idx[keep]])
        labels_gt.append(label_gt[keep])
        labels_pred.append(label_pred[keep])
    labels_gt = np.hstack(labels_gt)
    labels_pred = np.hstack(labels_pred)

    classes = np.arange(1, len(class_names)+1)
    print_fun(classification_report(labels_gt, labels_pred, labels=classes, target_names=class_names))
    res = classification_report(labels_gt, labels_pred, labels=classes, target_names=class_names, output_dict=True)
    summary = f"{num_lesion_cls} cls: lesion cls acc={res['accuracy']:.4f}, mean F1={res['macro avg']['f1-score']:.4f}"
    print_fun(summary)
    conf = confusion_matrix(labels_gt, labels_pred, normalize=None)
    # draw_conf_mat(labels_gt, labels_pred, class_names)
    print_fun(np.round(conf).astype(int))

    for i, cs in enumerate(malig_benign_cyst_groups):
        for c in cs:
            labels_gt[labels_gt == c] = i+1
            labels_pred[labels_pred == c] = i+1
    valid_label_mask = labels_gt <= len(malig_benign_cyst_groups)
    labels_gt = labels_gt[valid_label_mask]
    labels_pred = labels_pred[valid_label_mask]
    labels_pred[labels_pred > len(malig_benign_cyst_groups)] = 1  # set all others prediction as malig

    print_fun('malig vs benign vs cyst')
    labels, target_names = [1, 2, 3], ['malig', 'benign', 'cyst']
    print_fun(classification_report(labels_gt, labels_pred, labels=labels, target_names=target_names))
    res = classification_report(labels_gt, labels_pred, labels=labels, target_names=target_names, output_dict=True)
    summary1 = f"3-cls: lesion cls acc={res['accuracy']:.4f}, mean F1={res['macro avg']['f1-score']:.4f}"
    print_fun(summary1)
    conf = confusion_matrix(labels_gt, labels_pred, normalize=None)
    print_fun(np.round(conf).astype(int))
    return summary+'\n'+summary1


def evaluate_clsf_v2(gt_classes, pred_classes, ovlps, ovlp_th, ignore_masks, classes, cls_names):
    """If a gt is not detected, set clsf res as wrong, i.e. don't ignore lesions that are missed"""
    labels_gt, labels_pred = [], []
    for i in range(len(gt_classes)):
        if len(gt_classes[i]) == 0:
            continue
        label_gt = gt_classes[i]
        if len(pred_classes[i]) > 0:
            matched_pred_idx = ovlps[i].argmax(axis=0)
            label_pred = pred_classes[i][matched_pred_idx]
            label_pred[ovlps[i].max(axis=0) == 0] = 0
        else:
            label_pred = np.zeros((len(label_gt),))
        keep = (~ignore_masks[i])
        labels_gt.append(label_gt[keep])
        labels_pred.append(label_pred[keep])
    labels_gt = np.hstack(labels_gt)
    labels_pred = np.hstack(labels_pred)
    print_fun(classification_report(labels_gt, labels_pred, labels=classes, target_names=cls_names))
    res = classification_report(labels_gt, labels_pred, labels=classes, target_names=cls_names, output_dict=True)
    accuracy = (labels_gt==labels_pred).sum()/len(labels_gt)
    summary = f"lesion cls acc={accuracy:.4f}, mean F1={res['macro avg']['f1-score']:.4f}"
    conf = confusion_matrix(labels_gt, labels_pred, normalize=None)
    print_fun(conf)
    return summary


def FROC_mask(ovlps_all, scores_all, overlap_th=.3, ignore_masks=None, classes=None, gt_classes=None):
    """Compute the Free ROC curve of given overlaps and scores of all detections.
    Supports ignored gts. If a box hits an ignored gt, it is neither counted as TP or FP.
    When a box hits N gts, num_hit_gts is increased by N (corresponding to recall) but tps is increased by 1
    (corresponding to precision).
    If a box hits a gt that has been hit before, this box should be ignored, so neither num_hit_gts nor tps increases"""
    nImg = len(ovlps_all)
    img_idxs = np.hstack([[i]*len(ovlps_all[i]) for i in range(nImg)]).astype(int)
    ovlps_cat = [o for ovlps in ovlps_all for o in ovlps]
    scores = np.hstack(scores_all)
    ord = np.argsort(scores)[::-1]
    ovlps_cat = [ovlps_cat[o] for o in ord]
    img_idxs = img_idxs[ord]
    nBox = len(ovlps_cat)

    nHits = 0
    if gt_classes:
        # classes = np.unique(np.hstack(gt_classes))
        # classes = sorted(list(classes))
        nClasses = len(classes)
        nHitsPerClass = np.zeros((nClasses,), dtype=int)
    gt_hits = [np.zeros((ovlps.shape[1],), dtype=bool) for ovlps in ovlps_all]
    box_types = []
    num_hit_gts = []
    num_hit_gts_per_class = []
    for i in range(nBox):
        overlaps = ovlps_cat[i]
        if len(overlaps) == 0 or np.any(np.isnan(overlaps)) or (overlaps.max() < overlap_th):
            this_box = 'FP'
        elif overlaps.max() >= overlap_th and np.all(ignore_masks[img_idxs[i]][overlaps >= overlap_th]):  # hit ignored gt
            this_box = 'ignore'
        else:
            this_box = 'ignore'  # if box duplicate hit gt that has been hit before, this box should be ignored
            for j in range(len(overlaps)):
                if overlaps[j] >= overlap_th:
                    if not gt_hits[img_idxs[i]][j]:
                        gt_hits[img_idxs[i]][j] = True
                        this_box = 'TP'
                        nHits += 1
                        if gt_classes:
                            this_class = gt_classes[img_idxs[i]][j]
                            nHitsPerClass[classes.index(this_class)] += 1

        box_types.append(this_box)
        num_hit_gts.append(nHits)
        if gt_classes:
            num_hit_gts_per_class.append(nHitsPerClass.copy())

    nGt = sum(~np.hstack(ignore_masks))
    num_hit_gts, box_types = [np.array(l) for l in (num_hit_gts, box_types)]
    sens = num_hit_gts / nGt
    fps = np.cumsum(box_types == 'FP')
    fp_per_img = fps / nImg
    tps = np.cumsum(box_types == 'TP')  # tps may be smaller than num_hit_gts if one tp hits multiple gts
    num_non_ignored_boxes = np.cumsum(box_types != 'ignore')
    precisions = tps / num_non_ignored_boxes
    if gt_classes:
        num_hit_gts_per_class = np.vstack(num_hit_gts_per_class)
        gt_classes_all, ignore_mask_all = np.hstack(gt_classes), np.hstack(ignore_masks)
        nGtPerClass = np.array([((gt_classes_all==c) & (~ignore_mask_all)).sum() for c in classes])
        sens_per_class = num_hit_gts_per_class / nGtPerClass
        return sens, fp_per_img, precisions, sens_per_class
    return sens, fp_per_img, precisions


def det_stat_single(pred, gt, spacing, volume_th_mm):
    """Compute the overlap matrix of each CC, also return the size and class of each CC"""
    pixel_size = np.prod(spacing)
    pred_cc, num_pred = measure.label(pred, connectivity=2, return_num=True)
    gt_cc, num_gt = measure.label(gt, connectivity=2, return_num=True)
    pred_sizes = np.array([(pred_cc==i+1).sum() * pixel_size for i in range(num_pred)])
    pred_classes = np.array([pred[pred_cc==i+1][0] for i in range(num_pred)], dtype=int)
    pred_scores = pred_classes * 0 + 1  # currently we don't have scores
    gt_sizes = np.array([(gt_cc==i+1).sum() * pixel_size for i in range(num_gt)])
    gt_classes = np.array([gt[gt_cc==i+1][0] for i in range(num_gt)], dtype=int)
    pred_keep = np.where(pred_sizes >= volume_th_mm)[0]
    gt_keep = gt_sizes >= volume_th_mm
    gt_classes[~gt_keep] = gt_ignore_label
    ovlps = []

    for i in pred_keep:
        pred_i = np.where(pred_cc==i+1)
        pred_gt_ovlp = np.array([(gt_cc[pred_i] == c+1).sum()*pixel_size for c in range(num_gt)])
        if ovlp_metric == 'dice':
            ovlp = pred_gt_ovlp*2/(pred_sizes[i]+gt_sizes)
        elif ovlp_metric == 'recall':
            ovlp = pred_gt_ovlp/(gt_sizes)
        elif ovlp_metric == 'precision':
            ovlp = pred_gt_ovlp/(pred_sizes[i])
        ovlps.append(ovlp)

    # ovlps = (np.vstack(ovlps) if len(pred_keep) > 0 else np.empty((0, num_gt)))[:, gt_keep]
    # return ovlps, pred_sizes[pred_keep], pred_classes[pred_keep], pred_scores[pred_keep], \
    #        gt_sizes[gt_keep], gt_classes[gt_keep]
    ovlps = np.vstack(ovlps) if len(pred_keep) > 0 else np.empty((0, num_gt))
    return ovlps, pred_sizes[pred_keep], pred_classes[pred_keep], pred_scores[pred_keep], \
           gt_sizes, gt_classes
