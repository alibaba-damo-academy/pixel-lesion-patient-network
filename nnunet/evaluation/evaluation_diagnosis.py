# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
'''
Given two folders of gt and predicted masks, compute pixel-wise, lesion-wise, and patient-wise accuracies for
    diagnosis.
Postprocess each CC to have only one label, except cyst can be touching other lesions.
Pixel-wise: Dice, precision, and recall of each class, each patient. Also compute overall fg vs bg accuracy.
Lesion-wise: Compute detection and classification accuracy separately, stratified on lesion class and size.
Patient-wise: Multi-label acc, based on doctor annotated mask, since one patient can have multiple diseases
    Two results: 1. All 8 classes; 2. Benign vs. malignant vs. cyst
    2 ways to compute patient-wise score of each CT: mask size and class score (from the patient branch)
'''
import sys
import time
from datetime import datetime
from multiprocessing.pool import ThreadPool

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, load_json
from easydict import EasyDict as edict
import torch

from nnunet.evaluation import evaluation_utils, evaluation_det_utils
from scipy.spatial.distance import cdist
from skimage import measure
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import os.path as osp
import os
import csv
import shutil
from sklearn.metrics import roc_curve, auc

from .evaluation_utils import Logger, process_gt_mask, \
    compute_metrics, num_lesion_classes_in_annot, gt_ignore_label, lesion_volume_th, merge_gt_class, num_lesion_cls, \
    compute_sens_on_classes, det_lesion_volume_th, class_names, malig_benign_cyst_groups, fmt, t, class_channel_dict, \
    gen_case_report
from .evaluation_det_utils import det_stat_single, visualize_det_acc
from configs.config_utils import cfg


print_fun = print
stats_fn = 'acc_stats_diagnosis.pth'
use_cache_stats = True


def eval_diagnosis(pred_folder, gt_folder):
    """
    pred_folder: A folder containing post-processed lesion masks (only lesion labels) and json files of patient-wise
        predictions.
    gt_folder: A folder of gt masks.
    """
    global print_fun
    print_fun = Logger(pred_folder).print_to_log_file
    evaluation_det_utils.print_fun = print_fun
    evaluation_utils.print_fun = print_fun
    print_fun('Evaluating', pred_folder)
    pids = sorted([pid[:-7] for pid in os.listdir(pred_folder) if pid.endswith('.nii.gz')])
    np.set_printoptions(4, suppress=True)

    stats_path = osp.join(pred_folder, stats_fn)
    if not use_cache_stats or not osp.exists(stats_path):
        accs_patient = edict(
            mask_size=[],
            class_score=[],
            gt=[]
        )
        accs_det_seg = edict(
            seg_binary=edict(dices=[], precisions=[], recalls=[]),
            seg_classes=edict(dices=[], precisions=[], recalls=[]),
            det=edict(stats=[]),
        )
        for pid in tqdm(pids):
            has_gt_flag = True
            try:
                gt = sitk.ReadImage(osp.join(gt_folder, pid + '.nii.gz'))
                spacing = gt.GetSpacing()
                gt = sitk.GetArrayFromImage(gt)
                gt[(gt > num_lesion_classes_in_annot) & (gt != gt_ignore_label)] = 0
                gt = process_gt_mask(gt, spacing)
            except:  # if don't have gt file, assume it is a normal case and don't have annotation
                gt = np.zeros((1,))  # fake empty gt
                print(pid, 'gt reading failed')
                has_gt_flag = False

            pred = sitk.ReadImage(osp.join(pred_folder, pid + '.nii.gz'))
            spacing = pred.GetSpacing()
            pred = sitk.GetArrayFromImage(pred)

            ### CLS
            gt_labels, pred_labels = clsf_acc_single(pred, gt, spacing, num_lesion_cls)
            accs_patient.gt.append(gt_labels)
            accs_patient.mask_size.append(pred_labels)
            clsf_pred_fn = osp.join(pred_folder, pid + '.json')
            class_pred = load_json(clsf_pred_fn)
            if 'class_scores' in class_pred:
                class_scores = class_pred['class_scores']
                accs_patient.class_score.append(class_scores)

            if has_gt_flag:  # compute seg and det accuracy
                ### SEG
                # multi-class seg results
                dice, precision, recall = seg_acc_single(pred * (gt != gt_ignore_label), gt, num_lesion_cls)
                accs_det_seg.seg_classes.dices.append(dice)
                accs_det_seg.seg_classes.precisions.append(precision)
                accs_det_seg.seg_classes.recalls.append(recall)

                # all lesions as one class
                dice, precision, recall = seg_acc_single((pred > 0) & (gt != gt_ignore_label),
                                                         (gt > 0) & (gt != gt_ignore_label), 1)
                accs_det_seg.seg_binary.dices.append(dice)
                accs_det_seg.seg_binary.precisions.append(precision)
                accs_det_seg.seg_binary.recalls.append(recall)

                # DET
                det_stats = det_stat_single(pred, gt, spacing, det_lesion_volume_th)
                accs_det_seg.det.stats.append(list(det_stats)+[pid])

        torch.save((accs_patient, accs_det_seg), stats_path)
    else:
        accs_patient, accs_det_seg = torch.load(stats_path)

    summary = ''
    print_fun('>>>>>> seg acc')
    summary1 = visualize_seg_acc(accs_det_seg, class_names)
    summary += summary1+'\n'
    print_fun('>>>>>> det acc')
    summary1 = visualize_det_acc(accs_det_seg.det.stats)
    summary += summary1 + '\n'

    if len(accs_patient.class_score) > 0:
        print_fun('>>>>>> clsf acc by class score')
        summary1 = visualize_clsf_acc(accs_patient.class_score, accs_patient.gt, 'class score')
        summary += summary1 + '\n'
    print_fun('>>>>>> clsf acc by mask size')
    summary1 = visualize_clsf_acc(accs_patient.mask_size, accs_patient.gt, 'mask size')
    summary += summary1 + '\n'
    print_fun()
    print_fun(cfg.exp_name)
    print_fun(summary)

    path = osp.join(pred_folder, 'case_report.xlsx')
    gen_case_report(accs_det_seg, path)


def seg_acc_single(pred, gt, num_cls):
    dices, precisions, recalls = [], [], []
    for c in range(num_cls):
        pred1 = (pred == c + 1)
        gt1 = gt == c + 1
        p_area = pred1.sum()
        g_area = gt1.sum()
        intersect = (pred1 & gt1).sum()
        dice = intersect * 2 / (p_area + g_area)
        precision = intersect / (p_area)
        recall = intersect / (g_area)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)
    # conf_mat = confusion_matrix(gt.ravel(), pred.ravel(), labels=np.arange(9)) if compute_conf_mat else 0
    return dices, precisions, recalls


def clsf_acc_single(pred, gt, spacing, num_lesion_cls):
    gt_labels = np.array([int((gt == cls + 1).sum()) for cls in range(num_lesion_cls)]) * np.prod(spacing)
    pred_labels = np.array([int((pred == cls + 1).sum()) for cls in range(num_lesion_cls)]) * np.prod(spacing)
    return gt_labels, pred_labels


def visualize_seg_acc(accs, class_names):
    summary = 'SEG: '
    if 'seg_classes' in accs:
        print_fun('\tprecision\trecall\tDice')
        res = [np.nanmean(accs.seg_classes.precisions, axis=0),
               np.nanmean(accs.seg_classes.recalls, axis=0),
               np.nanmean(accs.seg_classes.dices, axis=0)]
        for c in range(num_lesion_cls):
            print_fun(class_names[c], t, fmt % res[0][c], fmt % res[1][c], fmt % res[2][c], )
        print_fun('Avg', t, fmt % np.mean(res[0]), fmt % np.mean(res[1]), fmt % np.mean(res[2]), )
        summary += f"cls avg Dice={np.mean(res[2]):.4f}, prec={np.mean(res[0]):.4f}, rec={np.mean(res[1]):.4f}\n"
    print_fun('Binary', t,
          fmt % np.nanmean(accs.seg_binary.precisions),
          fmt % np.nanmean(accs.seg_binary.recalls),
          fmt % np.nanmean(accs.seg_binary.dices), )
    summary += f"binary Dice={np.nanmean(accs.seg_binary.dices):.4f}, " \
              f"prec={np.nanmean(accs.seg_binary.precisions):.4f}, " \
              f"rec={np.nanmean(accs.seg_binary.recalls):.4f}; " \

    return summary


def visualize_clsf_acc(preds, gts, type):
    """
    1. malignant vs. non-malig, benign vs. non-benign
    2. each class, pos vs. neg
    """
    gts = np.vstack(gts)
    preds = np.vstack(preds)

    print_fun('\t#gt \t#pred\tacc \tsens\tspec\tprec\tAUC')
    print_fun(f'malignant vs. non, and benign vs. non')
    class_names1 = ['malig', 'benign']
    gts_mb = np.vstack([gts[:, np.array(lbs)-1].sum(1) for lbs in malig_benign_cyst_groups[:-1]]).T
    if type == 'class score':  # variable channels
        preds_mb = np.vstack([preds[:, np.array(class_channel_dict[lb])].sum(1) for lb in class_names1]).T
    elif type == 'mask size':
        preds_mb = np.vstack([preds[:, np.array(lbs)-1].sum(1) for lbs in malig_benign_cyst_groups[:-1]]).T
    summary2 = compute_metrics(gts_mb, preds_mb, class_names1)

    print_fun()
    print_fun(f'each type of tumor vs non')
    if type == 'class score':  # variable channels
        preds_cls = preds[:, class_channel_dict['each_class']]
    elif type == 'mask size':
        preds_cls = preds
    summary3 = compute_metrics(gts, preds_cls, class_names)

    summary = f'CLS by {type}: malig/benign '+summary2 + f'\n{num_lesion_cls}-class '+summary3
    return summary

