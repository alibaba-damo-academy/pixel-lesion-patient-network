# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
'''
Evaluate tumor screening accuracy, i.e. whether the subject is normal or patient.
We split cases into tumor, normal, and hard normal, by their file names.
We compute sens at different levels of specificities.
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
from nnunet.evaluation.evaluation_utils import compute_metrics, Logger, det_lesion_volume_th, \
    num_lesion_classes_in_annot, gt_ignore_label, process_gt_mask, class_names, compute_sens_on_classes, \
    class_channel_dict

from scipy.spatial.distance import cdist
from skimage import measure
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import os.path as osp
import os
import csv
import shutil
from sklearn.metrics import roc_curve, auc


print_fun = print
stats_fn = 'acc_stats_screening.pth'
use_cache_stats = False
spec_ths = [1, .999, .998, .995, .99]
is_tumor_case = lambda pid: '-' in pid
is_hard_normal_case = lambda pid: ('Steatosis' in pid) or ('sju_normal' in pid)


def eval_sens_spec(pred_folder, gt_folder):
    global print_fun
    print_fun = Logger(pred_folder).print_to_log_file
    print_fun('Evaluating', pred_folder)

    pids = sorted([pid[:-7] for pid in os.listdir(pred_folder) if pid.endswith('.nii.gz')])
    np.set_printoptions(4, suppress=True)
    stats_path = osp.join(pred_folder, stats_fn)

    if not use_cache_stats or not osp.exists(stats_path):
        stats = edict(
            mask_size=[],
            class_score=[],
            gt=[]
        )
        for pid in tqdm(pids):
            if is_tumor_case(pid):  # tumor case, has gt
                gt = sitk.ReadImage(osp.join(gt_folder, pid + '.nii.gz'))
                spacing = gt.GetSpacing()
                gt = sitk.GetArrayFromImage(gt)
                gt[(gt > num_lesion_classes_in_annot)] = 0
                gt = process_gt_mask(gt, spacing)
                gt1 = sum([(gt == lb).sum() for lb in compute_sens_on_classes])

            elif is_hard_normal_case(pid):
                gt1 = -1
            else:
                gt1 = 0
            stats.gt.append(gt1)
            pred = sitk.ReadImage(osp.join(pred_folder, pid + '.nii.gz'))
            spacing = pred.GetSpacing()
            pred = sitk.GetArrayFromImage(pred)
            pred1 = sum([(pred == lb).sum() for lb in compute_sens_on_classes]) * np.prod(spacing)
            stats.mask_size.append(pred1)

            clsf_pred_fn = osp.join(pred_folder, pid + '.json')
            class_pred = load_json(clsf_pred_fn)
            if 'class_scores' in class_pred:
                class_score = np.array(class_pred['class_scores'])[class_channel_dict['tumor_non_cyst']].sum()
                stats.class_score.append(class_score)

        torch.save((stats,), stats_path)
    else:
        stats, = torch.load(stats_path)

    print_fun(f'tumor vs. normal (may contain cyst)')
    gts = np.hstack(stats.gt)
    if len(stats.class_score) > 0:
        print_fun('>>>>>> clsf acc by class score')
        preds = np.hstack(stats.class_score)
        compute_metrics(gts, preds)
    preds = np.hstack(stats.mask_size)
    print_fun('>>>>>> clsf acc by mask size')
    compute_metrics(gts, preds)


def compute_metrics(gts_tumor, preds):
    def metrics(th):
        pred = preds > th
        sens = (pred & gt).sum() / (gt.sum())
        spec = ((pred == 0) & (gt == 0)).sum() / (gt == 0).sum()
        spec1 = ((pred == 0) & (gts_tumor == 0)).sum() / (gts_tumor == 0).sum()
        spec2 = ((pred == 0) & (gts_tumor < 0)).sum() / (gts_tumor < 0).sum()  # hard set
        res = np.array([th, auc1, sens, spec, spec1, spec2])
        print_fun(res)

    gt = gts_tumor > det_lesion_volume_th
    fpr, tpr, threshold = roc_curve(gt, preds)
    auc1 = auc(fpr, tpr)
    idx = np.argmax(tpr - fpr)
    Youden_index = threshold[idx]
    ths = [0, Youden_index]
    print_fun('th\tauc1\tsens\tspec\tspec1\tspec2')
    for spec in spec_ths:
        idx = np.where(1-fpr >= spec)[0][-1]
        ths.append(threshold[idx])
    for th in ths:
        metrics(th)
