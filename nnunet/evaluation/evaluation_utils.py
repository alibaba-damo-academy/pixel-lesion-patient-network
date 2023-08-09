# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import sys
import time
from datetime import datetime
from multiprocessing.pool import ThreadPool

import SimpleITK as sitk
import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from easydict import EasyDict as edict
import torch
from scipy.spatial.distance import cdist
from skimage import measure
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import os.path as osp
import os
import csv
import shutil
from sklearn.metrics import roc_curve, auc
from configs.config_utils import cfg


print_fun = print
fmt = '\t%.4f'
t = '\t'
num_lesion_cls = cfg.num_lesion_classes
lesion_volume_th = cfg.validation.lesion_volume_th
merge_gt_class = cfg.data_loader.merge_classes if 'merge_classes' in cfg.data_loader else None
num_lesion_classes_in_annot = cfg.data_loader.num_lesion_classes_in_annot
gt_ignore_label = cfg.ignore_class_label
compute_sens_on_classes = np.array(cfg.validation.compute_sens_on_classes)
det_lesion_volume_th = cfg.validation.det_lesion_volume_th
class_names = cfg.validation.class_names
malig_benign_cyst_groups = cfg.validation.malig_benign_cyst_groups
class_channel_dict = cfg.validation.class_channel_dict


def process_gt_mask(mask, spacing):
    """remove CCs too small; Optionally merge some gt classes"""
    CC, num_CC = measure.label(mask, connectivity=2, return_num=True)
    for i in range(num_CC):
        if (CC==i+1).sum() * np.prod(spacing) < lesion_volume_th:
            mask[CC==i+1] = 0
    if merge_gt_class is not None:
        mask1 = mask*0
        for i, cs in enumerate(merge_gt_class[1:num_lesion_cls+1]):
            for c in cs:
                mask1[mask == c] = i+1
        mask1[mask == gt_ignore_label] = gt_ignore_label
        mask = mask1
    return mask


def compute_metrics(gts, preds, class_names):
    accs = []
    for c in range(gts.shape[1]):
        gt = gts[:, c] > det_lesion_volume_th
        scores = preds[:, c]  # rank by lesion prediction size
        fpr, tpr, threshold = roc_curve(gt, scores)
        auc1 = auc(fpr, tpr)
        idx = np.argmax(tpr - fpr)
        Youden_index = threshold[idx]

        pred = scores >= Youden_index
        acc = (pred == gt).sum() / len(gt)
        sens = (pred & gt).sum() / (gt.sum())
        prec = (pred & gt).sum() / (pred.sum())
        spec = ((pred == 0) & (gt == 0)).sum() / (gt == 0).sum()
        print_fun(class_names[c], t, '% 2d' % gt.sum(), t, fmt % Youden_index, t, '% 2d' % pred.sum(), fmt % acc, fmt % sens, fmt % spec,
              fmt % prec, fmt % auc1)
        accs.append([acc, sens, spec, prec, auc1])
    m = np.vstack(accs).mean(axis=0)
    np.set_printoptions(2, suppress=True)
    print_fun('mean', t, len(gts), t, t, t, t, t, *[fmt % s for s in m])
    summary = f"AUC={m[4]:.3f}, sens={m[1]:.3f}, spec={m[2]:.3f}, prec={m[3]:.3f}"
    return summary


def gen_case_report(accs, case_rpt_path):
    """Summarize pixel and lesion-wise acc of each case into a spreadsheet"""
    ovlp_th = .2
    fns = [stat[6] for stat in accs.det.stats]
    num_of_gt_lesions = [len(stat[4]) for stat in accs.det.stats]
    num_of_pred_lesions = [len(stat[1]) for stat in accs.det.stats]
    size_of_gt_lesions = [stat[4].astype(int) for stat in accs.det.stats]
    total_size_of_gt_lesions = [int(sum(stat[4])) for stat in accs.det.stats]
    binary_dices = np.hstack(accs.seg_binary.dices)
    binary_precs = np.hstack(accs.seg_binary.precisions)
    binary_recs = np.hstack(accs.seg_binary.recalls)
    lesion_ovlps = [stat[0] >= ovlp_th for stat in accs.det.stats]
    lesion_pred_hits = [ovlp.max(1) if ovlp.shape[1]>0 else np.zeros((npred,), dtype=bool)
                        for ovlp, npred in zip(lesion_ovlps, num_of_pred_lesions)]
    lesion_precs = [hit.sum()/npred for hit, npred in zip(lesion_pred_hits, num_of_pred_lesions)]
    lesion_gt_hits = [ovlp.max(0) if ovlp.shape[0]>0 else np.zeros((ngt,))
                   for ovlp, ngt in zip(lesion_ovlps, num_of_gt_lesions)]
    lesion_recs = [hit.sum()/ngt for hit, ngt in zip(lesion_gt_hits, num_of_gt_lesions)]
    lesion_pred_sizes = [stat[4] for stat in accs.det.stats]
    lesion_pred_classes = [stat[2] for stat in accs.det.stats]
    lesion_gt_classes = [stat[5] for stat in accs.det.stats]
    lesion_hit_gt_classes, lesion_pred_class_accs = [], []
    for i in range(len(fns)):
        if lesion_ovlps[i].shape[1] == 0:
            lesion_hit_gt_classes.append(np.empty((0,)))
            lesion_pred_class_accs.append(np.nan)
            continue
        hit_gt_idxs = lesion_ovlps[i].argmax(1)
        hit_gt_classes = accs.det.stats[i][5][hit_gt_idxs] * lesion_pred_hits[i]
        lesion_hit_gt_classes.append(hit_gt_classes)
        pred_accs = (lesion_pred_classes[i][lesion_pred_hits[i]] == hit_gt_classes[lesion_pred_hits[i]]).sum()\
                    /(hit_gt_classes[lesion_pred_hits[i]] != gt_ignore_label).sum()
        lesion_pred_class_accs.append(pred_accs)
    overall_sheet = pd.DataFrame({'case': fns,
        '#gt_lesion': num_of_gt_lesions, 'gt_size': total_size_of_gt_lesions,
        'gt_cls': [' '.join(['%d'%h for h in hs]) for hs in lesion_gt_classes],
        'gt_sizes': [' '.join(['%d'%h for h in hs]) for hs in size_of_gt_lesions],
        'seg_Dice': binary_dices, 'seg_prec': binary_precs, 'seg_rec': binary_recs,
        '#pred_lesion': num_of_pred_lesions, 'les_prec': lesion_precs, 'les_rec': lesion_recs,
        'les_cls_acc': lesion_pred_class_accs,
        'pred_hit': [' '.join(['%d'%h for h in hs]) for hs in lesion_pred_hits],
        'pred_size': [' '.join(['%d'%h for h in hs]) for hs in lesion_pred_sizes],
        'pred_cls': [' '.join(['%d'%h for h in hs]) for hs in lesion_pred_classes],
        'pred_gt_cls': [' '.join(['%d'%h for h in hs]) for hs in lesion_hit_gt_classes],
        'gt_hit': [' '.join(['%d'%h for h in hs]) for hs in lesion_gt_hits],
    })
    with pd.ExcelWriter(case_rpt_path) as writer:
        overall_sheet.to_excel(writer, sheet_name='overall', index=False,
                               float_format='%.4f', freeze_panes=(1, 1))

    for c in range(num_lesion_cls):
        num_of_gt_lesions1 = [(cls==c+1).sum() for cls in lesion_gt_classes]
        # num_of_pred_lesions1 = [(cls==c+1).sum() for cls in lesion_pred_classes]
        size_of_gt_lesions1 = [sz[cls==c+1].astype(int) for cls, sz in zip(lesion_gt_classes, size_of_gt_lesions)]
        total_size_of_gt_lesions1 = [int(sum(sz)) for sz in size_of_gt_lesions1]
        cls_dices = np.hstack([d[c] for d in accs.seg_classes.dices])
        cls_precs = np.hstack([d[c] for d in accs.seg_classes.precisions])
        cls_recs = np.hstack([d[c] for d in accs.seg_classes.recalls])

        lesion_ovlps1 = [ovlp[:, cls==c+1] for ovlp, cls in zip(lesion_ovlps, lesion_gt_classes)]
        lesion_gt_hits1 = [ovlp.max(0) if ovlp.shape[0] > 0 else np.zeros((ngt,))
                          for ovlp, ngt in zip(lesion_ovlps1, num_of_gt_lesions1)]
        lesion_recs1 = [hit.sum() / ngt for hit, ngt in zip(lesion_gt_hits1, num_of_gt_lesions1)]
        # lesion_hit_pred_classes1, lesion_gt_class_accs1 = [], []
        # for i in range(len(fns)):
        #     hit_pred_idxs = lesion_ovlps[i].argmax(0) if len(lesion_ovlps[i]) > 0 else np.empty((0,), dtype=int)
        #     # hit_gt_classes = accs.det.stats[i][5][hit_gt_idxs] * lesion_pred_hits[i]
        #     # lesion_hit_gt_classes.append(hit_gt_classes)
        #     # pred_accs = (lesion_pred_classes[i][lesion_pred_hits[i]] == hit_gt_classes[lesion_pred_hits[i]]).sum() \
        #     #             / (hit_gt_classes[lesion_pred_hits[i]] != gt_ignore_label).sum()
        #     pred_cls1 = lesion_pred_classes[i][hit_pred_idxs]
        #     lesion_gt_class_accs1.append(((pred_cls1==c+1) & (lesion_gt_hits[i]>0)).sum()/lesion_gt_hits1[i].sum())
        #     lesion_hit_pred_classes1.append(pred_cls1[lesion_gt_classes[i] == c+1])
        cls_sheet = pd.DataFrame({'case': fns,
                                      '#gt_lesion': num_of_gt_lesions1, 'gt_size': total_size_of_gt_lesions1,
                                      # 'gt_cls': [' '.join(['%d' % h for h in hs]) for hs in lesion_gt_classes],
                                      # 'gt_sizes': [' '.join(['%d' % h for h in hs]) for hs in size_of_gt_lesions],
                                      'seg_Dice': cls_dices, 'seg_prec': cls_precs, 'seg_rec': cls_recs,
                                      # '#pred_lesion': num_of_pred_lesions, 'les_prec': lesion_precs,
                                      'les_rec': lesion_recs1,
                                      # 'les_cls_acc': lesion_gt_class_accs1,
                                      # 'pred_hit': [' '.join(['%d' % h for h in hs]) for hs in lesion_pred_hits],
                                      # 'pred_size': [' '.join(['%d' % h for h in hs]) for hs in lesion_pred_sizes],
                                      # 'pred_cls': [' '.join(['%d' % h for h in hs]) for hs in lesion_pred_classes],
                                      # 'pred_cls': [' '.join(['%d' % h for h in hs]) for hs in lesion_hit_pred_classes1],
                                      # 'gt_hit': [' '.join(['%d' % h for h in hs]) for hs in lesion_gt_hits],
                                      })
        with pd.ExcelWriter(case_rpt_path, mode='a') as writer:
            cls_sheet.to_excel(writer, sheet_name=class_names[c], index=False,
                                   float_format='%.4f', freeze_panes=(1, 1))


def map_pred_to_ori_size(pred, gt, box):
    pred_new = gt * 0
    x1, y1, z1, x2, y2, z2 = box
    pred_new[z1:z2 + 1, y1:y2 + 1, x1:x2 + 1] = pred
    # pred_new[z1:z2 + 1] = pred
    return pred_new


class Logger:
    log_file = None

    def __init__(self, output_folder):
        timestamp = datetime.now()
        maybe_mkdir_p(output_folder)
        self.log_file = osp.join(output_folder, "eval_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                            (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                             timestamp.second))
        with open(self.log_file, 'w') as f:
            f.write("Starting... \n")

    def print_to_log_file(self, *args):
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                timestamp = datetime.now()
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                time.sleep(0.5)
                ctr += 1
        print(*args)
