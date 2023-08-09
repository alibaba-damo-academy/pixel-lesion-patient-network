# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
"""
A sample to prepare training data according to nnUNet's folder format.
Including mask preprocessing, image cropping, file saving, and dataset json generation.
"""
import os
import os.path as osp
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
from skimage import measure
from sklearn.model_selection import KFold
from tqdm import tqdm
import SimpleITK as sitk
from multiprocessing.dummy import Pool as ThreadPool


task_target_fd = '/data/vdd/user/codes/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task286_LiverLesionSeg'
all_data_target_fd = '/data/vdd/user/codes/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/all_data'

image_fd = '/data/vdc/user/data/liver/DEEDS_output/'  # if use multi-phase data, should be registered images
lesion_annotation_fd = '/data/vdc/user/data/liver/2019/masks'
lesion_label_filename = 'mask.nii.gz'
lesion_labels = {1: 'HCC', 2: 'cholangioma', 3: 'metastasis', 4: 'hepatoblastoma',
                 5: 'hemangioma', 6: 'FNH', 7: 'cyst', 8: 'other', 9: 'unknown'}
lesion_label_num = len(lesion_labels)-1
ignore_cls_idx = 100  # set label 100 in the new mask as ignored label

organ_pseudo_label_fd = '/data/vdd/user/codes/nnUNet/predictions/Task250_HCOHCMnav'
organ_labels = {"01": "pancreas", "02": "spleen", "03": "liver", "04": "portal vein and splenic vein",
        "05": "hepatic vein", "06": "hemangioma", "07": "gallBladder", "08": "stomach",
        "09": "cyst", "10": "others", "11": "hcc", "12": "cholangio", "13": "meta"}
select_organ_labels = {3: 1, 6: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1,
                       7: 2, 5: 3, 4: 3, 2: 4, 8: 5, 1: 6}  # {A:B} map organ label A to label B in the new mask
organ_label_filename = 'pred_rmvAir.nii.gz'
lesion_labels_in_organ = [6]+list(range(9, 14))

crop_organ_labels = [3, 5, 6, 7] + list(range(9, 14))  # crop images based on these organ labels and save for training
box_dilate = 5  # e.g. crop liver with a padding of 5 pixels in each direction

exclude_pids = []  # these data will not be used

modalities = ('non-contrast', 'arterial', 'venous')
task_modalities = (0,)  # 0 for NC and (0, 1, 2) for NAV phases


def save_itk(im, path, template_sitk):
    im_sitk = sitk.GetImageFromArray(im)
    im_sitk.SetSpacing(template_sitk.GetSpacing())
    im_sitk.SetOrigin(template_sitk.GetOrigin())
    im_sitk.SetDirection(template_sitk.GetDirection())
    sitk.WriteImage(im_sitk, path)


def get_box(mask, box_dilate, sel_labels):
    mask = sum([mask == i for i in sel_labels]) > 0
    idxs = np.where(mask)
    x1 = np.maximum(0, idxs[2].min()-box_dilate)
    y1 = np.maximum(0, idxs[1].min()-box_dilate)
    z1 = np.maximum(0, idxs[0].min()-box_dilate)
    x2 = np.minimum(idxs[2].max()+box_dilate, mask.shape[2]-1)
    y2 = np.minimum(idxs[1].max()+box_dilate, mask.shape[1]-1)
    z2 = np.minimum(idxs[0].max()+box_dilate, mask.shape[0]-1)
    return x1, y1, z1, x2, y2, z2


def convert_mask(src_path, organ_mask):
    """Add organ labels after lesion labels"""
    mask_sitk = sitk.ReadImage(src_path)
    mask = sitk.GetArrayFromImage(mask_sitk)
    mask[mask == 9] = ignore_cls_idx  # unknown class as ignore label
    mask_new = organ_mask * 0
    for l in select_organ_labels:
        mask_new[organ_mask == l] = lesion_label_num + select_organ_labels[l]
    mask_new[mask > 0] = mask[mask > 0]  # overlay lesion label on top of organ labels
    mask = mask_new
    return mask


def crop_save_im_mask(params):
    pat, = params
    print(pat)
    organ_path = osp.join(organ_pseudo_label_fd, pat, organ_label_filename)
    if not osp.exists(organ_path):
        print(organ_path, 'not exist')
        return pat, None
    organ_sitk = sitk.ReadImage(organ_path)
    organ_mask = sitk.GetArrayFromImage(organ_sitk)
    x1, y1, z1, x2, y2, z2 = get_box(organ_mask, box_dilate, crop_organ_labels)

    src_path = osp.join(lesion_annotation_fd, pat, lesion_label_filename)
    tgt_path = osp.join(all_data_target_fd, f"{pat}.nii.gz")
    mask = convert_mask(src_path, organ_mask)
    mask = mask[z1:z2 + 1, y1:y2 + 1, x1:x2 + 1].astype('uint8')
    save_itk(mask, tgt_path, organ_sitk)

    for m, mod in enumerate(modalities):
        src_path = osp.join(image_fd, pat, f"{mod}.nii.gz")
        if not osp.exists(src_path):
            print(src_path, 'does not exist')
            continue
        tgt_path = osp.join(all_data_target_fd, f"{pat}_{m:04d}.nii.gz")
        # if not os.path.exists(tgt_path):
        im_sitk = sitk.ReadImage(src_path)
        vol = sitk.GetArrayFromImage(im_sitk)
        vol = vol[z1:z2+1, y1:y2+1, x1:x2+1].astype('int16')
        save_itk(vol, tgt_path, im_sitk)
    return pat, (x1, y1, z1, x2, y2, z2)


def sort_raw_data():
    os.makedirs(all_data_target_fd, exist_ok=True)
    os.makedirs(task_target_fd, exist_ok=True)
    os.makedirs(osp.join(task_target_fd, 'imagesTr'), exist_ok=True)
    os.makedirs(osp.join(task_target_fd, 'imagesTs'), exist_ok=True)
    os.makedirs(osp.join(task_target_fd, 'labelsTr'), exist_ok=True)
    os.makedirs(osp.join(task_target_fd, 'labelsTs'), exist_ok=True)
    pats = [fn[:6] for fn in os.listdir(image_fd)]
    pats = np.setdiff1d(pats, exclude_pids)
    pats = sorted(pats)

    threads = 16
    pool = ThreadPool(threads)
    params_list = [[pat, ] for pat in pats]
    results = pool.map(crop_save_im_mask, params_list)
    pool.close()
    pool.join()

    liver_boxes = {pat: box for pat, box in results if box is not None}
    torch.save(liver_boxes, 'liver_boxes_pad%d.pth' % box_dilate)


def generate_dataset_json():
    from nnunet.dataset_conversion.utils import generate_dataset_json

    labels = {f"{lb:02d}": name for lb, name in lesion_labels.items() if name != 'unknown'}
    for l in select_organ_labels:
        if l in lesion_labels_in_organ:
            continue
        key = f"{lesion_label_num + select_organ_labels[l]:02d}"
        val = organ_labels[f"{l:02d}"]
        labels[key] = val
    generate_dataset_json(output_file=task_target_fd + '/dataset.json', imagesTr_dir=task_target_fd + '/imagesTr',
                          imagesTs_dir=task_target_fd + '/imagesTs',
                          modalities=('CT', )*len(task_modalities),
                          labels=labels, dataset_name='liver_dataaset')


if __name__ == '__main__':
    sort_raw_data()
    generate_dataset_json()
