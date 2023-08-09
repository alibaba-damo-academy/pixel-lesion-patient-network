#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Copyright (c) Medical AI Lab, Alibaba DAMO Academy

import pprint
from statistics import mode

import numpy as np
import torch
import warnings

from configs.config_utils import cfg
from skimage import measure
from sklearn.metrics import classification_report

from nnunet.evaluation.evaluation_diagnosis import seg_acc_single
from nnunet.evaluation.evaluation_det_utils import FROC_mask, volume_points, ovlp_th

from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
# from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.mask2former.my_generic_UNet_emb import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import unpack_dataset

from nnunet.utilities.tensor_utilities import sum_tensor

import os

from utils.my_utils import instance_to_semantic


class panoSegTrainer(nnUNetTrainerV2):
    """
    joint nnUNet + panoptic/semantic segmentation in maskFormer
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = cfg.optimizer.max_num_epochs
        self.initial_lr = cfg.optimizer.initial_lr
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True
        self.fix_unet_epoch = cfg.optimizer.fix_unet_epoch
        self.online_eval_prec = []  # if a case contains lesion (except cyst) and it is detected with the correct class, set as 1
        self.online_eval_rec = []  # if a case contains NO lesion (except cyst) and no lesion (except cyst) is detected, set as 1

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        self.print_to_log_file(pprint.pformat(cfg))
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.plans['num_classes'] = cfg.num_all_classes - 1
            self.process_plans(self.plans)

            self.setup_DA_params()
            if 'flip_aug' in cfg.data_loader:
                if len(cfg.data_loader.flip_aug) == 0:
                    self.data_aug_params['do_mirror'] = False
                    self.data_aug_params['mirror_axes'] = ()
                else:
                    self.print_to_log_file('mirror_axes '+str(cfg.data_loader.flip_aug))
                    self.data_aug_params['do_mirror'] = True
                    self.data_aug_params['mirror_axes'] = cfg.data_loader.flip_aug

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            cfg.model.nnunet_ds_loss_weights = self.ds_loss_weights
            # now wrap the loss
            # self.seg_loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            # class_weight = None
            # self.class_loss = ClassLossV3(lamda_class=1.0, epoch_thresh=None, weight=class_weight)
            # self.class_loss = ClassLossMultiLabel(lamda_class=1.0, epoch_thresh=None, weight=class_weight)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            # assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()
        loader_inst = eval(cfg.data_loader.type)
        dl_tr = loader_inst(True, self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                             cfg.data_loader,
                             False, oversample_foreground_percent=self.oversample_foreground_percent,
                             pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        dl_val = loader_inst(False, self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                              cfg.data_loader,
                              False, oversample_foreground_percent=self.oversample_foreground_percent,
                              pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr, dl_val

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, cfg.model.nnunet_deep_supervision, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)

        self.network = eval(cfg.model.type)(self.network, cfg.model, self.loss)

        if self.config.seg_pretrain_dir is not None:
            self.network.load_from(  # YK
                os.path.join(self.config.seg_pretrain_dir, 'fold_' + str(self.fold) + '/model_final_checkpoint.model'))
        else:
            print('not loaded from self.configs.seg_pretrain_dir')
        if self.fix_unet_epoch > 0:
            self.network.fix_unet(True)
        if torch.cuda.is_available():
            self.network.cuda(cfg.gpu)

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"

        backbone_names = [name for name, _ in self.network.named_parameters()
                          if any([n in name for n in cfg.optimizer.backbone_name_include]) and
                          not any([n in name for n in cfg.optimizer.backbone_name_exclude])]
                          # if 'unet.' in name and 'emb' not in name and 'fpn' not in name]
        param_group1 = [param for name, param in self.network.named_parameters() if name in backbone_names]
        param_group2 = [param for name, param in self.network.named_parameters() if name not in backbone_names]
        backbone_multiplier = cfg.optimizer.backbone_multiplier
        if backbone_multiplier < 0:
            backbone_multiplier = abs(backbone_multiplier)  # linearly from 0 to 1
        param_groups = [
                   {'params': param_group1, 'lr': self.initial_lr * backbone_multiplier},
                   {'params': param_group2}
               ]
        # according to MaskFormer. Not showing superiority
        # param_groups = [{'params': [], 'lr': self.initial_lr * cfg.optimizer.backbone_multiplier},
        #                 {'params': []},
        #                 {'params': [], 'weight_decay': 0}]  # per maskformer
        # for module_name, module in self.network.named_modules():
        #     for module_param_name, param in module.named_parameters(recurse=False):
        #         if 'unet.' in module_name and 'emb' not in module_name and 'fpn' not in module_name:
        #             param_groups[0]['params'].append(param)
        #         elif isinstance(module, (nn.Embedding, nn.InstanceNorm3d, nn.LayerNorm)) or module_param_name == 'bias':
        #             param_groups[2]['params'].append(param)
        #         else:
        #             param_groups[1]['params'].append(param)
        if cfg.optimizer.method == 'SGD':
            self.optimizer = torch.optim.SGD(param_groups, self.initial_lr, weight_decay=self.weight_decay,
                                             momentum=0.99, nesterov=True)
        elif cfg.optimizer.method == 'RAdam':  # Yingda's
            self.optimizer = torch.optim.RAdam(param_groups, self.initial_lr, weight_decay=1e-4)
        elif cfg.optimizer.method == 'AdamW':  # MaskFormer's
            self.optimizer = torch.optim.AdamW(param_groups, self.initial_lr, weight_decay=cfg.optimizer.weight_decay)

        self.lr_scheduler = None

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        properties = data_dict['properties']
        label = data_dict.get('label', None)

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data, gpu_id=cfg.gpu)
            target = to_cuda(target, gpu_id=cfg.gpu)

        self.optimizer.zero_grad()

        if self.epoch == self.fix_unet_epoch:
            self.network.fix_unet(False)

        cfg.cur_epoch = self.epoch
        if self.fp16:
            with autocast():
                outputs = self.network.get_all_outputs(data)
                del data
                l, loss_dict = self.network.loss(outputs, target, label, properties)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            outputs = self.network.get_all_outputs(data)
            del data
            l, loss_dict = self.network.loss(outputs, target, label)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            pred_mask = self.network.inference(outputs)
            if cfg.data_loader.preprocess_mask:
                semantic_target = instance_to_semantic(target[0], label,
                                                       {cfg.model.ignore_class_label: 0})
                semantic_target = to_cuda(semantic_target, gpu_id=cfg.gpu)
                self.run_online_evaluation(pred_mask, semantic_target)
            else:
                self.run_online_evaluation(pred_mask, target[0])

        del target
        loss_dict = {key: val.detach().cpu().numpy() for key, val in loss_dict.items()}
        if 'copy_paste_lesion' in data_dict['properties'][0]:
            loss_dict['cpaug'] = sum([d['copy_paste_lesion'] for d in data_dict['properties']]) * 250
        return l.detach().cpu().numpy(), loss_dict

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        new_lr = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        backbone_multiplier = cfg.optimizer.backbone_multiplier
        if backbone_multiplier < 0:  # linearly from 0 to 1
            backbone_multiplier = abs(backbone_multiplier) + ep/self.max_num_epochs * (1-abs(backbone_multiplier))
        self.optimizer.param_groups[0]['lr'] = new_lr * backbone_multiplier
        self.optimizer.param_groups[1]['lr'] = new_lr
        # self.optimizer.param_groups[2]['lr'] = new_lr
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[1]['lr'], decimals=6))

    def my_online_evaluation(self, output, target):
        """Compute patient-averaged Dice, lesion-wise precision, recall, and clsf accuracy, and patient-wise AUC"""
        # assert output.shape[1] == self.num_classes  # not support classifier output now
        num_lesion_cls = cfg.num_lesion_classes
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        target[(target > num_lesion_cls) & (target != cfg.ignore_class_label)] = 0
        output[output > num_lesion_cls] = 0
        volume_th_px = 10
        warnings.filterwarnings("ignore")
        for pred, gt in zip(output, target):
            dice, precision, recall = seg_acc_single(pred * (gt != cfg.ignore_class_label), gt, num_lesion_cls)
            pred_cc, num_pred = measure.label(pred > 0, connectivity=2, return_num=True)
            gt_cc, num_gt = measure.label(gt, connectivity=2, return_num=True)
            pred_sizes = np.array([(pred_cc == i + 1).sum() for i in range(num_pred)])
            pred_classes = np.array([mode(pred[pred_cc == i + 1]) for i in range(num_pred)], dtype=int)
            gt_sizes = np.array([(gt_cc == i + 1).sum() for i in range(num_gt)])
            gt_classes = np.array([gt[gt_cc == i + 1][0] for i in range(num_gt)], dtype=int)
            pred_keep = np.where(pred_sizes >= volume_th_px)[0]
            gt_keep = np.where(gt_sizes >= volume_th_px)[0]
            ovlps = []

            for i in pred_keep:
                pred_i = np.where(pred_cc == i + 1)
                pred_gt_ovlp = np.array([(gt_cc[pred_i] == c + 1).sum() for c in range(num_gt)])
                ovlp = pred_gt_ovlp * 2 / (pred_sizes[i] + gt_sizes)
                ovlps.append(ovlp)
            ovlps = (np.vstack(ovlps) if len(pred_keep) > 0 else np.empty((0, num_gt)))[:, gt_keep]
            self.my_online_eval_data.append((
                dice, ovlps, pred_classes[pred_keep], gt_sizes[gt_keep], gt_classes[gt_keep]))
        warnings.filterwarnings("default")

    def run_online_evaluation(self, output, target):
        with torch.no_grad():
            num_classes = self.num_classes
            if output.shape[1] > num_classes:  # has classifier output
                output = output[:, :num_classes]
            # output_softmax = softmax_helper(output)
            output_seg = output.argmax(1)
            target = target[:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))

            pred_lesion = (output_seg > 0) & (output_seg <= cfg.num_lesion_classes)
            gt_lesion = (target > 0) & (target <= cfg.num_lesion_classes)
            tp = (pred_lesion & gt_lesion).sum()
            self.online_eval_prec.append((tp / pred_lesion.sum()).cpu().numpy())
            self.online_eval_rec.append((tp / gt_lesion.sum()).cpu().numpy())

            if cfg.my_online_val and cfg.cur_epoch % 2 == 0:
                self.my_online_evaluation(output_seg, target)

    def finish_online_evaluation(self):
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]]
                               # if not np.isnan(i)]
        val_metric = np.nanmean(global_dc_per_class[:cfg.num_lesion_classes])
        self.all_val_eval_metrics.append(val_metric)
        val_MA = val_metric if self.val_eval_criterion_MA is None else self.val_eval_criterion_MA

        self.print_to_log_file("Average global foreground Dice:", [np.round(i, 4) for i in global_dc_per_class],
                               'val_metric', round(val_metric, 4), 'val_metric_MA', round(val_MA, 4))
        # self.print_to_log_file("(interpret this as an estimate for the Dice of the different classes. This is not "
        #                        "exact.)")

        prec = np.nanmean(self.online_eval_prec)
        rec = np.nanmean(self.online_eval_rec)
        # self.print_to_log_file("Lesion pixel precision: %.4f recall: %.4f" % (prec, rec))
        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []
        self.online_eval_prec = []
        self.online_eval_rec = []

        if cfg.my_online_val and cfg.cur_epoch % 2 == 0:
            dices, ovlps, pred_classes, gt_sizes, gt_classes = list(zip(*self.my_online_eval_data))
            if len(np.hstack(pred_classes)) == 0:
                return
            warnings.filterwarnings("ignore")
            all_scores = [p*0+1 for p in pred_classes]
            dice = np.nanmean(np.nanmean(np.vstack(dices), axis=0))
            ignore_masks = [cls == cfg.ignore_class_label for cls in gt_classes]
            sens, fp_per_vol, precisions, sens_per_class = \
                FROC_mask(ovlps, all_scores, ovlp_th, ignore_masks, list(range(1, cfg.num_lesion_classes+1)), gt_classes)
            all_gt_volumes = [np.searchsorted(volume_points/2.5, sz.astype(int)) for sz in gt_sizes]
            size_classes = list(range(len(volume_points)+1))
            _, _, _, sens_per_size = FROC_mask(ovlps, all_scores, ovlp_th, ignore_masks, size_classes, all_gt_volumes)

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
            res = classification_report(labels_gt, labels_pred, output_dict=True)
            lesion_clsf_acc, lesion_sens, lesion_prec = res['accuracy'], np.nanmean(sens_per_class[-1]), precisions[-1]
            # val_metric = sum([lesion_clsf_acc, lesion_sens, lesion_prec, dice])/4
            val_metric = sum([lesion_clsf_acc, lesion_sens, lesion_prec])/3

            fmt = lambda x: [np.round(i, 4) for i in x]
            # self.print_to_log_file(f"Avg lesion case Dice {dice:.4f}, cls sens {lesion_sens:.4f} prec {lesion_prec:.4f}",
            self.print_to_log_file(f"lesion-wise sens {lesion_sens:.4f} prec {lesion_prec:.4f}",
                                   f"clsf acc {lesion_clsf_acc:.4f}, val_metric {round(val_metric, 4)}")
            self.print_to_log_file(f"sens of each class {fmt(sens_per_class[-1])} and size {fmt(sens_per_size[-1])}")
            self.my_online_eval_data = []
            warnings.filterwarnings("default")
