# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import torch
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.network_architecture.attn_unet.my_generic_UNet import MTUNet

from torch import nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from .criterion import SetCriterion
from .matcher import HungarianMatcher


class Mask2Former(MTUNet):
    def __init__(self, nnunet, cfg, nnunet_loss):
        super(MTUNet, self).__init__()
        self.unet = nnunet
        self.conv_op = self.unet.conv_op  # for validation
        self.unet_fixed = False
        self.predictor = MultiScaleMaskedTransformerDecoder(num_classes=cfg.num_all_classes, **cfg.transformer_predictor)
        self.cfg = cfg
        self.num_classes = self.cfg.num_all_classes
        self.do_ds = self.cfg.nnunet_deep_supervision
        self.pred_scale_factor = {1: 1, 2: [1, 2, 2]}[cfg.prediction_stride]

        if self.cfg.norm_pixel_emb == 'IN':
            self.norm_pixel_emb = nn.InstanceNorm3d(self.cfg.mask_emb_dim, affine=True)

        deep_supervision = cfg.transformer_predictor.deep_supervision
        no_object_weight = cfg.loss.no_object_weight
        dice_weight = cfg.loss.dice_weight
        mask_weight = cfg.loss.mask_weight
        cls_weight = cfg.loss.cls_weight

        # building criterion
        matcher = HungarianMatcher(
            cost_class=cls_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.loss.train_num_points,
        )

        weight_dict = {"clsf_ce": cls_weight, "mask_ce": mask_weight, "dice": dice_weight,
                       'dice_ce': 1, 'clsf_focal': cls_weight}
        if deep_supervision:  # deep supervision for transformer layers. TODO: is this necessary?
            dec_layers = cfg.transformer_predictor.dec_layers
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks",]

        criterion = SetCriterion(
            self.cfg.num_all_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.loss.train_num_points, oversample_ratio=cfg.loss.oversample_ratio,
            importance_sample_ratio=cfg.loss.importance_sample_ratio,
            cfg=cfg,
        )
        self.criterion = criterion

        self.criterion_nnunet = nnunet_loss

    def forward(self, x, online_val=False):
        """return logits of each class, to be compatible to nnUNet's output format"""
        outputs = self.get_all_outputs(x)
        pred_mask = self.inference(outputs)
        return pred_mask

    def get_all_outputs(self, x):
        pixel_embs = self.unet(x)
        im_features = [self.unet.encoder_fts[f] if f > 0 else self.unet.decoder_fts[f]
                       for f in self.cfg.transformer_in_feature]
        if self.cfg.norm_pixel_emb == 'L2':
            pixel_embs = [F.normalize(ft, p=2, dim=1) for ft in pixel_embs]  # from Jieneng's paper
        elif self.cfg.norm_pixel_emb == 'IN':
            pixel_embs = [self.norm_pixel_emb(ft) for ft in pixel_embs]
        outputs = self.predictor(im_features, pixel_embs[0])
        # del self.unet.encoder_fts
        return outputs

    def loss(self, outputs, targets, labels, properties):
        # mask classification target
        if not self.cfg.nnunet_deep_supervision:
            targets = [targets[int(np.log2(self.cfg.prediction_stride))]]
        targets = self.prepare_targets(targets, labels)
        # not completely done in data loader because there are a series of transforms
        # in data loader that may hurt multi-channel targets

        # bipartite matching-based loss
        losses = self.criterion(outputs, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        l = sum(losses.values())
        for k in list(losses.keys()):  # remove loss items to reduce printing
            if k[-2] == '_' and k[-1].isdigit():
                losses.pop(k)
        return l, losses

    def prepare_targets(self, targets, index_to_labels):
        """Generate one-hot multi-channel masks for each multi-level target.
        Optionally merge lesion instances to semantic masks.
        Not used now: Merge liver, liver lesion, and ignore masks as new liver mask.
        Meanwhile keep the original liver mask as "healthy_liver" class, which is liver mask minus all lesion masks.
         Remove ignore mask in target.
        """
        num_ds_levels = len(targets)
        batch_size = len(index_to_labels)
        new_targets = [[] for _ in range(num_ds_levels)]
        for b in range(batch_size):
            label_to_index = {label: index_to_labels[b][index_to_labels[b][:, 1] == label, 0]
                              for label in list(range(self.cfg.num_all_classes)) + [self.cfg.ignore_class_label]}

            # Collect indices of bg and lesions, optionally merge lesion instances to semantic masks.
            indices_label_tuples = []
            for c in range(self.cfg.num_lesion_classes+1):
                if len(label_to_index[c]) == 0:
                    continue
                if self.cfg.lesion_instance_seg:
                    indices_class_tuples1 = [([i], c) for i in label_to_index[c]]
                else:
                    indices_class_tuples1 = [(label_to_index[c], c)]
                indices_label_tuples.extend(indices_class_tuples1)

            # Collect indices of organs
            for c in range(self.cfg.num_lesion_classes+1, self.cfg.num_all_classes):  # exclude ignore_label
                if len(label_to_index[c]) > 0:
                    indices_label_tuples.extend([(label_to_index[c], c)])

            # build targets and labels for each deep supervision level
            for lvl in range(num_ds_levels):
                mask1 = targets[lvl][b, 0].int()
                targets1, labels1 = [], []
                for idx, (indices, label) in enumerate(indices_label_tuples):
                    one_hot_mask1 = sum([mask1 == index for index in indices]) > 0

                    if lvl > 0 and one_hot_mask1.sum() == 0 and new_targets[0][b]['masks'][idx].sum() > 0:
                        # CC to small, disappeared after downsampling
                        coord = torch.nonzero(new_targets[0][b]['masks'][idx]).float().mean(0).cpu()
                        coord = coord / (torch.tensor(new_targets[0][b]['masks'][idx].shape) / torch.tensor(mask1.shape))
                        c = coord.int()
                        one_hot_mask1[c[0], c[1], c[2]] = True

                    targets1.append(one_hot_mask1)
                    labels1.append(label)
                new_targets[lvl].append({
                    'labels': torch.Tensor(labels1).to(mask1.device).long(),
                    'masks': torch.stack(targets1)
                })

        return new_targets

    def inference(self, outputs):
        mask_cls_results = outputs["pred_logits"].detach()
        mask_pred_results = outputs["pred_masks"].detach()  # only use the finest level
        # upsample masks
        if self.cfg.prediction_stride > 1:
            mask_pred_results = F.interpolate(
                mask_pred_results,
                # size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                scale_factor=self.pred_scale_factor,
                mode="trilinear",
                align_corners=False,
            )

        processed_results = []
        for mask_cls_result, mask_pred_result in zip(
            mask_cls_results, mask_pred_results
        ):
            # semantic segmentation inference
            r = self.semantic_inference(mask_cls_result, mask_pred_result)
            processed_results.append(r)

        processed_results = torch.stack(processed_results)
        return processed_results

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qdhw->cdhw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask = cur_mask_ids == k
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()

                if mask_area > 0 and original_area > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def _internal_predict_3D_3Dconv_tiled(self, *args, **kwargs):
        return super(MTUNet, self)._internal_predict_3D_3Dconv_tiled(*args, **kwargs)

    def _internal_maybe_mirror_and_pred_3D(self, *args, **kwargs):
        return super(MTUNet, self)._internal_maybe_mirror_and_pred_3D(*args, **kwargs)
