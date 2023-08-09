# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import SimpleITK as sitk
import torch
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.network_architecture.attn_unet.my_generic_UNet import MTUNet
from utils.my_utils import instance_to_semantic
from .anchor_matcher import AnchorMatcher
from .losses import InstanceDiscLoss

from torch import nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ..mask2former.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from .my_transformer_decoder import MyTransformerDecoder
from configs.config_utils import cfg as cfg_all


class DetHead(nn.Module):
    """Pixel-Lesion-pAtient Network"""
    def __init__(self, num_classes, use_anchor_query, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.use_anchor_query = use_anchor_query  # if True, use input query embeddings instead of random ones, do not use matcher

        self.cfg = cfg
        predictor = MultiScaleMaskedTransformerDecoder if not self.use_anchor_query else MyTransformerDecoder
        self.predictor = predictor(num_classes=self.num_classes, **self.cfg.transformer_predictor)

        weight_dict = {"clsf_ce": self.cfg.loss.cls_weight,
                       "mask_ce": self.cfg.loss.mask_weight, "dice": self.cfg.loss.dice_weight,
                       'dice_ce': 1, 'clsf_focal': self.cfg.loss.cls_weight}
        if self.cfg.transformer_predictor.deep_supervision:  # deep supervision for transformer layers.
            dec_layers = self.cfg.transformer_predictor.dec_layers
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        if self.use_anchor_query:  # PLAN, use random+anchor query
            self.anchor_matcher = AnchorMatcher(self.num_classes, cfg)
            from .criterion import SetCriterion
            from .matcher import HungarianMatcher
        else:  # original Mask2Former, use random query and bipartite matching
            from ..mask2former.criterion import SetCriterion
            from ..mask2former.matcher import HungarianMatcher
        matcher = HungarianMatcher(
            cost_class=self.cfg.loss.cls_weight,
            cost_mask=self.cfg.loss.mask_weight,
            cost_dice=self.cfg.loss.dice_weight,
            num_points=self.cfg.loss.train_num_points,
        )
        criterion = SetCriterion(
            self.cfg.num_all_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=cfg.loss.no_object_weight,
            losses=["labels", "masks", ],
            num_points=cfg.loss.train_num_points, oversample_ratio=cfg.loss.oversample_ratio,
            importance_sample_ratio=cfg.loss.importance_sample_ratio,
            cfg=cfg,
        )
        self.criterion = criterion
        if 'instance_discrim' in cfg.loss:
            self.criterion_id = InstanceDiscLoss(**cfg.loss.instance_discrim,
                                labels=[cfg_all.model.liver_label]+list(range(1, cfg_all.num_lesion_classes+1)))
                                # labels=list(range(1, cfg_all.num_lesion_classes+1)))

    def forward(self, backbone, seg_pred=None):
        im_features = [backbone.encoder_fts[f] if f > 0 else backbone.decoder_fts[f]
                       for f in self.cfg.transformer_in_feature]
        pixel_emb = backbone.decoder_fts[-1]
        if self.use_anchor_query:
            anchor_queries, anchor_labels, CCs = self.anchor_matcher.gen_anchor_query(seg_pred, [pixel_emb])
            outputs = self.predictor(im_features, pixel_emb, anchor_queries)
            # outputs['CCs'] = CCs
            outputs['anchor_labels'] = anchor_labels
            outputs['pixel_emb'] = pixel_emb
        else:
            outputs = self.predictor(im_features, pixel_emb)
        return outputs

    def loss(self, outputs, targets, labels):
        det_targets = self.prepare_targets_mask2former(targets, labels)

        losses = self.criterion(outputs, det_targets)
        for k in list(losses.keys()):
            losses[k] *= self.criterion.weight_dict[k]
        if hasattr(self, 'criterion_id'):
            losses['id'] = self.criterion_id(outputs['pixel_emb'], targets[0].int(), labels)
        for k in list(losses.keys()):
            losses[k] *= self.cfg.loss.weight
        return losses

    def prepare_targets_mask2former(self, targets, index_to_labels):
        """Generate one-hot multi-channel masks for each multi-level target.
        Optionally merge lesion instances to semantic masks.
         Remove ignore mask in target.
        """
        num_ds_levels = len(targets)
        batch_size = len(index_to_labels)
        new_targets = [[] for _ in range(num_ds_levels)]
        for b in range(batch_size):  #TODO: need double check
            label_to_index = {label: index_to_labels[b][index_to_labels[b][:, 1] == label, 0]
                              for label in list(range(self.num_classes)) + [self.cfg.ignore_class_label]}

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
            for c in range(self.cfg.num_lesion_classes+1, self.num_classes):  # exclude ignore_label
                if len(label_to_index[c]) > 0:
                    indices_label_tuples.extend([(label_to_index[c], c)])

            # build targets and labels for each deep supervision level
            for lvl in range(num_ds_levels):
                mask1 = targets[lvl][b, 0].int()
                targets1, labels1 = [], []
                for idx, (indices, label) in enumerate(indices_label_tuples):
                    one_hot_mask1 = sum([mask1 == index for index in indices]) > 0

                    if lvl > 0 and one_hot_mask1.sum() == 0 and new_targets[0][b]['masks'][idx].sum() > 0:
                        # CC too small, disappeared after downsampling
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
        processed_results = []
        for b in range(len(outputs['pred_logits'])):
            mask_cls_result = outputs["pred_logits"][b].detach()
            mask_pred_result = outputs["pred_masks"][b].detach()  # only use the finest level
            # upsample masks
            # if self.cfg.prediction_stride > 1:
            #     mask_pred_results = F.interpolate(
            #         mask_pred_result[None],
            #         # size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            #         scale_factor=self.pred_scale_factor,
            #         mode="trilinear",
            #         align_corners=False,
            #     )[0]

            r = self.semantic_inference(mask_cls_result, mask_pred_result)
            processed_results.append(r)

        processed_results = torch.stack(processed_results)
        return processed_results

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]  # last channel is 'no_object'
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qdhw->cdhw", mask_cls, mask_pred)
        semseg /= semseg.sum(0)  # normalize to have sum 1 for each pixel. We have tried no normalize or normalize if
        # sum > threshold, but no significant improvement
        return semseg
