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

from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional
from typing import Union, Tuple, List
from batchgenerators.augmentations.utils import pad_nd_image
from nnunet.utilities.random_stuff import no_op
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch


class MTUNet(SegmentationNetwork):
    def __init__(self, nnunet, num_classes=3):
        super(MTUNet, self).__init__()
        self.unet = nnunet
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv3d(320, 32, kernel_size=1), nn.InstanceNorm3d(32, affine=True), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv3d(320, 32, kernel_size=1), nn.InstanceNorm3d(32, affine=True), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv3d(256, 32, kernel_size=1), nn.InstanceNorm3d(32, affine=True), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv3d(128, 32, kernel_size=1), nn.InstanceNorm3d(32, affine=True), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv3d(64, 32, kernel_size=1), nn.InstanceNorm3d(32, affine=True), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv3d(32, 32, kernel_size=1), nn.InstanceNorm3d(32, affine=True), nn.ReLU(inplace=True)),
        ])
        self.num_classes = num_classes
        self.classifier = nn.Linear(192, self.num_classes)
        self.conv_op = self.unet.conv_op
        self.unet_fixed = False

    def forward(self, x, online_val=False):
        seg_outputs = self.unet(x)
        fts = self.unet.get_fts()

        projected_fts = [proj(ft) for ft, proj in zip(fts, self.projections)]
        projected_fts = [nn.AdaptiveMaxPool3d((1, 1, 1))(ft) for ft in projected_fts]
        projected_ft = torch.cat(projected_fts, dim=1)[:, :, 0, 0, 0]

        if self.training or online_val:
            return seg_outputs, self.classifier(projected_ft)
        else:
            class_prob = self.classifier(projected_ft)
            b, c, w, h, d = seg_outputs.shape
            classprob2concat = class_prob.view(b, self.num_classes, 1, 1, 1).expand(b, self.num_classes, w, h, d)
            return torch.cat([seg_outputs, classprob2concat], dim=1)

    def load_from(self, ckpt_path):
        pretrained_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        target = self if any([k.startswith('unet.') for k in pretrained_dict]) else self.unet
        try:
            target.load_state_dict(pretrained_dict)
        except Exception as e:
            print(e)
            print('Try only load weights of the same shape in the UNet: ')
            model_dict = target.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
            model_dict.update(pretrained_dict)
            target.load_state_dict(model_dict)
            print(pretrained_dict.keys())
            print('done')

    def fix_unet(self, fix=True):
        if fix:
            if not self.unet_fixed:
                for name, params in self.unet.named_parameters():
                    if 'emb' not in name:
                        params.requires_grad = False
                    else:
                        params.requires_grad = True
                self.unet_fixed = True
        else:
            if self.unet_fixed:
                for params in self.unet.parameters():
                    params.requires_grad = True
                self.unet_fixed = False

    def _internal_predict_3D_3Dconv_tiled(self, x: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                          patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                          pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        # better safe than sorry
        assert len(x.shape) == 4, "x must be (c, x, y, z)"
        assert self.get_device() != "cpu"
        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape  # still c, x, y, z

        # compute the steps for sliding window
        steps = self.unet._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self.unet._gaussian_3d is None or not all(
                    [i == j for i, j in zip(patch_size, self.unet._patch_size_for_gaussian_3d)]):
                if verbose: print('computing Gaussian')
                gaussian_importance_map = self.unet._get_gaussian(patch_size, sigma_scale=1. / 8)

                self.unet._gaussian_3d = gaussian_importance_map
                self.unet._patch_size_for_gaussian_3d = patch_size
            else:
                if verbose: print("using precomputed Gaussian")
                gaussian_importance_map = self.unet._gaussian_3d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map).cuda(self.unet.get_device(),
                                                                                     non_blocking=True)

        else:
            gaussian_importance_map = None

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                    gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(data.shape[1:], device=self.unet.get_device())

            if verbose: print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.unet.num_classes + self.num_classes] + list(data.shape[1:]),
                                             dtype=torch.half,
                                             device=self.unet.get_device())

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.unet.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros(
                [self.unet.num_classes + self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                device=self.unet.get_device())
        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self.unet._gaussian_3d
            else:
                add_for_nb_of_preds = np.ones(data.shape[1:], dtype=np.float32)
            aggregated_results = np.zeros([self.unet.num_classes + self.num_classes] + list(data.shape[1:]),
                                          dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.unet.num_classes + self.num_classes] + list(data.shape[1:]),
                                                    dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]

                    predicted_patch = self._internal_maybe_mirror_and_pred_3D(
                        data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], mirror_axes, do_mirroring,
                        gaussian_importance_map)[0]

                    if all_in_gpu:
                        predicted_patch = predicted_patch.half()
                    else:
                        predicted_patch = predicted_patch.cpu().numpy()

                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                    aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
             range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        class_probabilities = aggregated_results / aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = class_probabilities[:self.unet.num_classes].argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = class_probabilities.detach().cpu().numpy()
            else:
                class_probabilities_here = class_probabilities
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose: print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            class_probabilities = class_probabilities.detach().cpu().numpy()

        if verbose: print("prediction done")
        return predicted_segmentation, class_probabilities

    def _internal_maybe_mirror_and_pred_3D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: np.ndarray or torch.tensor = None) -> torch.tensor:
        assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'
        # everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        # we now return a cuda tensor! Not numpy array!

        x = to_cuda(maybe_to_torch(x), gpu_id=self.get_device())
        result_torch = torch.zeros([1, self.unet.num_classes + self.num_classes] + list(x.shape[2:]),
                                   dtype=torch.float).cuda(self.get_device(), non_blocking=True)

        if mult is not None:
            mult = to_cuda(maybe_to_torch(mult), gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                pred = self.inference_apply_nonlin(self(x))
                result_torch += 1 / num_results * pred

            if m == 1 and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4,))))
                result_torch += 1 / num_results * torch.flip(pred, (4,))

            if m == 2 and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3,))))
                result_torch += 1 / num_results * torch.flip(pred, (3,))

            if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3))

            if m == 4 and (0 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (2,))))
                result_torch += 1 / num_results * torch.flip(pred, (2,))

            if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 2))

            if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch
