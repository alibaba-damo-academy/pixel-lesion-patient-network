# YK, based on nnunet/network_architecture/attn_unet/my_generic_UNet.py, increase
# the dim of last few layers of decoder to generate mask embedding for maskFormer

from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin, StackedConvLayers, Upsample
from nnunet.network_architecture.mask2former.fpn import BasePixelDecoder
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional

from configs.config_utils import cfg


class Generic_UNet(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(Generic_UNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        self.cfg = cfg.model

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.emb_outputs = []

        output_features = base_num_features
        input_features = input_channels
        print('input_features', input_features, 'output_features', output_features)
        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)
            print('input_features', input_features, 'output_features', output_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels
        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        # noinspection PyTypeChecker
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)

        if self.cfg.pixel_decoder == 'unet':  # ori nnunet decoder, cause OOM in maskFormer
            self.build_decoder_unet(pool_op_kernel_sizes, num_pool, final_num_features, num_conv_per_stage,
                                    basic_block, dropout_in_localization, upsample_mode, transpconv, conv_op,
                                    seg_output_use_bias)
            if self.weightInitializer is not None:
                self.apply(self.weightInitializer)  # fpn has diff init method than unet
                # self.apply(print_module_training_status)

        elif self.cfg.pixel_decoder == 'fpn':  # fpn decoder in maskFormer
            assert not self._deep_supervision
            self.build_decoder_fpn(self.cfg, final_num_features)

    def build_decoder_fpn(self, cfg, bottleneck_channels):
        feature_channels = [block.output_channels for block in self.conv_blocks_context[:-1]] + [bottleneck_channels]
        cfg.backbone_feature_channels = feature_channels
        self.fpn = BasePixelDecoder(feature_channels, conv_dim=cfg.mask_emb_dim, mask_dim=cfg.mask_emb_dim,
                                    final_stride=cfg.prediction_stride)

    def build_decoder_unet(self, pool_op_kernel_sizes, num_pool, final_num_features, num_conv_per_stage,
                           basic_block, dropout_in_localization, upsample_mode, transpconv, conv_op,
                           seg_output_use_bias):
        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2

            # TODO: what is the best way to generate embeddings? Should we use FPN like in MaskFormer?
            nfeatures_after_tu = max(nfeatures_from_skip,
                                     self.cfg.mask_emb_dim)  # YK, keep the dim of decoder ft for mask emb
            # n_features_after_tu_and_concat = nfeatures_from_skip * 2  # from skip + from tu
            n_features_after_tu_and_concat = nfeatures_from_skip + nfeatures_after_tu

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip
            final_num_features = max(final_num_features,
                                     self.cfg.mask_emb_dim)  # YK, keep the dim of decoder ft for mask emb
            nfeatures_med = max(nfeatures_from_skip,
                                self.cfg.mask_emb_dim)  # YK, keep the dim of decoder ft for mask emb
            print('n_features_after_tu_and_concat', n_features_after_tu_and_concat, 'nfeatures_from_skip',
                  nfeatures_from_skip, 'final_num_features', final_num_features)

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_after_tu, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_med, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_med, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.emb_outputs.append(
                conv_op(self.conv_blocks_localization[ds][-1].output_channels, self.cfg.mask_emb_dim,
                        1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.tu = nn.ModuleList(self.tu)
        self.emb_outputs = nn.ModuleList(self.emb_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

    def fix_unet(self, fix=True):
        backbone_names = [name for name, _ in self.network.named_parameters()
                          if any([n in name for n in cfg.optimizer.backbone_name_include]) and
                          not any([n in name for n in cfg.optimizer.backbone_name_exclude])]
        if fix and (not self.unet_fixed):
            for name, params in self.unet.named_parameters():
                if name in backbone_names:
                    params.requires_grad = False
            self.unet_fixed = True
        elif (not fix) and self.unet_fixed:
            for params in self.unet.parameters():
                if name in backbone_names:
                    params.requires_grad = True
            self.unet_fixed = False

    def forward(self, x, return_ft=False):
        skips = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        self.encoder_fts = skips
        x = self.conv_blocks_context[-1](x)

        if self.cfg.pixel_decoder == 'unet':  # ori nnunet decoder, cause OOM in maskFormer
            fts = []
            emb_outputs = []
            for u in range(len(self.tu)):
                x = self.tu[u](x)
                x = torch.cat((x, skips[-(u + 1)]), dim=1)
                x = self.conv_blocks_localization[u](x)
                fts.append(x)
                emb_outputs.append(self.final_nonlin(self.emb_outputs[u](x)))

            self.decoder_fts = fts

            if self._deep_supervision and self.do_ds:  # TODO: is deep supervision of embeddings useful?
                return tuple([emb_outputs[-1]] + [i(j) for i, j in
                                                  zip(list(self.upscale_logits_ops)[::-1], emb_outputs[:-1][::-1])])
            else:
                return emb_outputs[-1],
        elif self.cfg.pixel_decoder == 'fpn':  # fpn decoder in maskFormer
            ft, decoder_fts = self.fpn(skips + [x])
            self.decoder_fts = decoder_fts
            return ft,


class Generic_UNet_FPN_Decoder(Generic_UNet):
    """To test the effect of unet encoder + fpn decoder for direct segmentation wo deep sup"""
    def __init__(self, *args, **kwargs):
        super(Generic_UNet_FPN_Decoder, self).__init__(*args, **kwargs)
        assert self.cfg.pixel_decoder == 'fpn'
        self.seg_output = nn.Conv3d(self.cfg.mask_emb_dim, self.num_classes,
                                        1, 1, 0, 1, 1, False)

    def forward(self, x, return_ft=False):
        fts = super().forward(x, return_ft)
        seg_outputs = self.final_nonlin(self.seg_output(fts[0]))
        return [seg_outputs]

    def load_from(self, ckpt_path):
        pretrained_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        try:
            self.load_state_dict(pretrained_dict)
        except Exception as e:
            print(e)
            print('Try only load weights of the same shape in the UNet: ')
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(pretrained_dict.keys())
            print('done')
