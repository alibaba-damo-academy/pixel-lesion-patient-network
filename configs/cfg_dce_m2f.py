exp_name = 'Mask2Former wo fg enhancing loss, MICCAI DCE seg'
results_dir = "./nnUNet_trained_models/PLAN/" + exp_name

gpu = ''  # if '', pick the idlest GPU; otherwise, use an integer
seg_pretrain_dir = None  # pretraining weights
num_lesion_classes, num_all_classes = 8, 15
# Class 0 is bg, 1 ~ num_lesion_classes are lesion classes, num_lesion_classes+1 ~ num_all_classes are organ classes

mode = 'train'  # options: train, cont_train, val, eval_only
# cont_train: continue training from existing checkpoint;
# val: infer the test sets specified in validation.folders below. If mode==train or cont_train, validation.folders will
#   also be inferred.
# eval_only: after inference, compute accuracy for the validation set

network_trainer = 'panoSegTrainer'
task = 287
p = 'nnUNetPlansv2.1'  # plans_identifier
valbest = False  # use the best or final checkpoint to infer
my_online_val = True  # compute lesion-wise validation metrics in every other epoch

mask_emb_dim = 64
lesion_instance_seg = True  # whether to split lesions to instances in training
# If True, do instance/panoptic segmentation; if False, do semantic segmentation
model = dict(
    type='Mask2Former',
    pixel_decoder='fpn',  # fpn or unet, unet causes OOM
    transformer_in_feature=[-5, -4, -3],
    mask_emb_dim=mask_emb_dim,
    norm_pixel_emb='none',  # none or L2 or IN
    nnunet_deep_supervision=False,  # disable deep sup still OOM 16G memory
    lesion_instance_seg=lesion_instance_seg,
    prediction_stride=1,  # set as 2, 4... to reduce memory usage
    transformer_predictor=dict(
        in_channels=[64, 64, 64], mask_classification=True, hidden_dim=mask_emb_dim, num_queries=50, nheads=8,
        num_feature_levels=3, dim_feedforward=mask_emb_dim*8, dec_layers=3, pre_norm=False,
        deep_supervision=True, mask_dim=mask_emb_dim, enforce_input_project=False,
        # MaskFormer original paper:
        # in_channels=256, mask_classification=True, hidden_dim=256, num_queries=100, nheads=8,
        # dropout=0., dim_feedforward=2048, dec_layers=9, pre_norm=False,
        # mask_dim=256, enforce_input_project=False,
    ),
    loss=dict(
        type='maskformer',  # maskformer or nnunet
        no_object_weight=.1, dice_weight=5, mask_weight=5, cls_type='ce', cls_weight=2,
        train_num_points=12544,  # TODO is this number good for 3D patch? or x1.5?
        oversample_ratio=3.0, importance_sample_ratio=0.75, hard_sample_ratio=0.0,
        fg_sample_ratio=0.000, hard_neg_sample_ratio=0.0,  # set fg_sample_ratio to 0.0002 to enhance foreground sens
    )
)
optimizer = dict(
    max_num_epochs=1000,
    backbone_multiplier=1, backbone_name_include=('unet.',), backbone_name_exclude=('fpn', 'emb')
)

data_loader = dict(
    type='DataLoader3D_Inst',
    num_lesion_classes=num_lesion_classes,  # How many lesion classes we want to segment in our model
    num_lesion_classes_in_annot=8,  # How many lesion classes are their in the training mask. We will merge them using
    # `merge_classes` if merge_classes is set
    lesion_instance_seg=lesion_instance_seg,  # if True, split lesion in each class to instances
    preprocess_mask=True,  # convert semantic masks to instance masks and labels
)

# inference data
data_fd = '/data/vdd/user/codes/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task287_Task286_NAV/'
output_fd = results_dir + '/test_results'
validation = dict(
    trainer_class_name=network_trainer,
    folders=[
        # each tuple: image input folder, mask & class output folder,
        # evaluation function (if exist), mask ground-truth folder for evaluation (if exist)
        (data_fd + 'imagesTs', output_fd, 'eval_diagnosis', data_fd + 'labelsTs'),
    ],
    class_names=['HCC', 'chola', 'meta', 'hepato', 'heman', 'FNH', 'cyst', 'other',],
    # the class names of our model's mask prediction
    malig_benign_cyst_groups=[[1, 2, 3, 4], [5, 6], [7]],  # `other` can be either malig or benign
    compute_sens_on_classes=[1, 2, 3, 4, 5, 6, 8],  # since normal samples may also have cyst, so we ignore cyst prediction when
    # computing sensitivity
    class_channel_dict=dict(tumor_non_cyst=[0], malig=[1], benign=[2], each_class=list(range(3, 11))),
    combine_label_idx=[1, 2, 3, 4, 5, 6, 8]  # In postprocess, we don't combine cyst label with other labels since cysts
    # are sometimes adjacent to other lesion types
)
