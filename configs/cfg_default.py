num_lesion_classes, num_all_classes = 8, 15
# in raw mask, 1...num_lesion_classes is lesion, num_lesion_classes+1...num_all_classes-1 is organ
# Class 0 in num_all_classes is bg (not in any organ)
seg_pretrain_dir = None
gpu = ''  # if '', pick the idlest one. Otherwise, use an integer

# nnUNet basic parameters
network = '3d_fullres'
p = 'nnUNetPlansv2.1'  # plans_identifier
task = 282
fold = 0
use_simplified_train_folder = True  # nnunet v1 has a complicated folder structure. We simplify it.
split_file = 'splits_final.pkl'
disable_postprocessing_on_folds = True  # don't try to keep only the largest CC for each class and compute accuracy
my_online_val = True  # nnUNet use global Dice in online validation, which may be inconsistent with our
# needs which focus on detection precision, recall, and accuracy. So we add some lesion-wise metrics in validation.
# But we are not using this customized val metric to select best ep yet.
deterministic = True  # setting to true benefits algorithm comparison, but may cause overfit. After testing, I found
# setting to True actually cannot be deterministic

ignore_class_label = 100  # don't compute the segmentation and classification loss for this class id
model = dict(
    num_all_classes=num_all_classes, num_lesion_classes=num_lesion_classes,
    prediction_stride=1,  # set as 2, 4... to reduce memory usage
    ignore_class_label=ignore_class_label, liver_label=9,
    nnunet_deep_supervision=False,  # disable deep sup still OOM 16G memory
    loss=dict()
)
optimizer = dict(
    method='RAdam',
    max_num_epochs=1000,
    initial_lr=1e-4,
    fix_unet_epoch=0,
    backbone_multiplier=1, backbone_name_include=('unet.', ), backbone_name_exclude=('fpn', 'emb')
)

data_loader = dict(
    num_lesion_classes=num_lesion_classes,
    fg_force_lesion=True,  # when sampling fg in data loader, only consider lesions as fg (organs are not)
    fg_prob_continuous=True,  # if False, in the current nnUNet, if p=0.33, batch_size=2, the actual p=.5.
    oversample_foreground_percent=0.33,
    min_train_lesion_size=10,  # if a lesion mask is smaller than this number of pixels in training, set as ignore,
    # because it may be a wrong annotation or too insignificant
    max_neg_lesion_size_mm3=0,  # if a lesion mask's volume is smaller than this number of mm3 in training, set as bg,
    # because it may be a wrong annotation or too insignificant
    ignore_class_label=ignore_class_label,
)

validation = dict(
    model=network,
    plans_identifier=p,
    folds=[str(fold)],
    verbose=False,
    torch_interpolate=False,  # if True, faster but less sensitivity for small lesions
    use_sliding_window=True,
    folders=[],

    # post processing
    remove_lesion_outside_liver=3,
    lesion_volume_th=25,  # in mm3
    det_lesion_volume_th=113,  # ball with R=3mm
    combine_label_idx=list(range(1, num_lesion_classes+1)),
)
