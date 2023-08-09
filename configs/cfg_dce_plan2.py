exp_name = 'MICCAI paper, DCE plan cls+det, no consist loss, no pretr'
results_dir = "./nnUNet_trained_models/PLAN/" + exp_name
#TODO Note: According to PLAN paper, first train with config cfg_dce_plan1 for 500 epochs, then train with this for 500

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
split_file = 'MICCAI_split/splits_final.pkl'
valbest = False  # use the best or final checkpoint to infer
my_online_val = True  # compute lesion-wise validation metrics in every other epoch

mask_emb_dim = 64
lesion_instance_seg = True  # whether to split lesions to instances in training
# If True, do instance/panoptic segmentation; if False, do semantic segmentation
model = dict(
    type='PLAN',
    components=('cls', 'det', 'seg'),
    num_all_classes=num_all_classes, num_lesion_classes=num_lesion_classes,
    liver_label=9,  # liver label in the raw training ground-truth masks
    mask_emb_dim=mask_emb_dim,
    pixel_decoder='fpn',  # fpn or unet
    nnunet_deep_supervision=False,
    prediction_stride=1,  # set as 2, 4... to reduce memory usage
    add_anchor_query=False,  # see PLAN paper, add anchor queries to det branch generated from seg branch

    seg=dict(
        ft_dim=mask_emb_dim, conv_layers=0, loss=dict(weight=2),
    ),
    det=dict(
        train_start_ep=0,  # Do not compute loss for the det branch until this epoch
        transformer_in_feature=[-5, -4, -3],  # negative indicates decoder, positive indicates encoder
        mask_emb_dim=mask_emb_dim,
        lesion_instance_seg=lesion_instance_seg,
        transformer_predictor=dict(
            in_channels=[64, 64, 64], mask_classification=True, hidden_dim=mask_emb_dim, num_queries=50, nheads=8,
            num_feature_levels=3, dim_feedforward=mask_emb_dim * 8, dec_layers=3, pre_norm=False,
            deep_supervision=True, mask_dim=mask_emb_dim, enforce_input_project=False, use_pos_emb=True,
        ),
        # anchor_matcher=dict(CC_th=.1, min_CC_size=10, max_CC_num=20, lesion_label_groups=[[1, 2, 3, 4, 5, 6, 7, 8]]),
        loss=dict(
            # Mask2Former original parameters
            no_object_weight=.1, dice_weight=5, mask_weight=5, cls_type='ce', cls_weight=2,
            train_num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75,

            fg_sample_ratio=0.0002,  # See foreground-enhanced sampling loss in PLAN paper
            hard_neg_sample_ratio=0.0, hard_sample_ratio=0.0,  # useless in my experiments
            weight=1,
        ),
    ),
    cls=dict(
        merge_classes=[[1, 2, 3, 4, 5, 6, 8], [1, 2, 3, 4], [5, 6],
                       *[[c] for c in range(1, 9)]],  # in patient branch, we do multi label classification,
        # each label indicating whether the current patch contains certain classes.
        # If merge_classes[0]==[1,2,3], it means the first label of the patient branch classifies the existence of
        # any class in classes 1,2,3
        train_start_ep=200,  # It seems good to start learning the patient branch after learning the det/seg branch
        # for several epochs
        num_patches=192, embed_dim=128, size_th=100, use_pos_emb=True,
        transformer_in_feature=[2, 4, -5, -3],  # negative indicates decoder, positive indicates encoder
        input_dims=[128, 320, 64, 64],
    ),
    consist_loss=dict(train_start_ep=50, type='det_cls', weight=0.1),
)
optimizer = dict(
    max_num_epochs=1000,
    backbone_multiplier=1, backbone_name_include=('unet.',), backbone_name_exclude=('fpn', 'emb')
)

data_loader = dict(
    type='DataLoader3D_Inst',
    num_lesion_classes=num_lesion_classes,  # How many lesion classes we want to segment in our model
    num_lesion_classes_in_annot=8,  # How many lesion classes are their in the training mask. We will merge them using
    # `merge_classes`
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
