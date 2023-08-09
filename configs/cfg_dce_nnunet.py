exp_name = 'original nnunet, task 287, w Fk pretr, MICCAI DCE seg'
num_lesion_classes = 8
seg_pretrain_dir = None
results_dir = "./nnUNet_trained_models/ori_nnunet/" + exp_name
gpu = ''  # if '', pick the idlest one

# nnUNet basic parameters
mode = 'train'  # train, cont_train, val, eval_only
network = '3d_fullres'
p = 'nnUNetPlans_pretrained_FINETUNE'  # plans_identifier
split_file = 'MICCAI_split/splits_final.pkl'

task = 287
fold = 0
network_trainer = 'nnUNetTrainerV2'
valbest = False
my_online_val = True

optimizer = dict(max_num_epochs=1000)
initial_lr = 0.01

data_loader = dict(
    num_lesion_classes=num_lesion_classes,  # How many lesion classes we want to segment in our model
    num_lesion_classes_in_annot=8,  # How many lesion classes are their in the training mask.
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
