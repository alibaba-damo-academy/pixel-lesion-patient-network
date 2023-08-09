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

"""Modified by Ke for support of config files"""
import time
from functools import partial

from configs.config_utils import load_args  # modify nnunet.paths.network_training_output_dir in the beginning
args = load_args(is_train=False)

import argparse
import torch
from nnunet.inference.predict import predict_from_folder
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from batchgenerators.utilities.file_and_folder_operations import join, isdir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.evaluation.evaluation_diagnosis import eval_diagnosis
from nnunet.evaluation.evaluation_screening import eval_sens_spec


def main():
    # input_folder = args.validation.input_folder
    # output_folder = args.validation.output_folder
    part_id = args.validation.part_id
    num_parts = args.validation.num_parts
    folds = args.validation.folds
    save_npz = args.validation.save_npz
    lowres_segmentations = args.validation.lowres_segmentations
    num_threads_preprocessing = args.validation.num_threads_preprocessing
    num_threads_nifti_save = args.validation.num_threads_nifti_save
    disable_tta = args.validation.disable_tta
    step_size = args.validation.step_size
    # interp_order = args.validation.interp_order
    # interp_order_z = args.validation.interp_order_z
    # force_separate_z = args.validation.force_separate_z
    overwrite_existing = args.validation.overwrite_existing
    mode = args.validation.mode
    all_in_gpu = args.validation.all_in_gpu
    model = args.validation.model
    trainer_class_name = args.validation.trainer_class_name
    cascade_trainer_class_name = args.validation.cascade_trainer_class_name
    args.validation.chk = 'model_best' if args.valbest else 'model_final_checkpoint'
    task_name = str(args.task)

    if not task_name.startswith("Task"):
        task_id = int(task_name)
        task_name = convert_id_to_task_name(task_id)

    assert model in ["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"], "-m must be 2d, 3d_lowres, 3d_fullres or " \
                                                                             "3d_cascade_fullres"

    if lowres_segmentations == "None":
        lowres_segmentations = None

    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == "None":
        folds = None
    else:
        raise ValueError("Unexpected value for argument folds")

    assert all_in_gpu in ['None', 'False', 'True']
    if all_in_gpu == "None":
        all_in_gpu = None
    elif all_in_gpu == "True":
        all_in_gpu = True
    elif all_in_gpu == "False":
        all_in_gpu = False

    if model == "3d_cascade_fullres":
        trainer = cascade_trainer_class_name
    else:
        trainer = trainer_class_name

    if 'use_simplified_train_folder' in args and args.use_simplified_train_folder:  # YK
        model_folder_name = join(network_training_output_dir)  # don't know why we need such a complex folder structure
    else:
        model_folder_name = join(network_training_output_dir, model, task_name, trainer + "__" +
                                 args.validation.plans_identifier)
    print("using model stored in", model_folder_name)
    assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

    folders = args.validation.folders
    t = time.time()
    for item in folders:
        input_folder, output_folder = item[0], item[1]
        if args.mode != 'eval_only':
            predict_from_folder(model_folder_name, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                            num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                            overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                            mixed_precision=not args.validation.disable_mixed_precision,
                            step_size=step_size, checkpoint_name=args.validation.chk)
            print(f'Validation took {(time.time() - t) / 60:.1f} minutes')
        if len(item) == 4:  # contains information for evaluation
            output_folder += '_postproc'
            eval_fun, gt_folder = eval(item[2]), item[3]
            eval_fun(output_folder, gt_folder)


if __name__ == "__main__":
    main()
