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
# Modified by YK

import sys
import os.path as osp
import time
from functools import partial

import torch

sys.path.append(osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, 'configs')))
from configs.config_utils import load_args  # modify nnunet.paths.network_training_output_dir in the beginning
args = load_args()
from nnunet.inference.my_predict import main as validate_folders

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration
from nnunet.paths import default_plans_identifier
from nnunet.run.load_pretrained_weights import load_pretrained_weights
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name


def main():
    if args.mode in ['val', 'eval_only']:
        validate_folders()
        return

    task = str(args.task)
    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    plans_identifier = args.p
    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data
    deterministic = args.deterministic
    fp32 = args.fp32
    run_mixed_precision = not fp32

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
    trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")

    if network == "3d_cascade_fullres":
        assert issubclass(trainer_class, (nnUNetTrainerCascadeFullRes, nnUNetTrainerV2CascadeFullRes)), \
            "If running 3d_cascade_fullres then your " \
            "trainer class must be derived from " \
            "nnUNetTrainerCascadeFullRes"
    else:
        assert issubclass(trainer_class,
                          nnUNetTrainer), "network_trainer was found but is not derived from nnUNetTrainer"

    trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic,
                            fp16=run_mixed_precision)
    if args.disable_saving:
        trainer.save_final_checkpoint = False  # whether or not to save the final checkpoint
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to
        # self.best_val_eval_criterion_MA
        trainer.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
        # the training chashes
        trainer.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

    trainer.initialize()

    if args.mode == 'cont_train':
        # continue a previous training and ignore pretrained weights
        trainer.load_latest_checkpoint()
    elif (args.mode != 'cont_train') and (args.pretrained_weights is not None):
        # we start a new training. If pretrained_weights are set, use them
        load_pretrained_weights(trainer.network, args.pretrained_weights, verbose=True)
    else:
        # new training without pretrained weights, do nothing
        pass
    t = time.time()
    trainer.run_training()
    trainer.print_to_log_file(f'Training took {(time.time()-t)/3600:.1f} hours')

    torch.cuda.empty_cache()
    validate_folders()


if __name__ == "__main__":
    main()

