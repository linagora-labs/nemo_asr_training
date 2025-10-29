# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import lightning.pytorch as pl
import torch
torch.set_float32_matmul_precision("high")
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from train_utils import get_base_model, setup_dataloaders, check_vocabulary, resolve_trainer_cfg



@hydra_runner(config_path="yamls", config_name="finetuning_linto_stt_fr_fastconformer.yaml")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    if hasattr(cfg, 'init_from_ptl_ckpt') and cfg.init_from_ptl_ckpt is not None:
        raise NotImplementedError(
            "Currently for simplicity of single script for all model types, we only support `init_from_nemo_model` and `init_from_pretrained_model`"
        )

    asr_model = get_base_model(trainer, cfg)
    
    # Check vocabulary type and update if needed
    asr_model = check_vocabulary(asr_model, cfg)

    # Setup Data
    asr_model = setup_dataloaders(asr_model, cfg)

    # Setup Optimizer
    asr_model.setup_optimization(cfg.model.optim)

    # Setup SpecAug
    if hasattr(cfg.model, 'spec_augment') and cfg.model.spec_augment is not None:
        asr_model.spec_augment = ASRModel.from_config_dict(cfg.model.spec_augment)
    if hasattr(cfg.model, 'freeze_encoder') and cfg.model.freeze_encoder:
        asr_model.encoder.freeze()
        logging.info("Encoder is frozen")
    else:
        logging.info("Encoder is not frozen")
    asr_model.wer.log_prediction=False
    # asr_model = torch.compile(asr_model)
    trainer.fit(asr_model)

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter