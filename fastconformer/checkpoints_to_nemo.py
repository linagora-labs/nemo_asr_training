import os
import argparse
import logging
import torch
import yaml
import nemo.collections.asr as nemo_asr
import lightning.pytorch as pl
from lightning.fabric.utilities.cloud_io import _load as pl_load
from lightning.pytorch.utilities.migration import pl_legacy_patch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_tokenizer(tokenizer_folder):
    new_tokenizer_dict = {'type': 'bpe'}
    if os.path.isdir(tokenizer_folder):
        new_tokenizer_dict['dir'] = tokenizer_folder
        for file in os.listdir(tokenizer_folder):
            if file.endswith(".model"):
                new_tokenizer_dict["model_path"] = os.path.join(tokenizer_folder, file)
            elif file.endswith(".vocab"):
                new_tokenizer_dict["spe_tokenizer_vocab"] = os.path.join(tokenizer_folder, file)
            elif file.endswith(".txt"):
                new_tokenizer_dict["vocab_path"] = os.path.join(tokenizer_folder, file)
        logger.info(f"Loaded {new_tokenizer_dict} from tokenizer folder {tokenizer_folder}")
    else:
        raise FileNotFoundError(f"Tokenizer folder {tokenizer_folder} does not exist")
    return new_tokenizer_dict


def checkpoint_to_nemo(checkpoint_path, input_dir, output_dir, new_tokenizer_dict, model_type):
    checkpoint_name = checkpoint_path
    checkpoint_path = os.path.join(input_dir, checkpoint_path)
    output_path = os.path.join(output_dir, checkpoint_name)
    output_path = output_path.replace(".ckpt", ".nemo")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint {checkpoint_path} does not exist, skipping")
    elif not os.path.exists(output_path):
        logger.info(f"Loading checkpoint {checkpoint_path}")
        cfg=None
        if new_tokenizer_dict:
            with pl_legacy_patch():
                checkpoint = pl_load(checkpoint_path, map_location="cuda")
                cfg = checkpoint[pl.LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]
                cfg['cfg']['tokenizer'] = new_tokenizer_dict
        if model_type == "hybrid":
            model_class = nemo_asr.models.EncDecHybridRNNTCTCBPEModel
        elif model_type == "rnnt":
            model_class = nemo_asr.models.EncDecRNNTBPEModel
        else:
            model_class = nemo_asr.models.EncDecCTCModelBPE
        if cfg:
            model = model_class.load_from_checkpoint(checkpoint_path, map_location="cuda", **cfg)
        else:
            model = model_class.load_from_checkpoint(checkpoint_path, map_location="cuda")
        logger.info(f"Model loaded {checkpoint_name}, saving to {output_path}")
        model.save_to(output_path)
        logger.info(f"Saved model to {output_path}")
    else:
        logger.info(f"Model already exists at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getenv("SCRATCH")))
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None)
    parser.add_argument("--model_type", type=str, default="hybrid")
    parser.add_argument("--tokenizer_folder", type=str, default=os.path.join(os.getenv("WORK"),"model_tokenizers/stt_fr_fastconformer_hybrid_large_pc"))
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir
    os.makedirs(args.output_dir, exist_ok=True)
    if args.checkpoints is None:
        checkpoints = os.listdir(args.input_dir)
        checkpoints = [i for i in checkpoints if i.endswith('.ckpt')]
    logger.info(f"Found {len(checkpoints)} checkpoints in {args.input_dir}")
    new_tokenizer_dict = None
    if args.tokenizer_folder is not None:
        new_tokenizer_dict = fix_tokenizer(args.tokenizer_folder)
    for checkpoint in checkpoints:
        # checkpoint_dir = checkpoint_dirs.get(checkpoint.split("-")[0], None)
        checkpoint_dir = args.input_dir
        checkpoint_to_nemo(checkpoint, args.input_dir, args.output_dir, new_tokenizer_dict, args.model_type)
    logger.info("All done!")