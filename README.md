# NeMo ASR Training

This repository provides recipes to train (french) ASR models (either from scratch or by finetuning) using the NVIDIA NeMo toolkit.

## Data

The data used in these recipes is processed using the [ssak NeMo pipeline](https://github.com/linagora-labs/ssak/tree/main/tools/nemo). This pipeline takes Kaldi datasets and converts and cleans them into a single `train_manifest.jsonl`. Once generated, this manifest can be used to create a tokenizer or bucketed datasets to accelerate training.

You can analyze the number of hours per dataset and per bucket in your training, development, and test manifests by using the [visualization tool](fastconformer/visualization).

## Monitoring

Training checkpoints can be monitored using [eval.py](fastconformer/monitoring/eval.py), which provides insights into training progress.

## Evaluation

You can use the [ASR Benchmark repository](https://github.com/linagora-labs/asr_benchmark) to evaluate the model's performance against other models.

## Recipes

### Finetuning a French FastConformer

The recipe [linto\_stt\_fr\_fastconformer](fastconformer/yamls/finetuning_linto_stt_fr_fastconformer.yaml) finetunes the [FastConformer-Hybrid Large FR model](https://huggingface.co/nvidia/stt_fr_fastconformer_hybrid_large_pc) ([base config file](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_transducer_ctc_bpe.yaml)) on more than 9,000 hours of French speech to enhance its performance.

The list of datasets used during finetuning is available [here](fastconformer/datasets/dataset_list_nocacepunc.json).

The resulting model is available on Hugging Face: [linto\_stt\_fr\_fastconformer](https://huggingface.co/linagora/linto_stt_fr_fastconformer).
