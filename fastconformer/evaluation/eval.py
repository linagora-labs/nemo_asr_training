import argparse
import logging
import json
import os
import nemo.collections.asr as nemo_asr
from tqdm import tqdm
from ssak.utils.wer import compute_wer
from utils import datasets_names, logger
from plots import make_data_plots, plot_results

logging.getLogger('nemo_logger').setLevel(logging.ERROR)

def compute_checkpoint_wer(checkpoint, data: dict, output_dir, common_dir, cer=False, model_type="ctc", format_text=False, batch_size=16):
    asr_model = None
    results = dict()
    pbar = tqdm(data)
    for d in pbar:
        pbar.set_description(f"Processing dataset {d}")
        audio_files = [r["audio_filepath"] for r in data[d]]
        transcriptions = [r["text"] for r in data[d]]
        if "huggingface/hub" in checkpoint or not os.path.exists(checkpoint):
            saved_path = os.path.join(common_dir, "saved", os.path.basename(checkpoint))
        else:
            saved_path = os.path.join(output_dir, "saved", os.path.basename(os.path.splitext(checkpoint)[0]))
        pred_path = os.path.join(saved_path, f"predictions_{d}.jsonl")
        if not os.path.exists(pred_path):
            logger.info(f"Predictions not found for {pred_path}, computing them")
            os.makedirs(saved_path, exist_ok=True)
            manifest_path = os.path.join(saved_path, f"manifest_{d}.jsonl")
            num_workers = 4
            with open(manifest_path, "w") as f:
                for row in data[d]:
                    f.write(json.dumps(row) + "\n")
            del audio_files
            if asr_model is None:
                try:
                    if model_type=="ctc":
                        model_class = nemo_asr.models.EncDecCTCModelBPE
                    elif model_type.startswith("hybrid"):
                        model_class = nemo_asr.models.EncDecHybridRNNTCTCBPEModel
                    elif model_type=="rnnt":
                        model_class = nemo_asr.models.EncDecRNNTBPEModel
                    else:
                        raise RuntimeError(f"Wrong model type: {model_type}")   
                    if os.path.exists(checkpoint):
                        asr_model = model_class.restore_from(checkpoint)
                    else:
                        asr_model = model_class.from_pretrained(model_name=checkpoint)
                    if model_type=="hybrid_ctc":
                        asr_model.change_decoding_strategy(decoder_type="ctc")
                except Exception as e:
                    raise RuntimeError(f"Error while loading model from {checkpoint}: {e}")
            hypothesis = asr_model.transcribe(manifest_path, batch_size=batch_size, num_workers=num_workers)
            with open(pred_path, "w", encoding="utf-8") as f:
                for hypothesis, row in zip(hypothesis, data[d]):
                    new_row = row.copy()
                    new_row["prediction"] = hypothesis.text
                    f.write(json.dumps(new_row, ensure_ascii=False) + "\n")
        predictions = list()
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                predictions.append(json.loads(line)["prediction"])
        pred_only_path = os.path.join(saved_path, f"raw_predictions_{d}.jsonl")
        if not os.path.exists(pred_only_path):
            with open(pred_only_path, "w", encoding="utf-8") as f:
                for pred in predictions:
                    f.write(pred)
                    f.write("\n")
        perf_path = os.path.join(saved_path, f"{'cer' if cer else 'wer'}_{'nocasepunc' if format_text else 'casepunc'}_{d}.json")
        if not os.path.exists(perf_path):
            try:
                cleaned_transcriptions = []
                cleaned_predictions = []
                for transcription, prediction in zip(transcriptions, predictions):
                    cleaned_transcriptions.append(transcription.replace(".", " .").replace(",", " ,").replace("?", " ?"))
                    cleaned_predictions.append(prediction.replace(".", " .").replace(",", " ,").replace("?", " ?"))
                results[d] = compute_wer(cleaned_transcriptions, cleaned_predictions, normalization="fr+" if format_text else None, use_percents=True, character_level=cer, alignment=os.path.join(saved_path, f"{'cer' if cer else 'wer'}_alignment_{'nocasepunc' if format_text else 'casepunc'}_{d}.json"))
                results[d].pop('alignment')
                results[d].pop('raw_alignement')
            except Exception as e:
                raise RuntimeError(f"Error while computing WER on {os.path.basename(checkpoint)} for {d} ({len(transcriptions)} vs {len(predictions)}): {e}")
            with open(perf_path, "w", encoding="utf-8") as f:
                json.dump(results[d], f)
        else:
            with open(perf_path, "r", encoding="utf-8") as f:
                results[d] = json.load(f)
    return results

def compare_checkpoints_wer(checkpoints, data, output_dir, common_dir, model_type, cer=False, format_text=False, batch_size=16):
    results = dict()
    pbar = tqdm(checkpoints)
    for checkpoint in pbar:
        pbar.set_description(f"Computing WER for {os.path.basename(os.path.splitext(checkpoint)[0])}")
        results[os.path.basename(os.path.splitext(checkpoint)[0])] = compute_checkpoint_wer(checkpoint, data, output_dir, common_dir, cer=cer, model_type=model_type, format_text=format_text, batch_size=batch_size)
    if len(results) > 1:
        plot_results(results, plot_folder, cer=cer, nocasepunc=format_text)
    return results

def load_manifest(manifest_path, datasets=None):
    data = list()
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            json_line = json.loads(line)
            if float(json_line['duration']) < 0.05:
                logger.warning(f"Skipping {json_line['audio_filepath']} with duration {json_line['duration']}")
                continue
            data.append(json_line)
    logger.info(f"Computing WER on {len(data)} samples")
    data_sorted = dict()
    for i in data:
        name = i["name"] if "name" in i else i["dataset_name"]
        name = name.replace("_max30", "")
        name = name.replace("_eval", "")
        name = name.replace("_nocasepunc", "")
        name = name.replace("_casepunc", "")
        if name.lower() in datasets_names:
            name = datasets_names[name.lower()]
        
        if datasets is None or name.lower() in datasets: # and not name.lower()=="epac":
            if name not in data_sorted:
                data_sorted[name] = list()
            data_sorted[name].append(i)
    data = data_sorted
    logger.info(f"Computing WER on {len(data)} datasets")
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor checkpoints during training')
    parser.add_argument('--manifest', help="Input manifest", type=str, default="/mnt/d/linagora/finetuning_benchmark/manifests/casepunc_test_manifest_clean.jsonl")
    parser.add_argument('--checkpoints', help="Checkpoints to compare", type=str, nargs="+", default=["nvidia/stt_fr_fastconformer_hybrid_large_pc", "linagora/linto_stt_fr_fastconformer"])
    parser.add_argument('--datasets', help="Datasets to evaluate", default=None, type=str, nargs="+")
    parser.add_argument('--output_dir', help="Output directory", type=str, default="output/linto_stt_fr_fastconformer")
    parser.add_argument('--common_dir', help="Output directory", type=str, default="output/commons")
    parser.add_argument('--batch_size', help="Batch size used for inference", type=int, default=16)
    parser.add_argument('--model_type', default="hybrid")
    parser.add_argument('--mode', default="test")
    parser.add_argument('--cer', default=False, action="store_true")
    parser.add_argument('--format_text', default=False, action="store_true")
    parser.add_argument('--plot_folder', default="plots")
    parser.add_argument("--only_last_checkpoint", default=False, action="store_true")
    args = parser.parse_args()
    
    if len(args.checkpoints) == 0:
        logger.error("No checkpoints provided")
        exit(1)
    elif len(args.checkpoints) == 1 and os.path.isfile(args.checkpoints[0]) and args.checkpoints[0].endswith(".txt"):
        with open(args.checkpoints[0], "r", encoding="utf-8") as f:
            args.checkpoints = [line.strip() for line in f if not line.startswith("#")]
    elif len(args.checkpoints) == 1 and os.path.isdir(args.checkpoints[0]):
        args.checkpoints = [os.path.join(args.checkpoints[0], i) for i in os.listdir(args.checkpoints[0])]
        if args.only_last_checkpoint:
            args.checkpoints = [i for i in args.checkpoints if i.endswith("-last.nemo")]
    check_list = []
    for checkpoint in args.checkpoints:
        if os.path.isdir(checkpoint):
            check_list.extend([os.path.join(checkpoint, i) for i in os.listdir(checkpoint)])
        else:
            check_list.append(checkpoint)
    args.checkpoints = check_list

    logger.info(f"Reading manifest from {args.manifest}")

    data = load_manifest(args.manifest, args.datasets)
    if not data:
        raise ValueError(f"No data found with the provided datasets: {args.datasets}")

    # mode = os.path.basename(args.manifest).split("_")[0]
    mode = args.mode
    
    plot_folder = os.path.join(args.output_dir, mode, args.plot_folder)
    make_data_plots(data, plot_folder)
    logger.info(f"Generated data plots in {plot_folder}")
    compare_checkpoints_wer(
        args.checkpoints, 
        data, 
        os.path.join(args.output_dir, mode), 
        os.path.join(args.common_dir, mode), 
        model_type=args.model_type, 
        cer=args.cer, 
        format_text=args.format_text, 
        batch_size=args.batch_size)