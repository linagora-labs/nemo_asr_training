from eval import load_manifest, logger
import argparse
import json
import os
from tqdm import tqdm
from linastt.utils.wer import compute_wer
from linastt.utils.text import format_text_latin
from faster_whisper import WhisperModel
import pydub

converted_folder = "converted_audios"

import logging
logging.getLogger('faster_whisper').setLevel(logging.ERROR)

CER = False

def compute_whisper(asr_model, data):
    if isinstance(asr_model, str):
        asr_model = WhisperModel(asr_model.replace("faster-whisper_", ""), device="cuda", compute_type="int8", num_workers=4)
    new_data = []
    for row in tqdm(data):
        file = row["audio_filepath"]
        offset = float(row.get("offset", 0))
        duration = float(row['duration'])
        new_name = os.path.join(converted_folder, row['name'], os.path.splitext(os.path.basename(file))[0], f"{os.path.splitext(os.path.basename(file))[0]}_{offset:.1f}_{duration:.1f}.wav")
        os.makedirs(os.path.dirname(new_name), exist_ok=True)
        if not os.path.exists(new_name):
            audio = pydub.AudioSegment.from_file(file)
            audio = audio[offset*1000:offset*1000+duration*1000]
            audio.export(new_name, format="wav")
        file = new_name
        segments, info = asr_model.transcribe(file, beam_size=1, best_of=1, temperature=0.0, condition_on_previous_text=False, language="fr")#, batch_size=16)
        p = " ".join([seg.text for seg in segments]).strip()
        new_row = row.copy()
        new_row["prediction"] = format_text_latin(p)
        new_data.append(new_row)
    return asr_model, new_data

def compute_model_wer(model_path, data: dict, output_dir):
    asr_model = None
    results = dict()
    pbar = tqdm(data)
    for d in pbar:
        pbar.set_description(f"Computing WER for {d}")
        saved_path = os.path.join(output_dir, "saved", os.path.basename(os.path.splitext(model_path)[0]), f"predictions_{d}.jsonl")
        if os.path.exists(saved_path):
            transcriptions = [r["text"] for r in data[d]]
            predictions = list()
            with open(saved_path, "r", encoding="utf-8") as f:
                for line in f:
                    predictions.append(json.loads(line)["prediction"])
        else:
            # save manifest
            os.makedirs(os.path.dirname(saved_path), exist_ok=True)
            manifest_path = os.path.join(output_dir, "saved", os.path.basename(os.path.splitext(model_path)[0]), f"manifest_{d}.jsonl")
            with open(manifest_path, "w") as f:
                for row in data[d]:
                    f.write(json.dumps(row) + "\n")
            if model_path.startswith("faster-whisper"):
                asr_model, new_data =  compute_whisper(model_path if asr_model is None else asr_model, data[d])
            os.makedirs(os.path.dirname(saved_path), exist_ok=True)
            predictions = []
            transcriptions = [] 
            with open(saved_path, "w", encoding="utf-8") as f:
                for row in new_data:
                    f.write(json.dumps(row) + "\n")
                    predictions.append(row["prediction"])
                    transcriptions.append(row["text"])
        saved_path = os.path.join(output_dir, "saved", os.path.basename(os.path.splitext(model_path)[0]), f"{'cer' if CER else 'wer'}_{d}.json")
        if not os.path.exists(saved_path):
            try:
                results[d] = compute_wer(transcriptions, predictions, use_percents=True, character_level=CER)
            except Exception as e:
                raise RuntimeError(f"Error while computing WER on {os.path.basename(model_path)} for {d} ({len(transcriptions)} vs {len(predictions)}): {e}")
            with open(saved_path, "w", encoding="utf-8") as f:
                json.dump(results[d], f)
        else:
            with open(saved_path, "r", encoding="utf-8") as f:
                results[d] = json.load(f)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute WER or CER for whisper')
    parser.add_argument('--manifest', help="Input manifest", type=str, default="input_manifests/test_manifest_clean_subsampled_v2.jsonl")
    parser.add_argument('--models', help="Models to compare", type=str, nargs="+", default=["faster-whisper_large-v3"])
    parser.add_argument('--datasets', help="Datasets to evaluate", default=None, type=str, nargs="+")
    parser.add_argument('--output_dir', help="Output directory", type=str, default="whisper")
    args = parser.parse_args()
    
    if len(args.models) == 0:
        logger.error("No checkpoints provided")
        exit(1)
    elif len(args.models) == 1 and os.path.isfile(args.models[0]):
        with open(args.models[0], "r", encoding="utf-8") as f:
            args.models = [line.strip() for line in f]

    logger.info(f"Reading manifest from {args.manifest}")

    data = load_manifest(args.manifest, datasets=args.datasets)

    mode = os.path.basename(args.manifest).split("_")[0]
        
    for model in args.models:
        results = compute_model_wer(model, data, os.path.join(args.output_dir, mode))
        print(f"Results for {model}:")
        for i in results:
            print(f"\t{i}: {results[i]}")

