import argparse
import json
import os
import logging
from tqdm import tqdm
from matplotlib import pyplot as plt
from train_data_analysis import hours_per_dataset, hours_per_dataset_pie, process_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plot_folder = "plots"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot details about test data')
    parser.add_argument('--manifest', help="Input manifest", type=str, default="manifests/test_manifest_clean.jsonl")
    parser.add_argument('--output_dir', help="Output directory", type=str, default="output_eval/test_data")
    args = parser.parse_args()
    
    logger.info(f"Reading manifest from {args.manifest}")
    
    buckets = False
    if not os.path.exists(args.manifest):
        raise FileNotFoundError(f"Manifest {args.manifest} not found")
    data = list()
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            json_data = json.loads(line)
            data.append(json_data)

    mode = os.path.basename(args.manifest).split("_")[0]
    plot_folder = "plots"
    plot_folder = os.path.join(args.output_dir, mode, plot_folder)
    os.makedirs(plot_folder, exist_ok=True)
    
    data_per_dataset, hours = process_data(data, "name")
    hours_per_dataset(hours, plot_folder)
    hours_per_dataset(hours, plot_folder, log=True)
    hours_per_dataset_pie(hours, plot_folder)

    # make a plot with distribution of durations globally
    durations = [i["duration"] for i in data]
    plt.figure(figsize=(10, 5))
    plt.hist(durations, bins=100, log=True)
    plt.xlabel("Duration (s)")
    plt.ylabel("Number of samples (log)")
    plt.title("Distribution of durations for training")
    plt.savefig(os.path.join(plot_folder, "durations_hist_log.png"), bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.hist(durations, bins=100, log=False)
    plt.xlabel("Duration (s)")
    plt.ylabel("Number of samples")
    plt.title("Distribution of durations for training")
    plt.savefig(os.path.join(plot_folder, "durations_hist.png"), bbox_inches='tight')
    plt.close()