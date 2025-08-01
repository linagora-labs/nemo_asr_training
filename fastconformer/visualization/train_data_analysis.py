import argparse
import json
import os
import logging
from tqdm import tqdm
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

datasets_names = {
    "mls_facebook_french": "MLS",
    "youtubefr_split6": "YouTube",
    "youtubefr_split5": "YouTube",
    "youtubefr_split4": "YouTube",
    "youtubefr_split3": "YouTube",
    "youtubefr_split2": "YouTube",
    "youtubefr_split1": "YouTube",
    "youtubefr_split0": "YouTube",
    "tcof_enfants": "TCOF",
    "tcof_adultes": "TCOF",
    "yodas_fr000": "Yodas"
}

def hours_per_dataset(data, plot_folder, plot_name="hours_per_dataset", log=False):
    plt.figure(figsize=(10, 5))
    plt.bar(data.keys(), data.values(), log=log)
    # write values on top of bars
    for i, v in enumerate(data.values()):
        if v<20:
            v = round(v,1)
        else:
            v = round(v)
        plt.text(i, v, f"{v}", ha='center', va='bottom')
    plt.ylabel(f"Hours {'(log)' if log else ''}")
    plt.title("Hours per dataset")
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(plot_folder, f"{plot_name}{'_log' if log else ''}.png"), bbox_inches='tight')
    plt.close()

def hours_per_dataset_pie(hours, plot_folder, plot_name="hours_pie", plot_title=""):
    total = sum(hours.values())
    other = sum([v for k, v in hours.items() if v/total <= 0.01])
    hours = {k: v for k, v in hours.items() if v/total > 0.01}
    if other > 0:
        hours["Others"] = other
    plt.figure(figsize=(10, 5))
    plt.pie(hours.values(), labels=hours.keys(), autopct='%1.1f%%', startangle=180)
    plt.legend(hours.keys(), title="Datasets", bbox_to_anchor=(1, 1), loc="upper right", bbox_transform=plt.gcf().transFigure)
    plt.title(f"Hours distribution per dataset for training {plot_title}")
    # add a legend saying total hours
    text = f"Total hours: {total:.0f}h\n"
    for k, v in hours.items():
        if v<10:
            text += f"{k}: {v:.1f}h\n"
        else:
            text += f"{k}: {v:.0f}h\n"
    plt.text(1, 0.5, text, ha='right', va='top', transform=plt.gcf().transFigure)
    plt.savefig(os.path.join(plot_folder, f"{plot_name}.png"), bbox_inches='tight')
    plt.close()

def process_data(data, dict_key="name"):
    data_sorted = dict()
    for i in tqdm(data, desc="Grouping by dataset"):
        if dict_key not in i:
            raise ValueError(f"Key {dict_key} not found in data")
        key = i[dict_key]
        if dict_key == "name":
            key = key.replace("_nocasepunc", "")
            key = key.replace("_casepunc", "")
            key = key.replace("_max30", "")
            key = key.replace("_cleaned", "")
            key = key.replace("_eval", "")
            key = key.replace("_test", "")
            key = key.replace("_devtest", "")
        if key.lower() in datasets_names:
            key = datasets_names[key.lower()]
        if key not in data_sorted:
            data_sorted[key] = list()
        data_sorted[key].append(i)
    hours = dict()
    for d in tqdm(data_sorted, desc="Computing hours"):
        hours[d] = sum([i["duration"] for i in data_sorted[d]]) / 3600
    # sort by hours
    hours = {k: hours[k] for k in sorted(hours.keys(), key=lambda x: hours[x], reverse=True)}
    return data_sorted, hours


def load_bucket(path, data, b):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            json_data = json.loads(line)
            json_data["bucket"] = b
            data.append(json_data)
    return data

def apply_upsampling(data):
    factor = {
        "up1": 1,
        "up2": 2,
        "up3": 3,
    }
    new_data = list()
    for d in tqdm(data, desc="Applying upsampling"):
        up = d["bucket"].split("_")[0]
        for i in range(factor[up]):
            new_data.append(d)
    return new_data

plot_folder = "plots"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for Nemo')
    parser.add_argument('--manifest', help="Input manifest", type=str, default="manifests/train_manifest_clean.jsonl")
    parser.add_argument('--output_dir', help="Output directory", type=str, default="output_train/train_data")
    args = parser.parse_args()
    
    logger.info(f"Reading manifest from {args.manifest}")
    
    
    def load_data():
        buckets = False
        if not os.path.exists(args.manifest):
            raise FileNotFoundError(f"Manifest {args.manifest} not found")
        elif os.path.isdir(args.manifest):
            buckets = os.listdir(args.manifest)
            data = list()
            for b in tqdm(buckets, desc="Loading buckets"):
                if os.path.exists(os.path.join(args.manifest, b, "tarred_audio_manifest.json")):
                    data = load_bucket(os.path.join(args.manifest, b, "tarred_audio_manifest.json"), data, b)
                else:
                    sub_buckets = os.listdir(os.path.join(args.manifest, b))
                    for sb in sub_buckets:
                        data = load_bucket(os.path.join(args.manifest, b, sb, "tarred_audio_manifest.json"), data, f"{b}_{sb}")
            buckets = True
        else:
            data = list()
            with open(args.manifest, "r", encoding="utf-8") as f:
                for line in tqdm(f):
                    json_data = json.loads(line)
                    data.append(json_data)
                    
        return data, buckets

    data, buckets = load_data()
    
    plot_folder = os.path.join(args.output_dir, plot_folder)
    os.makedirs(plot_folder, exist_ok=True)
    
    data_per_dataset, hours = process_data(data, "name")
    hours_per_dataset(hours, plot_folder)
    hours_per_dataset(hours, plot_folder, log=True)
    hours_per_dataset_pie(hours, plot_folder)

    durations = list()
    for i in tqdm(data, desc="Computing durations"):
        durations.append(i["duration"])
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
    
    if buckets:
        data_per_bucket, hours = process_data(data, "bucket")
        hours_per_dataset(hours, plot_folder, "hours_per_bucket")
        hours_per_dataset(hours, plot_folder, "hours_per_bucket",log=True)
        hours_per_dataset_pie(hours, plot_folder, "hours_pie_per_bucket")
    
    visualise_upsampling = False
    if visualise_upsampling:
        del data, data_per_dataset, data_per_bucket, hours
        data, buckets = load_data()
        data = apply_upsampling(data)
        
        plot_folder = os.path.join(args.output_dir, plot_folder)
        os.makedirs(plot_folder, exist_ok=True)
        
        data_per_dataset, hours = process_data(data, "name")
        hours_per_dataset(hours, plot_folder, "hours_per_dataset_upsampled")
        hours_per_dataset(hours, plot_folder, log=True, plot_name="hours_per_dataset_upsampled")
        hours_per_dataset_pie(hours, plot_folder, "hours_pie_upsampled", plot_title="after upsampling")

        # make a plot with distribution of durations globally
        durations = list()
        for i in tqdm(data, desc="Computing durations"):
            durations.append(i["duration"])
        plt.figure(figsize=(10, 5))
        plt.hist(durations, bins=100, log=True)
        plt.xlabel("Duration (s)")
        plt.ylabel("Number of samples (log)")
        plt.title("Distribution of durations for training after upsampling")
        plt.savefig(os.path.join(plot_folder, "durations_hist_log_upsampled.png"), bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10, 5))
        plt.hist(durations, bins=100, log=False)
        plt.xlabel("Duration (s)")
        plt.ylabel("Number of samples")
        plt.title("Distribution of durations for training after upsampling")
        plt.savefig(os.path.join(plot_folder, "durations_hist_upsampled.png"), bbox_inches='tight')
        plt.close()
        
        if buckets:
            data_per_bucket, hours = process_data(data, "bucket")
            hours_per_dataset(hours, plot_folder, "hours_per_bucket_upsampled")
            hours_per_dataset(hours, plot_folder, "hours_per_bucket_upsampled",log=True)
            hours_per_dataset_pie(hours, plot_folder, "hours_pie_per_bucket_upsampled", plot_title="after upsampling")