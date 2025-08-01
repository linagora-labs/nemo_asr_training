import argparse
from eval import load_manifest, logger, datasets_names
from sklearn.model_selection import train_test_split
from linastt.utils.text import format_text_latin
import json


subsample = {
    "commonvoice": 1000,
    "bref": 1000,
    "epac": 0,
    "ester": 1000,
    "mls": 500,
    "summ-re": 1000,
    "voxpopuli": 1000,
    "africanaccentedfrench_devtest": 0
}

def write_manifest(data, path):
    with open(path, 'w') as f:
        for d in data:
            for row in data[d]:
                row['text'] = format_text_latin(row['text'], lang='fr')
                f.write(json.dumps(row) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Subsample data')
    parser.add_argument('--manifest', help="Input manifest", type=str, default="manifests/test_manifest.jsonl")
    parser.add_argument('--output_manifest', help="Output directory", type=str, default="manifests/test_manifest_subsampled.jsonl")
    args = parser.parse_args()
    
    data = load_manifest(args.manifest, None)

    new_data = dict()
    for d, dataset_data in data.items():
        if d.lower() in subsample:
            # print(dataset_data)
            if subsample[d.lower()]>0:
                keep, _ = train_test_split(dataset_data, train_size=subsample[d.lower()], random_state=42)
                new_data[d] = keep
        else:
            new_data[d] = dataset_data
            
    write_manifest(new_data, args.output_manifest)