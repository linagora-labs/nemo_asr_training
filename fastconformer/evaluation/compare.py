import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import json

logging.getLogger('nemo_logger').setLevel(logging.ERROR)

def plot_wer_table(wer_means, wer_stds=None, output_filename='wer_table.png', show=True, y_label="WER (%)", best="lowest", color_lims=(0,50)):
    wer_means = wer_means.copy()
    n_rows, n_cols = wer_means.shape

    if wer_stds is not None:
        wer_stds = pd.DataFrame(wer_stds, index=wer_means.index, columns=wer_means.columns)

    fig, axis = plt.subplots(figsize=(1.8 * n_cols if n_cols>2 else 3*n_cols, 1.2 * n_rows))
    if best=="highest":
        min_indices = wer_means.idxmax()
        color_map = mcolors.LinearSegmentedColormap.from_list('purple_red_yellow_green', ['purple', 'red', 'yellow', 'green'])        
    else:
        min_indices = wer_means.idxmin()
        color_map = mcolors.LinearSegmentedColormap.from_list('green_red_purple', ['green', 'yellow', 'red', 'purple'])
    normalizer = mcolors.Normalize(vmin=color_lims[0], vmax=color_lims[1])

    for i in range(n_rows):
        for j in range(n_cols):
            val = wer_means.iat[i, j]

            if not pd.isna(val):
                color = color_map(normalizer(val))
                rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='gray')
                axis.add_patch(rect)

                fontweight = 'bold' if wer_means.index[i] == min_indices[j] else 'normal'
                axis.text(j + 0.5, i + 0.4 if wer_stds is not None else i + 0.5, f"{val:.2f}", ha='center', va='center',
                          fontsize=12, weight=fontweight, color='black')

                if wer_stds is not None:
                    std_val = wer_stds.iat[i, j]
                    if not pd.isna(std_val):
                        axis.text(j + 0.5, i + 0.7, f"±{std_val:.2f}", ha='center', va='center',
                                  fontsize=9, style='italic', color='black')

    axis.set_xlim(0, n_cols)
    axis.set_ylim(0, n_rows)
    axis.set_xticks(np.arange(n_cols) + 0.5)
    axis.set_yticks(np.arange(n_rows) + 0.5)

    axis.set_yticklabels(wer_means.index, fontsize=12, color='black')
    axis.set_xticklabels(wer_means.columns, rotation=0, ha='center', fontsize=12, color='black')

    axis.invert_yaxis()
    axis.xaxis.tick_top()
    axis.tick_params(length=0)

    for spine in axis.spines.values():
        spine.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=color_map, norm=normalizer)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axis, orientation='vertical', shrink=0.6, pad=0.01)
    cbar.set_label(y_label, fontsize=12)

    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename)
    if show:
        plt.show()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor checkpoints during training')
    parser.add_argument('--models', nargs="+", default=["/mnt/d/linagora/finetuning_benchmark/output"])
    parser.add_argument('--casepunc', action='store_true', default=False)
    parser.add_argument('--plot_folder', default="plots")
    args = parser.parse_args()
    data = dict()
    if len(args.models) == 1:
        args.models = [
            p for p in Path(args.models[0]).rglob("*")
            if p.is_dir()
            and not any(child.is_dir() for child in p.iterdir())
            and not any(part.startswith('.') for part in p.parts)
            and not "plots" in str(p)
        ]
    for model in args.models:
        model_path = Path(model)
        model_name = model_path.parents[2].name if model_path.name.startswith("training-epoch") else model_path.name
        data[model_name] = {}
        wer_files = list(model_path.glob("wer_nocasepunc_*.json" if not args.casepunc else "wer_casepunc_*.json"))
        if not wer_files:
            wer_files = list(model_path.glob("wer_*.json"))
        wer_files = [f for f in wer_files if "alignment" not in f.name.lower()]
        for file_path in wer_files:
            dataset_name = file_path.stem.replace("nocasepunc_", "").replace("wer_", "").replace("casepunc_", "")
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    wer_value = json.load(f)
                except json.JSONDecodeError:
                    wer_value = None  # handle invalid JSON if needed

            data[model_name][dataset_name] = wer_value
    
    plot_folder = args.plot_folder
    
    all_datasets_per_model = {model: set(datasets.keys()) for model, datasets in data.items()}
    print(all_datasets_per_model)
    common_datasets = set.intersection(*all_datasets_per_model.values())
    all_datasets = set.union(*all_datasets_per_model.values())
    dropped_datasets = all_datasets - common_datasets
    
    print(f"Dropped datasets: {dropped_datasets}")
    
    filtered_data = {
        model: {ds: vals for ds, vals in datasets.items() if ds in common_datasets}
        for model, datasets in data.items()
    }
    wer_df = pd.DataFrame({
        model: {dataset: vals["wer"] for dataset, vals in datasets.items()}
        for model, datasets in filtered_data.items()
    }).T  # models as rows, datasets as columns
    
    print(wer_df)
    plot_wer_table(wer_df)