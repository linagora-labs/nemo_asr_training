import logging
import os
import numpy as np
from matplotlib import pyplot as plt
from ssak.utils.wer import plot_wer
from utils import logger, CER_WHISPER_PER_DATASET, WER_WHISPER_PER_DATASET, COLORS

logging.getLogger('nemo_logger').setLevel(logging.ERROR)

def make_data_plots(data, output_dir):
    # make a plot with number of hour in each dataset
    durations = dict()
    for name, dataset in data.items():
        durations[name] = sum([r["duration"] for r in dataset]) / 3600
    os.makedirs(output_dir, exist_ok=True)
    plt.bar(durations.keys(), durations.values())
    # add values on top of the bars
    for i, v in enumerate(durations.values()):
        plt.text(i, v + 0.05, str(round(v, 1))+"h", ha='center')
    plt.title("Hours in each test dataset")
    plt.xticks(rotation=40, ha='right')
    plt.savefig(os.path.join(output_dir, "hours_subset.png"), bbox_inches='tight')
    plt.close()
    
def make_curve(wers, output_dir, lims=(0, 100), x_scale='linear', datasets_to_plot=None, plot_name_suffix="", cer=False, nocasepunc=False):
    if plot_name_suffix:
        plot_name_suffix = "_" + plot_name_suffix
    os.makedirs(output_dir, exist_ok=True)
    data_points = dict()
    datasets = list(list(wers.values())[0].keys())
    if datasets_to_plot is not None:
        for i in range(len(datasets)-1, -1, -1):
            if datasets[i].lower() not in datasets_to_plot:
                datasets.pop(i)
    for checkpoint, wer in wers.items():
        # get step number from checkpoint name
        checkpoint = os.path.basename(os.path.splitext(checkpoint)[0])
        step = None
        for part in checkpoint.split("-"):
            if part.startswith("step="):
                step = int(part.replace("step=", ""))/1000
                break
        if step is None:
            step = 0
            # continue
        data_points[step] = wer
    data_points = {k: data_points[k] for k in sorted(data_points.keys())}
    x_max = list(data_points.keys())[-1]
    whisper_values = CER_WHISPER_PER_DATASET if cer else WER_WHISPER_PER_DATASET
    for i, dataset in enumerate(datasets):
        plt.plot(data_points.keys(), [v[dataset]['wer'] for v in data_points.values()], label=dataset, color=COLORS[i])
        last_step_value = data_points[max(data_points.keys())]
        if last_step_value[dataset]['wer'] < lims[1] and x_scale != "log":
            # plt.text(max(data_points.keys())+10, last_step_value[dataset]['wer']+lims[1]*0.01*3, f"{last_step_value[dataset]['wer']:.1f}%", ha='right', va='center', color=COLORS[i])
            plt.text(x_max+x_max/6, last_step_value[dataset]['wer']+lims[1]*0.01*3, f"{last_step_value[dataset]['wer']:.1f}%", ha='right', va='center', color=COLORS[i])
        if dataset in whisper_values:
            plt.axhline(y=whisper_values[dataset], color=COLORS[i], linestyle='--', label=f"Whisper on {dataset}")
            # write value on y axis
            if whisper_values[dataset] < lims[1] and x_scale != "log":
                plt.text(-(x_max/6), whisper_values[dataset], f"{whisper_values[dataset]}%", ha='right', va='center', color=COLORS[i])
    plt.ylim(lims)
    # plt.axvline(x=305, color='k', linestyle='--', label="Retraining") # 305k steps
    plt.xscale(x_scale)
    plt.xlabel("Step (in thousands)")
    plt.ylabel("WER (%)" if not cer else "CER (%)")
    plt.title(f"{'WER' if not cer else 'CER'} evolution during training")
    # make a legend with the dataset names
    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1))
    plt.savefig(os.path.join(output_dir, f"{'wer' if not cer else 'cer'}_{'nocasepunc' if nocasepunc else 'casepunc'}_curve_lim{lims[1]}_x{x_scale}{plot_name_suffix}.png"), bbox_inches='tight')
    plt.close()

def make_wer_plots(wers, output_dir, lims=(0, 100), cer=False, nocasepunc=False):
    os.makedirs(output_dir, exist_ok=True)
    per_checkpoint = dict()
    for checkpoint, wers in wers.items():
        checkpoint = os.path.basename(os.path.splitext(checkpoint)[0])
        sub_dir = os.path.join(output_dir, checkpoint)
        os.makedirs(sub_dir, exist_ok=True)
        plot_wer(wers, title=f"{'WER' if not cer else 'CER'} on {checkpoint}", show=os.path.join(sub_dir, f"{'wer' if not cer else 'cer'}.png"), sort_best=0, label_rotation=45, scale=1, ymax=100)
        plot_wer(wers, title=f"{'WER' if not cer else 'CER'} on {checkpoint}", show=os.path.join(sub_dir, f"{'wer' if not cer else 'cer'}_lim50.png"), sort_best=0, label_rotation=45, scale=1, ymax=50)
        # mean = np.mean([v for v in wers.values()])
        # std = np.std([v for v in wers.values()])
        # median = np.median([v for v in wers.values()])
        per_checkpoint[checkpoint] = list(i['wer'] for i in wers.values())
        plt.close()
    # plot a violin plot
    plt.bar(per_checkpoint.keys(), [np.mean(v) for v in per_checkpoint.values()])
    # plt.violinplot([v for v in per_checkpoint.values()], showmeans=True, showmedians=True)
    # plt.xticks(range(1, len(per_checkpoint)+1), per_checkpoint.keys(), rotation=45)
    plt.ylim(lims)
    plt.xticks(rotation=40, ha='right') 
    plt.title(f"{'WER' if not cer else 'CER'} distribution")
    plt.savefig(os.path.join(output_dir, f"{'wer' if not cer else 'cer'}_{'nocasepunc' if nocasepunc else 'casepunc'}_bars_lim{lims[1]}.png"), bbox_inches='tight')
    plt.close()


def plot_results(results, plot_folder, cer=False, nocasepunc=False):
    make_wer_plots(results, plot_folder, cer=cer, nocasepunc=nocasepunc)
    try:
        make_curve(results, plot_folder, cer=cer, nocasepunc=nocasepunc)
        make_curve(results, plot_folder, (0,50), cer=cer, nocasepunc=nocasepunc)
        make_curve(results, plot_folder, (0,100), 'log', cer=cer, nocasepunc=nocasepunc)
        make_curve(results, plot_folder, (0,50), 'log', cer=cer, nocasepunc=nocasepunc)
        reduced_datasets = ['youtube', 'mls', 'eslo', 'voxpopuli', 'fleurs', 'africanaccentedfrench', 'cfpp2000']
        make_curve(results, plot_folder, datasets_to_plot=reduced_datasets, plot_name_suffix="main-datasets", cer=cer, nocasepunc=nocasepunc)
        make_curve(results, plot_folder, (0,50), datasets_to_plot=reduced_datasets, plot_name_suffix="main-datasets", cer=cer, nocasepunc=nocasepunc)
        make_curve(results, plot_folder, (0,100), 'log', datasets_to_plot=reduced_datasets, plot_name_suffix="main-datasets", cer=cer, nocasepunc=nocasepunc)
        make_curve(results, plot_folder, (0,50), 'log', datasets_to_plot=reduced_datasets, plot_name_suffix="main-datasets", cer=cer, nocasepunc=nocasepunc)
        reduced_datasets = ['bref', 'ester', 'summ-re']
        make_curve(results, plot_folder, datasets_to_plot=reduced_datasets, plot_name_suffix="notseen-datasets", cer=cer, nocasepunc=nocasepunc)
        make_curve(results, plot_folder, (0,50), datasets_to_plot=reduced_datasets, plot_name_suffix="notseen-datasets", cer=cer, nocasepunc=nocasepunc)
        make_curve(results, plot_folder, (0,100), 'log', datasets_to_plot=reduced_datasets, plot_name_suffix="notseen-datasets", cer=cer, nocasepunc=nocasepunc)
        make_curve(results, plot_folder, (0,50), 'log', datasets_to_plot=reduced_datasets, plot_name_suffix="notseen-datasets", cer=cer, nocasepunc=nocasepunc)
    except Exception as e:
        logger.error(f"Error while making curve: {e}")