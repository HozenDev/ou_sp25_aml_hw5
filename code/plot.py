import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mesonet_support import extract_station_timeseries, get_mesonet_folds
import keras

from parser import check_args, create_parser

plt.style.use('seaborn-v0_8-muted')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 14

def load_trained_model(model_dir, substring_name):
    """
    Load a trained models
    """
    model_files = [f for f in os.listdir(model_dir) if substring_name in f and f.endswith(".keras")]

    if not model_files:
        raise ValueError(f"No model found in {model_dir} matching {substring_name}")

    model_path = os.path.join(model_dir, model_files[0])
    model = keras.models.load_model(model_path)

    return model

def load_results_iter(results_dir):
    """
    Generator to load model results from a directory.
    Reduce memory usage compared to load results method.
    """
    files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".pkl")]
    
    for filename in files:
        with open(filename, "rb") as fp:
            data = pickle.load(fp)
            yield data
            

def load_results(results_dir):
    """
    Load model results from a directory
    """
    results = []
    files = []
    for r_dir in results_dir:
        files.extend([os.path.join(r_dir, f) for f in os.listdir(r_dir) if f.endswith(".pkl")])

    for filename in files:
        with open(filename, "rb") as fp:
            data = pickle.load(fp)
            results.append(data)

    return results

# ------------------------------
# Figure 1: Loss curves
# ------------------------------
def plot_loss_curves(results):
    os.makedirs("figures", exist_ok=True)

    plt.figure()
    for res in results:
        plt.plot(res['history']['loss'], label=f"Rotation {res['rotation']}")
    plt.xlabel("Epoch")
    plt.ylabel("Training NLL")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig("figures/figure1a_training_loss.png")

    plt.figure()
    for res in results:
        plt.plot(res['history']['val_loss'], label=f"Rotation {res['rotation']}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation NLL")
    plt.title("Validation Loss")
    plt.legend()
    plt.savefig("figures/figure1b_validation_loss.png")

# ------------------------------
# Figure 2: Time-series for 1 station using provided function
# ------------------------------
def plot_timeseries_example(dataset_path, rotation, res, station_index=0, nstations=17):

    _, _, _, _, _, _, test_x, _, _ = \
        get_mesonet_folds(dataset_fname=dataset_path, rotation=rotation)

    _, y_true = extract_station_timeseries(test_x, res["y_true"].reshape(-1, 1), nstations, station_index)
    _, p10 = extract_station_timeseries(test_x, res['percentile_10'].reshape(-1, 1), nstations, station_index)
    _, p25 = extract_station_timeseries(test_x, res['percentile_25'].reshape(-1, 1), nstations, station_index)
    _, p75 = extract_station_timeseries(test_x, res['percentile_75'].reshape(-1, 1), nstations, station_index)
    _, p90 = extract_station_timeseries(test_x, res['percentile_90'].reshape(-1, 1), nstations, station_index)
    _, mu = extract_station_timeseries(test_x, res['mu'].reshape(-1, 1), nstations, station_index)

    # Flatten all outputs
    x = test_x.flatten()
    y_true = y_true.flatten()
    mu = mu.flatten()
    p10 = p10.flatten()
    p25 = p25.flatten()
    p75 = p75.flatten()
    p90 = p90.flatten()

    plt.figure()
    plt.plot(x, y_true, label="True RAIN", color='black', alpha=0.5, style='.')
    plt.plot(x, mu, label="Predicted Mean", linestyle='--')
    plt.fill_between(range(len(p10)), p10, p90, alpha=0.2, label="10–90%")
    plt.fill_between(range(len(p25)), p25, p75, alpha=0.4, label="25–75%")
    plt.xlabel("Day Index")
    plt.ylabel("Precipitation")
    plt.title("Figure 2: Time-Series for One Station")
    plt.legend()
    plt.savefig("figures/figure2_timeseries.png")

# ------------------------------
# Figure 3: Scatter plots
# ------------------------------
def plot_param_scatter(all_results):
    y_true = np.concatenate([r['y_true'] for r in all_results])
    mu = [r['mu'] for r in all_results]
    std = np.concatenate([r['std'] for r in all_results])
    skew = np.concatenate([r['skew'] for r in all_results])
    tail = np.concatenate([r['tail'] for r in all_results])

    def scatter_plot(x, y, xlabel, ylabel, title, filename):
        plt.figure()
        plt.scatter(x, y, alpha=0.3, edgecolors='k', s=20)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.savefig(filename)

    scatter_plot(y_true, mu, "Observed RAIN", "Predicted Mean",
                 "Figure 3a: Predicted Mean vs. Observed", "figures/figure3a_mean_vs_observed.png")
    scatter_plot(y_true, std, "Observed RAIN", "Predicted Std Dev",
                 "Figure 3b: Std Dev vs. Observed", "figures/figure3b_std_vs_observed.png")
    scatter_plot(y_true, skew, "Observed RAIN", "Predicted Skewness",
                 "Figure 3c: Skewness vs. Observed", "figures/figure3c_skew_vs_observed.png")
    scatter_plot(y_true, tail, "Observed RAIN", "Predicted Tailweight",
                 "Figure 3d: Tailweight vs. Observed", "figures/figure3d_tail_vs_observed.png")

# ------------------------------
# Figure 4: MADs
# ------------------------------
def plot_mad_bars(results):
    rotations = [r['rotation'] for r in results]
    mad_mean = [r['mad_mean'] for r in results]
    mad_median = [r['mad_median'] for r in results]

    x = np.arange(len(rotations))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, mad_mean, width, label="Mean Prediction")
    plt.bar(x + width/2, mad_median, width, label="Median Prediction")
    plt.xticks(x, [f"R{r}" for r in rotations])
    plt.ylabel("MAD")
    plt.title("Figure 4: MAD Across Rotations")
    plt.legend()
    plt.savefig("figures/figure4_mad.png")

# ------------------------------
# Run all
# ------------------------------
if __name__ == "__main__":
    # Parse command-line arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)
    
    all_results = load_results(["./results/exp/"])

    print("Generating Figure 1...")
    plot_loss_curves(all_results)

    print("Generating Figure 2...")
    plot_timeseries_example(args.dataset, 0, all_results[0], station_index=0, nstations=17)

    print("Generating Figure 3...")
    plot_param_scatter(all_results)

    print("Generating Figure 4...")
    plot_mad_bars(all_results)
