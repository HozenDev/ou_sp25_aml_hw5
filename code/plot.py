import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mesonet_support import extract_station_timeseries

plt.style.use('seaborn-v0_8-muted')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 14

def load_all_results(result_dir="results/exp"):
    result_files = sorted(glob.glob(os.path.join(result_dir, "rotation_*_results.pkl")))
    return [pickle.load(open(f, "rb")) for f in result_files]

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
def plot_timeseries_example(res, station_index=0, nstations=17):
    _, y_true = extract_station_timeseries(
        ins=np.zeros((len(res["y_true"]), 1)),  # dummy input, not used
        outs=res["y_true"].reshape(-1, 1),
        nstations=nstations,
        station_index=station_index
    )

    _, p10 = extract_station_timeseries(None, res['percentile_10'].reshape(-1, 1), nstations, station_index)
    _, p25 = extract_station_timeseries(None, res['percentile_25'].reshape(-1, 1), nstations, station_index)
    _, p75 = extract_station_timeseries(None, res['percentile_75'].reshape(-1, 1), nstations, station_index)
    _, p90 = extract_station_timeseries(None, res['percentile_90'].reshape(-1, 1), nstations, station_index)
    _, pred_mean = extract_station_timeseries(None, res['pred_mean'].reshape(-1, 1), nstations, station_index)

    # Flatten all outputs
    y_true = y_true.flatten()
    pred_mean = pred_mean.flatten()
    p10 = p10.flatten()
    p25 = p25.flatten()
    p75 = p75.flatten()
    p90 = p90.flatten()

    plt.figure()
    plt.plot(y_true, label="True RAIN", color='black')
    plt.plot(pred_mean, label="Predicted Mean", linestyle='--')
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
    pred_mean = np.concatenate([r['pred_mean'] for r in all_results])
    std = np.concatenate([r['pred_std'] for r in all_results])
    skew = np.concatenate([r['pred_skew'] for r in all_results])
    tail = np.concatenate([r['pred_tail'] for r in all_results])

    def scatter_plot(x, y, xlabel, ylabel, title, filename):
        plt.figure()
        plt.scatter(x, y, alpha=0.3, edgecolors='k', s=20)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.savefig(filename)

    scatter_plot(y_true, pred_mean, "Observed RAIN", "Predicted Mean",
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
    all_results = load_all_results()

    print("Generating Figure 1...")
    plot_loss_curves(all_results)

    print("Generating Figure 2...")
    plot_timeseries_example(all_results[0], station_index=0, nstations=17)

    print("Generating Figure 3...")
    plot_param_scatter(all_results)

    print("Generating Figure 4...")
    plot_mad_bars(all_results)

    print("✅ All figures saved to /figures/")
