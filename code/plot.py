import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from mesonet_support import extract_station_timeseries, get_mesonet_folds
import keras

# Tensorflow
import tensorflow_probability as tfp
tfd = tfp.distributions

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

def plot_figure2(res, dataset_path, rotation=0, station_indices=[0, 5, 10, 15], window_size=100, output_path='figure_2.png'):
    """
    Plot Figure 2
    """
    _, _, _, _, _, _, test_x, test_y, test_nstations = get_mesonet_folds(
        dataset_fname=dataset_path, rotation=rotation)

    # If pred_mean is scalar, rebuild predictions
    mu = res['mu']
    std = res['std']
    skew = res['skew']
    tail = res['tail']

    dist = tfd.SinhArcsinh(loc=mu, scale=std, skewness=skew, tailweight=tail)
    samples = dist.sample(1000).numpy()

    print("samples shape:", samples.shape)

    res['pred_mean'] = np.mean(samples, axis=0)
    res['percentile_10'] = np.percentile(samples, 10, axis=0)
    res['percentile_25'] = np.percentile(samples, 25, axis=0)
    res['percentile_75'] = np.percentile(samples, 75, axis=0)
    res['percentile_90'] = np.percentile(samples, 90, axis=0)
    
    # Unpack predicted values
    pred_mean = res['mu']
    p10 = res['percentile_10']
    p25 = res['percentile_25']
    p75 = res['percentile_75']
    p90 = res['percentile_90']

    # Create plot
    n_stations = len(station_indices)
    fig, axs = plt.subplots(n_stations, 1, figsize=(12, 3.5 * n_stations), sharex=False)

    print("test_y shape:", test_y.shape)
    print("test_x shape:", test_x.shape)
    print("pred_mean shape:", pred_mean.shape)
    print("p10 shape:", p10.shape)
    
    for i, station_idx in enumerate(station_indices):
        # Extract station time series
        _, y_station = extract_station_timeseries(test_x, test_y, test_nstations, station_idx)
        
        pred_station_mean = pred_mean[station_idx::test_nstations]
        pred_p10 = p10[station_idx::test_nstations]
        pred_p25 = p25[station_idx::test_nstations]
        pred_p75 = p75[station_idx::test_nstations]
        pred_p90 = p90[station_idx::test_nstations]

        # Use a fixed window from middle of station's time series
        total_length = len(y_station)
        mid = total_length // 2
        start = max(0, mid - window_size // 2)
        end = min(total_length, start + window_size)

        t = np.arange(end - start)
        y = y_station[start:end].flatten()
        mean = pred_station_mean[start:end]
        p10_vals = pred_p10[start:end]
        p25_vals = pred_p25[start:end]
        p75_vals = pred_p75[start:end]
        p90_vals = pred_p90[start:end]

        ax = axs[i] if n_stations > 1 else axs
        ax.plot(t, y, label='Observed', color='black', linewidth=2)
        ax.plot(t, mean, label='Predicted Mean', linestyle='--', color='blue')
        ax.fill_between(t, p25_vals, p75_vals, alpha=0.4, color='blue', label='25–75%')
        ax.fill_between(t, p10_vals, p90_vals, alpha=0.2, color='blue', label='10–90%')

        ax.set_ylabel("Rainfall (mm)")
        ax.set_title(f"Station {station_idx}")
        ax.grid(True)
        if i == 0:
            ax.legend()

    axs[-1].set_xlabel("Time (days)")
    fig.suptitle("Figure 2: Predicted Rainfall Distribution by Station", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path)

# ------------------------------
# Figure 3: Scatter plots
# ------------------------------
def scatter_plot(x, y, xlabel, ylabel, title, filename):
    plt.figure()
    plt.scatter(x, y, alpha=0.3, edgecolors='k', s=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)

def plot_param_scatter(all_results):
    y_true = np.concatenate([r['y_true'] for r in all_results])

    print("mu shape:", all_results[0]['mu'].shape)
    print("y_true shape:", all_results[0]['y_true'].shape)
    
    mu = np.concatenate([r['mu'] for r in all_results])
    std = np.concatenate([r['std'] for r in all_results])
    skew = np.concatenate([r['skew'] for r in all_results])
    tail = np.concatenate([r['tail'] for r in all_results])
    
    print(mu.shape, std.shape, skew.shape, tail.shape, y_true.shape)

    scatter_plot(y_true, mu, "Observed RAIN", "Predicted Mean", "Figure 3a: Predicted Mean vs. Observed", "figures/figure_3a.png")
    scatter_plot(y_true, std, "Observed RAIN", "Predicted Std Dev", "Figure 3b: Std Dev vs. Observed", "figures/figure_3b.png")
    scatter_plot(y_true, skew, "Observed RAIN", "Predicted Skewness", "Figure 3c: Skewness vs. Observed", "figures/figure_3c.png")
    scatter_plot(y_true, tail, "Observed RAIN", "Predicted Tailweight", "Figure 3d: Tailweight vs. Observed", "figures/figure_3d.png")

# Figure 4
def plot_mad_bars(results):
    rotations = [r['rotation'] for r in results]
    mad_mean = [r['mad_mean'] for r in results]
    mad_median = [r['mad_median'] for r in results]
    mad_zero = [r['mad_zero'] for r in results]

    x = np.arange(len(rotations))
    bar_width = 0.10
    width = 0.35
    
    plt.figure()
    plt.bar(x - width/2, mad_median, bar_width, label="MAD Median")
    plt.bar(x, mad_mean, bar_width, label="MAD Mean")
    plt.bar(x + width/2, mad_zero, bar_width, label="MAD Zero")
    plt.xticks(x, [f"R{r}" for r in range(len(rotations))])
    plt.ylabel("MAD")
    plt.title("Figure 4: MAD Across Rotations")
    plt.legend()
    plt.savefig("figures/figure_4.png")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)
    
    all_results = load_results(["./models/exp_v/"])

    print("Plotting results...")
    plot_figure2(all_results[0], dataset_path=args.dataset, rotation=1, station_indices=[0, 3, 5, 7], window_size=120, output_path='figures/figure_2.png')
    # plot_param_scatter(all_results)
    # plot_mad_bars(all_results)
    print("Done")
