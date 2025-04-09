"""
Advanced Machine Learning, 2025, HW4

Author: Andrew H. Fagg (andrewhfagg@gmail.com)
Editor: Enzo B. Durel (enzo.durel@gmail.com)

Semantic labeling of the Chesapeake Bay
"""

#################################################################
#                           Imports                             #
#################################################################

from mesonet_support import get_mesonet_folds, extract_station_timeseries, SinhArcsinh
import tensorflow as tf

# Gpus initialization
gpus = tf.config.experimental.list_physical_devices('GPU')
n_visible_devices = len(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

# Set threading parallelism
import os
cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
if cpus_per_task > 1:
    tf.config.threading.set_intra_op_parallelism_threads(cpus_per_task // 2)
    tf.config.threading.set_inter_op_parallelism_threads(cpus_per_task // 2)

# Keras
import tf_keras as keras
from tf_keras.utils import plot_model
from tf_keras.callbacks import EarlyStopping

# WandB
import wandb

# Other imports
import pickle
import socket
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from job_control import JobIterator
from parser import *
from tools import *
from model import create_inner_model, create_outer_model

#################################################################
#                 Default plotting parameters                   #
#################################################################

FIGURESIZE=(10,6)
FONTSIZE=18

plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
    
#################################################################
#                         Experiment                            #
#################################################################

def execute_exp(args, multi_gpus:int=1):
    '''
    Perform the training and evaluation for a single model
    
    :param args: Argparse arguments
    :param multi_gpus: True if there are more than one GPU
    '''

    #################################
    #        Argument Parser        #
    #################################
    
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])
        
    print(args.exp_index)
    
    # Override arguments if we are using exp_index
    args_str = augment_args(args)

    #################################
    #         Load Datasets         #
    #################################

    if args.verbose >= 3:
        print('Starting data flow')

    train_x, train_y, _, valid_x, valid_y, _, test_x, test_y, _ = \
        get_mesonet_folds(dataset_fname=args.dataset, rotation = args.rotation)

    #################################
    #       Model Configuration     #
    #################################

    # Scale the batch size with the number of GPUs
    if multi_gpus > 1:
        args.batch = args.batch * multi_gpus

    print('Batch size', args.batch)

    if args.verbose >= 3:
        print('Building network')

    input_dim = train_x.shape[1]
    inner_model = create_inner_model(input_dim=input_dim, hidden_layers=args.hidden)
    outer_model = create_outer_model(inner_model,
                                     args.lrate)
            
    # Report model structure if verbosity is turned on
    if args.verbose >= 1:
        print(inner_model.summary())

    # Output file base and pkl file
    fbase = generate_fname(args, args_str)
    fname_out = "%s_results.pkl"%fbase

    # Plot the model
    render_fname = '%s_model_plot.png'%fbase
    if args.render:
        plot_model(inner_model, to_file=render_fname, show_shapes=True, show_layer_names=True)

    # Perform the experiment?
    if args.nogo:
        print("NO GO")
        return

    # Check if output file already exists
    if not args.force and os.path.exists(fname_out):
        # Results file does exist: exit
        print("File %s already exists"%fname_out)
        return

    #################################
    #             WandB             #
    #################################
    
    run = wandb.init(project=args.project, name='%s_R%d'%(args.label,args.rotation), notes=fbase, config=vars(args))

    # Log hostname
    wandb.log({'hostname': socket.gethostname()})

    # Log model design image
    if args.render:
        wandb.log({'model architecture': wandb.Image(render_fname)})
            
    #################################
    #            Callbacks          #
    #################################
    
    cbs = []
    early_stopping_cb = EarlyStopping(patience=args.patience,
                                                      restore_best_weights=True,
                                                      min_delta=args.min_delta,
                                                      monitor=args.monitor)
    cbs.append(early_stopping_cb)

    # Weights and Biases
    wandb_metrics_cb = wandb.keras.WandbCallback()
    cbs.append(wandb_metrics_cb)

    if args.verbose >= 3:
        print('Fitting model')

    #################################
    #              Learn            #
    #################################
        
    #  steps_per_epoch: how many batches from the training set do we use for training in one epoch?
    #          Note that if you use this, then you must repeat the training set
    #  validation_steps=None means that ALL validation samples will be used
    history = outer_model.fit(train_x,
                              train_y,
                              validation_data=(valid_x, valid_y),
                              epochs=args.epochs,
                              batch_size=args.batch,
                              steps_per_epoch=args.steps_per_epoch,
                              verbose=args.verbose>=2,
                              validation_steps=None,
                              callbacks=cbs)

    #################################
    #            Results            #
    #################################

    dist = outer_model.predict(test_x) # Predict the distribution

    # Extract stats
    pred_mean = dist.mean().numpy().flatten()
    pred_std = dist.stddev().numpy().flatten()
    pred_skew = dist.skewness().numpy().flatten()
    pred_tail = dist.tailweight().numpy().flatten()
    pred_p10 = dist.quantile(0.10).numpy().flatten()
    pred_p25 = dist.quantile(0.25).numpy().flatten()
    pred_p75 = dist.quantile(0.75).numpy().flatten()
    pred_p90 = dist.quantile(0.90).numpy().flatten()
    pred_median = dist.quantile(0.5).numpy().flatten()
    y_true = test_y.flatten()
    mad_mean = np.mean(np.abs(y_true - pred_mean))
    mad_median = np.mean(np.abs(y_true - pred_median))

    wandb.log({
        "MAD Mean": mad_mean,
        "MAD Median": mad_median,
        "Final Training Loss": history.history["loss"][-1],
        "Final Validation Loss": history.history["val_loss"][-1]
    })
    
    # Generate results data
    results = {
        'rotation': args.rotation,
        'args': args,
        'history': history.history,
        'y_true': y_true,
        'pred_mean': pred_mean,
        'pred_median': pred_median,
        'pred_std': pred_std,
        'pred_skew': pred_skew,
        'pred_tail': pred_tail,
        'percentile_10': pred_p10,
        'percentile_25': pred_p25,
        'percentile_75': pred_p75,
        'percentile_90': pred_p90,
        'mad_mean': mad_mean,
        'mad_median': mad_median
    }

    # Save results
    fbase = generate_fname(args, args_str)
    results['fname_base'] = fbase
    with open("%s_results.pkl"%(fbase), "wb") as fp:
        pickle.dump(results, fp)
    
    # Save model
    if args.save_model:
        inner_model.save("%s_model.keras"%(fbase))

    wandb.finish()

    return inner_model


def check_completeness(args):
    '''
    Check the completeness of a Cartesian product run.

    All other args should be the same as if you executed your batch, however, the '--check' flag has been set

    Prints a report of the missing runs, including both the exp_index and the name of the missing results file

    :param args: ArgumentParser

    '''
    
    # Get the corresponding hyperparameters
    p = exp_type_to_hyperparameters(args)

    # Create the iterator
    ji = JobIterator(p)

    print("Total jobs: %d"%ji.get_njobs())

    print("MISSING RUNS:")

    indices = []
    # Iterate over all possible jobs
    for i in range(ji.get_njobs()):
        params_str = ji.set_attributes_by_index(i, args)
        # Compute output file name base
        fbase = generate_fname(args, params_str)
    
        # Output pickle file name
        fname_out = "%s_results.pkl"%(fbase)

        if not os.path.exists(fname_out):
            # Results file does not exist: report it
            print("%3d\t%s"%(i, fname_out))
            indices.append(i)

    # Give the list of indices that can be inserted into the --array line of the batch file
    print("Missing indices (%d): %s"%(len(indices),','.join(str(x) for x in indices)))

    
#################################################################
#                            Main                               #
#################################################################


if __name__ == "__main__":
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    if args.verbose >= 3:
        print('Arguments parsed')

    # Turn off GPU?
    if not args.gpu or "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        tf.config.set_visible_devices([], 'GPU')
        print('NO VISIBLE DEVICES!!!!')

    if args.check:
        # Just check to see if all experiments have been executed
        check_completeness(args)
    else:
        # Execute the experiment

        # Do the work
        execute_exp(args, multi_gpus=n_visible_devices)
