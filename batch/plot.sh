#!/bin/bash
#
#SBATCH --partition=debug_5min
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=1G
#SBATCH --output=results/plot/%j_stdout.txt
#SBATCH --error=results/plot/%j_stderr.txt
#SBATCH --time=00:10:00
#SBATCH --job-name=plot
#SBATCH --mail-user=Enzo.B.Durel-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504305/hw5

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh
conda activate dnn

CODE_DIR=code
CONFIG_DIR=configs

## SHALLOW
python ${CODE_DIR}/plot.py \
       @${CONFIG_DIR}/exp.txt @${CONFIG_DIR}/net.txt \
       --results_path "./results/exp/" \
