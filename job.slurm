#!/bin/bash
#SBATCH --job-name=exp_real_ind_seed1     # create a short name for your job
#SBATCH --nodes=1                         # node count
#SBATCH --ntasks=1                        # total number of tasks across all nodes
#SBATCH --exclusive		                    # if the GPU has to be used solarly
#SBATCH --partition=small_gpu    
#SBATCH --time=20:00:00                   # total run time limit (HH:MM:SS)
#SBATCH --mail-type=ALL
#SBATCH --mail-user= ehrhardj@hsu-hh.de
#SBATCH --output=slurmjob%j.log

module purge
module load anaconda3/2021.11
module load cuda
eval "$(conda shell.bash hook)"
conda activate sslpp
conda info --envs

python main.py --path=experiments/real/experiments_real_ind_seed_1.json
