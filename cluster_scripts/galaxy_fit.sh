#!/bin/bash

#SBATCH --job-name fit_spec   ## name that will show up in the queue
#SBATCH --output fit_spec-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH -n 5
#SBATCH --time 24:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --mail-user mkarmen1@jhu.edu  ## your email address
#SBATCH --mail-type BEGIN  ## slurm will email you when your job starts
#SBATCH --mail-type END  ## slurm will email you when your job ends
#SBATCH --mail-type FAIL  ## slurm will email you when your job fails


export DIR=/home/mkarmen1

# Load any necessary modules or dependencies
# module load anaconda

# Activate the virtual environment
module unload python
module load anaconda
conda activate /home/mkarmen1/mkarmen

srun -n1 --exclusive python /home/mkarmen1/tde_forecasting/scripts/fit_spec.py --name AT2018zr
srun -n1 --exclusive python /home/mkarmen1/tde_forecasting/scripts/fit_spec.py --name AT2018bsi 
srun -n1 --exclusive python /home/mkarmen1/tde_forecasting/scripts/fit_spec.py --name AT2018hco
srun -n1 --exclusive python /home/mkarmen1/tde_forecasting/scripts/fit_spec.py --name AT2018iih
srun -n1 --exclusive python /home/mkarmen1/tde_forecasting/scripts/fit_spec.py --name AT2018hyz
wait
