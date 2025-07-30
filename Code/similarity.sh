#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=fat
#SBATCH --time=120:00:00
#SBATCH --mem=50000

module load 2021
module load Anaconda3/2021.05
module load GCCcore/10.3.0
source /sw/arch/Centos8/EB_production/2021/software/Anaconda3/2021.05/etc/profile.d/conda.sh

cd  /home/singhp/clustering
conda activate clustering
python similarity_openml.py $SLURM_ARRAY_TASK_ID
