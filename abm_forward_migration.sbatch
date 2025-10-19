#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 5:00:00
#SBATCH -p bii
#SBATCH --mem=64G
#SBATCH -A nssac_covid19
#SBATCH --job-name=abm_forwd_tpb
#SBATCH -o ./slurm_outputs/output-%j.out # STDOUT

module load miniforge
source activate migration_env

python abm_forward_migration_tpb.py --config "$1" --region_name "$2" --cpu_core_usage "$SLURM_CPUS_PER_TASK"
