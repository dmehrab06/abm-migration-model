#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128000
#SBATCH -t 05:00:00
#SBATCH -p bii
#SBATCH -A nssac_covid19

source activate biocomplexity

python ukr_mim_mdm_pipeline_parameter_sample.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14}


