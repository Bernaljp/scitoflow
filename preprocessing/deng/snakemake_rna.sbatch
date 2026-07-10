#!/bin/bash
#SBATCH --job-name=deng_rna
#SBATCH --partition=batch
#SBATCH -A pi-gomezcd
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --output=./out/deng_rna_%j.out
#SBATCH --error=./err/deng_rna_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"; mkdir -p out err
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate scitoflow-align
snakemake -s Snakefile_rna --configfile config_rna.yaml -j "${SLURM_CPUS_PER_TASK}" \
    --rerun-incomplete --keep-going --printshellcmds
