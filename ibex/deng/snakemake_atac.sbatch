#!/bin/bash
#SBATCH --job-name=deng_atac
#SBATCH --partition=batch
#SBATCH -A pi-gomezcd
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --output=./out/deng_atac_%j.out
#SBATCH --error=./err/deng_atac_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"; mkdir -p out err
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate scitoflow-align
snakemake -s Snakefile_atac --configfile config_atac.yaml -j "${SLURM_CPUS_PER_TASK}" \
    --resources mem_mb=150000 --rerun-incomplete --keep-going --printshellcmds
