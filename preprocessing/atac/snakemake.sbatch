#!/bin/bash
#SBATCH --job-name=scito_atac
#SBATCH --partition=batch
#SBATCH -A pi-gomezcd
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=./out/atac_%j.out
#SBATCH --error=./err/atac_%j.err
# ATAC raw->fragments->peaks. Submit from ibex/atac/:  sbatch snakemake.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p out err
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate scitoflow-align
snakemake -j "${SLURM_CPUS_PER_TASK}" --rerun-incomplete --printshellcmds
