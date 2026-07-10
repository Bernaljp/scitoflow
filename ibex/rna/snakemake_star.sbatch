#!/bin/bash
#SBATCH --job-name=scito_rna_star
#SBATCH --partition=batch
#SBATCH -A pi-gomezcd
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --output=./out/star_%j.out
#SBATCH --error=./err/star_%j.err
# Optional STARsolo Velocyto cross-check. Submit from ibex/rna/:  sbatch snakemake_star.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p out err
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate scitoflow-align
snakemake -s Snakefile_star -j "${SLURM_CPUS_PER_TASK}" --rerun-incomplete --printshellcmds
