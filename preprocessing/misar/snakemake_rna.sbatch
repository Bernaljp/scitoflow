#!/bin/bash
#SBATCH --job-name=misar_rna
#SBATCH --partition=batch
#SBATCH -A pi-gomezcd
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=./out/misar_rna_%j.out
#SBATCH --error=./err/misar_rna_%j.err
# MISAR-seq RNA (8 brain sections). Submit from ibex/misar/.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p out err
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate scitoflow-align
snakemake -s Snakefile_rna --configfile config_rna.yaml -j "${SLURM_CPUS_PER_TASK}" --rerun-incomplete --printshellcmds
