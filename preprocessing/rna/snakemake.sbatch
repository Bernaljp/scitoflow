#!/bin/bash
#SBATCH --job-name=scito_rna_kb
#SBATCH --partition=batch
#SBATCH -A pi-gomezcd
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=./out/kb_%j.out
#SBATCH --error=./err/kb_%j.err
# RNA spliced/unspliced (kb nac). Submit from ibex/rna/:  sbatch snakemake.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p out err

# Reproducible tools (kb-python, bustools, bbmap, sra-tools, pigz) from conda.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate scitoflow-align

snakemake -j "${SLURM_CPUS_PER_TASK}" --rerun-incomplete --printshellcmds
