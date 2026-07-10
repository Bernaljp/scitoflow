#!/bin/bash
#SBATCH --job-name=scito_brain_rna
#SBATCH --partition=batch
#SBATCH -A pi-gomezcd
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=./out/brain_%j.out
#SBATCH --error=./err/brain_%j.err
# Brain RNA without bbduk (bbduk stalls on this node). Submit from ibex/rna/.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p out err
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate scitoflow-align
snakemake -s Snakefile_nobbduk --configfile config_brain.yaml -j "${SLURM_CPUS_PER_TASK}" --rerun-incomplete --printshellcmds
