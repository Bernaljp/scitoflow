#!/bin/bash
#SBATCH --job-name=misar_atac
#SBATCH --partition=batch
#SBATCH -A pi-gomezcd
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --output=./out/misar_atac_%j.out
#SBATCH --error=./err/misar_atac_%j.err
# MISAR-seq ATAC (8 sections). Builds the mm10 bwa index once, then aligns. Submit from ibex/misar/.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p out err
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate scitoflow-align
# cap total memory across concurrent jobs (mem_mb resource) so bwa/samtools-sort don't OOM
snakemake -s Snakefile_atac --configfile config_atac.yaml -j "${SLURM_CPUS_PER_TASK}" \
    --resources mem_mb=150000 --rerun-incomplete --keep-going --printshellcmds
