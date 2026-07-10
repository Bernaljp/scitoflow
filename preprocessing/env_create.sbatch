#!/bin/bash
#SBATCH --job-name=scito_env
#SBATCH --partition=batch
#SBATCH -A pi-gomezcd
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=./out/env_%j.out
#SBATCH --error=./err/env_%j.err
# One-time build of the reproducible scitoflow-align conda env (on a compute node,
# not the login node). Submit from ibex/:  sbatch env_create.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p out err
source "$HOME/miniconda3/etc/profile.d/conda.sh"
mamba env create -f environment.yml 2>&1 || mamba env update -f environment.yml
conda activate scitoflow-align
echo "=== ENV READY; tool check ==="
for t in kb kallisto bustools bbduk.sh sinto taggd_demultiplex STAR bwa macs3 prefetch fasterq-dump trim_galore pigz samtools tabix; do
  command -v "$t" >/dev/null && echo "  OK  $t" || echo "  MISSING $t"
done
