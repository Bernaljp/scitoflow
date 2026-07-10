#!/bin/bash
# Sync the ibex/ pipeline to KAUST Ibex scratch and submit / dry-run a pipeline.
# Runs locally; uses the `ilogin` ssh host. Usage:
#   bash ibex/submit.sh env          # create/update the scitoflow-align conda env
#   bash ibex/submit.sh dryrun-rna   # snakemake -n on the login node (fast, safe)
#   bash ibex/submit.sh rna|atac|star# sbatch the pipeline (batch partition, pi-gomezcd)
set -euo pipefail
REMOTE=ilogin
DEST=/ibex/scratch/projects/c2012/bernaljp/scitoflow_pipeline
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "[submit] rsync ibex/ -> $REMOTE:$DEST"
rsync -avh --exclude 'out/' --exclude 'err/' --exclude '.snakemake/' "$HERE/" "$REMOTE:$DEST/"

act='source ~/miniconda3/etc/profile.d/conda.sh; conda activate scitoflow-align'
case "${1:-help}" in
  env)         ssh $REMOTE "source ~/miniconda3/etc/profile.d/conda.sh; mamba env create -f $DEST/environment.yml 2>/dev/null || mamba env update -f $DEST/environment.yml" ;;
  dryrun-rna)  ssh $REMOTE "$act; cd $DEST/rna  && snakemake -n" ;;
  dryrun-atac) ssh $REMOTE "$act; cd $DEST/atac && snakemake -n" ;;
  rna)         ssh $REMOTE "cd $DEST/rna  && sbatch snakemake.sbatch" ;;
  atac)        ssh $REMOTE "cd $DEST/atac && sbatch snakemake.sbatch" ;;
  star)        ssh $REMOTE "cd $DEST/rna  && sbatch snakemake_star.sbatch" ;;
  *) echo "usage: bash ibex/submit.sh [env|dryrun-rna|dryrun-atac|rna|atac|star]" ;;
esac
