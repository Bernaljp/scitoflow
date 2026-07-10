#!/bin/bash
#SBATCH --job-name=deng_dl
#SBATCH --partition=batch
#SBATCH -A pi-gomezcd
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/ibex/scratch/projects/c2012/bernaljp/scitoflow_pipeline/deng/out/dl_%j.out
#SBATCH --error=/ibex/scratch/projects/c2012/bernaljp/scitoflow_pipeline/deng/err/dl_%j.err
# Deng/Zhang 2023 spatial-ATAC-RNA-seq (GSE205055) mouse ME13 embryo + brain: download ENA fastqs.
set -euo pipefail
OUT=/ibex/scratch/projects/c2012/bernaljp/SpatialData/results/Deng_mm10
mkdir -p "$OUT"
B=https://ftp.sra.ebi.ac.uk/vol1/fastq
# SRR : subdir : label
declare -A SUB=( [SRR19441285]=085 [SRR19441281]=081 [SRR19441286]=086 [SRR19441282]=082 )
for srr in SRR19441285 SRR19441281 SRR19441286 SRR19441282; do
  d="$OUT/$srr"; mkdir -p "$d"
  for m in 1 2; do
    url="$B/SRR194/${SUB[$srr]}/$srr/${srr}_${m}.fastq.gz"
    echo "[$(date '+%T')] downloading $url"
    wget -c -q -O "$d/${srr}_${m}.fastq.gz" "$url" && echo "  done ${srr}_${m}" || { echo "  FAILED ${srr}_${m}"; exit 1; }
  done
done
echo "ALL DOWNLOADS DONE"; ls -la "$OUT"/*/*.fastq.gz | awk '{print $5, $9}'
