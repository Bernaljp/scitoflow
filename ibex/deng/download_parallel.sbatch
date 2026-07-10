#!/bin/bash
#SBATCH --job-name=deng_dl_par
#SBATCH --partition=batch
#SBATCH -A pi-gomezcd
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/ibex/scratch/projects/c2012/bernaljp/scitoflow_pipeline/deng/out/dlpar_%j.out
#SBATCH --error=/ibex/scratch/projects/c2012/bernaljp/scitoflow_pipeline/deng/err/dlpar_%j.err
# Parallel ENA download (4 concurrent, wget -c resumes partials from the sequential run).
set -uo pipefail
OUT=/ibex/scratch/projects/c2012/bernaljp/SpatialData/results/Deng_mm10
B=https://ftp.sra.ebi.ac.uk/vol1/fastq
declare -A SUB=( [SRR19441285]=085 [SRR19441281]=081 [SRR19441286]=086 [SRR19441282]=082 )
dl() { srr=$1; m=$2; d="$OUT/$srr"; mkdir -p "$d"; wget -c -q -O "$d/${srr}_${m}.fastq.gz" "$B/SRR194/${SUB[$srr]}/$srr/${srr}_${m}.fastq.gz" && echo "done ${srr}_${m}" || echo "FAIL ${srr}_${m}"; }
export -f dl; export OUT B; export SUB
# 8 files, 4 at a time
( for srr in SRR19441285 SRR19441281 SRR19441286 SRR19441282; do for m in 1 2; do echo "$srr $m"; done; done ) | \
  xargs -P4 -n2 bash -c 'dl "$0" "$1"'
echo "ALL DONE"; ls -la "$OUT"/*/*.fastq.gz | awk '{print $5, $9}'
