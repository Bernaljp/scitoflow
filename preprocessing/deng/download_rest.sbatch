#!/bin/bash
#SBATCH --job-name=deng_dl_rest
#SBATCH --partition=batch
#SBATCH -A pi-gomezcd
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --output=/ibex/scratch/projects/c2012/bernaljp/scitoflow_pipeline/deng/out/dlrest_%j.out
#SBATCH --error=/ibex/scratch/projects/c2012/bernaljp/scitoflow_pipeline/deng/err/dlrest_%j.err
# Fetch the 2 remaining Deng SRRs (Brain RNA 282, Brain ATAC 286), 4 files in parallel, wget -c.
set -uo pipefail
OUT=/ibex/scratch/projects/c2012/bernaljp/SpatialData/results/Deng_mm10
mkdir -p "$OUT/SRR19441282" "$OUT/SRR19441286"
urls=(
  "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR194/082/SRR19441282/SRR19441282_1.fastq.gz|$OUT/SRR19441282/SRR19441282_1.fastq.gz"
  "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR194/082/SRR19441282/SRR19441282_2.fastq.gz|$OUT/SRR19441282/SRR19441282_2.fastq.gz"
  "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR194/086/SRR19441286/SRR19441286_1.fastq.gz|$OUT/SRR19441286/SRR19441286_1.fastq.gz"
  "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR194/086/SRR19441286/SRR19441286_2.fastq.gz|$OUT/SRR19441286/SRR19441286_2.fastq.gz"
)
printf '%s\n' "${urls[@]}" | xargs -P4 -I{} bash -c 'u="${1%%|*}"; o="${1##*|}"; echo "get $o"; wget -c -q -O "$o" "$u" && echo "done $o" || echo "FAIL $o"' _ {}
echo "REST DONE"; ls -la "$OUT"/*/*.fastq.gz | awk '{print $5, $9}'
