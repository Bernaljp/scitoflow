# Data-processing pipelines

Reproducible Snakemake pipelines that take the raw spatial multi-omic sequencing for
each dataset from FASTQ to the per-spot RNA and chromatin matrices consumed by
`scitoflow.preprocess.build_dataset`. They were developed to run on the KAUST Ibex
SLURM cluster; the SLURM headers, module names, and scratch paths in the `config.yaml`
files and `*.sbatch` submitters are cluster-specific and should be adapted to your site.

## What each group does

| Directory | Platform / dataset | Produces |
|-----------|--------------------|----------|
| `rna/`   | spatial-Mux-seq RNA (Guo et al. 2025) | spliced (mature) + unspliced (nascent) counts via kb-python `nac` |
| `atac/`  | spatial-Mux-seq ATAC (Guo et al. 2025) | fragments (bwa + sinto) and MACS3 peaks |
| `misar/` | MISAR-seq RNA + ATAC (E11.0-E18.5) | per-section spliced/unspliced RNA and ATAC fragments/peaks |
| `deng/`  | spatial-ATAC-RNA-seq (Zhang, Deng et al. 2023) | per-section RNA and ATAC fragments/peaks |

RNA is quantified with kb-python (kallisto/bustools) using the `nac` workflow, which emits
`mature` (spliced) and `nascent` (unspliced) layers. ATAC goes raw FASTQ to fragments to
peaks: bbduk barcode extraction, spatial-barcode demultiplexing (`taggd`), `bwa mem`
alignment, `sinto fragments`, then MACS3 peak calling. Chromatin gene activity is then
built downstream in `scitoflow.preprocess.gene_activity` by aggregating promoter-linked
peaks per gene. Every pipeline uses the same GRCm38/mm10 NCBI reference
(`GCF_000001635.20`) so the four datasets are processed identically.

## Layout of a pipeline group

- `Snakefile` (RNA) or `Snakefile_rna` / `Snakefile_atac` (MISAR, Deng): the rules.
- `config.yaml` (or `config_rna.yaml` / `config_atac.yaml`): accessions, reference URLs,
  output/reference/log paths, and the barcode whitelist to use.
- `snakemake*.sbatch`: the SLURM driver that activates the env and runs `snakemake`.
- `spatial_barcodes.txt` / `misar_barcodes.txt`: the 2,500-spot platform barcode whitelist.
- Helpers: `fastq_process.py` (reshapes R2 to a 16 bp cell barcode + 10 bp UMI for kb's
  `10xv2` preset), `BC_process.py` / `add_BC.py` (ATAC barcode handling),
  `remap_chrom.awk` + `ncbi2ucsc.tsv` (NCBI `NC_0...` to `chr` contig remapping),
  `misar_split_*.py` and `srr_map.json` (MISAR per-section splitting).

## Environment

`environment.yml` defines the `scitoflow-align` conda/mamba environment (kb-python,
kallisto, bustools, bwa, samtools, htslib, sinto, taggd, bedtools, MACS3, sra-tools,
snakemake, bbmap). Tools available as cluster modules (star, bwa, samtools, htslib,
bedtools, macs3, sra-tools) can be loaded instead; the non-module tools come from the env.

```bash
mamba env create -f environment.yml     # or: sbatch env_create.sbatch
```

## Running

`submit.sh` is a convenience wrapper (rsync the pipeline to cluster scratch, then dry-run
or submit):

```bash
bash submit.sh env          # create/update the scitoflow-align env on the cluster
bash submit.sh dryrun-rna   # snakemake -n (fast, safe)
bash submit.sh rna          # sbatch the RNA pipeline
bash submit.sh atac         # sbatch the ATAC pipeline
```

Or, on an allocation with the env active, run a group directly:

```bash
cd rna && snakemake -n            # dry run
cd rna && snakemake --cores 16    # execute
```

## Accessions

- spatial-Mux-seq (Guo et al. 2025): GEO `GSE263333`. 10k (20 um) brain: RNA `SRR30535682`,
  native ATAC `GSM8494157`.
- MISAR-seq (E11.0-E18.5): SRA `SRP491963` (eight RNA + eight ATAC libraries).
- spatial-ATAC-RNA-seq (Zhang, Deng et al. 2023): GEO `GSE205055` (ME13 embryo RNA
  `SRR19441281` / ATAC `SRR19441285`; adult brain RNA `SRR19441282` / ATAC `SRR19441286`).

The processed per-spot matrices these produce are assembled into a model-ready AnnData by
`scitoflow.preprocess.build_dataset` (moment smoothing, gene-activity construction, HVG
selection, and off-tissue masking via the authors' `tissue_positions_list.csv`).
