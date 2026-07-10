"""
scATAC fragments -> promoter-linked gene-activity matrix (calculate_gene_activity).

FAITHFUL first-pass extraction from the method-development notebook
(Base/model1new_multivelo_organized.ipynb), Phase A1 consolidation.
Logic preserved verbatim; only module imports were added/normalized.
NOT yet unit-tested or made device-agnostic (hardcoded .cuda() remains) -
that is the research-software hardening pass. See PLAN.md Phase A.
"""

# numpy>=2 removed the deprecated `numpy.NaN` alias that episcanpy 0.4.0 imports at
# load time; restore it before importing episcanpy.
import numpy as _np
if not hasattr(_np, "NaN"):
    _np.NaN = _np.nan

# --- faithful extraction: notebook cell 7 (model1new_multivelo_organized.ipynb) ---
import scanpy as sc
import episcanpy as epi
import anndata as ad
import pandas as pd
import numpy as np
import pyranges as pr
import pysam
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix, lil_matrix
import os
from gtfparse import read_gtf

def calculate_gene_activity(
    fragments_file: str,
    peaks_file: str,
    gtf_file: str,
    output_file: str = None,
    min_counts_per_cell: int = 1000,
    min_cells_per_peak: int = 10,
    promoter_upstream: int = 2000,
    promoter_downstream: int = 500
) -> ad.AnnData:
    """
    Calculates a gene activity matrix from scATAC-seq fragments and peaks.

    This pipeline performs the following steps:
    1.  Constructs a cell-by-peak count matrix from a fragments file.
    2.  Performs basic QC filtering and TF-IDF normalization.
    3.  Links peaks to gene promoters, handling chromosome name mismatches.
    4.  Aggregates peak counts into a cell-by-gene activity matrix.
    5.  Returns the final AnnData object and optionally saves it to a file.

    Args:
        fragments_file: Path to the indexed (bgzipped and tabix) fragments file.
        peaks_file: Path to the .narrowPeak file defining accessibility peaks.
        gtf_file: Path to the gene transfer format (GTF) annotation file.
        output_file: Optional. Path to save the final AnnData object as an .h5ad file.
        min_counts_per_cell: Minimum number of fragments required for a cell to pass QC.
        min_cells_per_peak: Minimum number of cells a peak must be detected in to pass QC.
        promoter_upstream: Distance upstream of the TSS to define the promoter region.
        promoter_downstream: Distance downstream of the TSS to define the promoter region.

    Returns:
        An AnnData object where rows are cells and columns are genes, containing the
        gene activity scores.
    """
    print("--- Starting Gene Activity Calculation Pipeline ---")
    print(f"Fragments: {fragments_file}")
    print(f"Peaks:     {peaks_file}")
    print(f"GTF:       {gtf_file}")
    print("-------------------------------------------------")

    # ==============================================================================
    # 1. CREATE CELL-BY-PEAK COUNT MATRIX
    # ==============================================================================
    print("\nStep 1: Creating cell-by-peak count matrix from fragment file...")
    peaks_df = pd.read_csv(
        peaks_file, sep='\t', header=None, usecols=[0, 1, 2],
        names=['Chromosome', 'Start', 'End']
    )
    peaks_df['peak_name'] = peaks_df['Chromosome'].astype(str) + ':' + peaks_df['Start'].astype(str) + '-' + peaks_df['End'].astype(str)
    peak_names = peaks_df['peak_name'].tolist()
    peak_map = {name: i for i, name in enumerate(peak_names)}
    all_barcodes = pd.read_csv(fragments_file, sep='\t', header=None, usecols=[3]).squeeze().unique()
    barcode_map = {name: i for i, name in enumerate(all_barcodes)}
    print(f"--> Found {len(all_barcodes)} unique barcodes.")
    n_cells, n_peaks = len(all_barcodes), len(peak_names)
    count_matrix = lil_matrix((n_cells, n_peaks), dtype=np.int32)
    tabix_file = pysam.TabixFile(fragments_file)
    print("--> Counting fragments in peaks...")
    for idx, peak in tqdm(peaks_df.iterrows(), total=n_peaks, desc="Processing peaks"):
        peak_idx = peak_map[peak['peak_name']]
        try:
            for read in tabix_file.fetch(peak['Chromosome'], peak['Start'], peak['End']):
                barcode = read.split('\t')[3]
                if barcode in barcode_map:
                    cell_idx = barcode_map[barcode]
                    count_matrix[cell_idx, peak_idx] += 1
        except ValueError:
            pass # Ignore chromosomes not found in the fragment file
    adata_peaks = ad.AnnData(
        X=count_matrix.tocsr(),
        obs=pd.DataFrame(index=all_barcodes),
        var=pd.DataFrame(index=peak_names)
    )
    print(f"--> Initial matrix shape: {adata_peaks.shape}")

    # ==============================================================================
    # 2. BASIC QC AND NORMALIZATION
    # ==============================================================================
    print("\nStep 2: Performing QC and TF-IDF normalization...")
    adata_peaks.layers['counts'] = adata_peaks.X.copy()
    sc.pp.calculate_qc_metrics(adata_peaks, percent_top=None, inplace=True)
    sc.pp.filter_cells(adata_peaks, min_counts=min_counts_per_cell)
    print(f"--> Shape after filtering cells: {adata_peaks.shape}")
    sc.pp.filter_genes(adata_peaks, min_cells=min_cells_per_peak)
    print(f"--> Shape after filtering peaks: {adata_peaks.shape}")
    adata_peaks.X[adata_peaks.X > 1] = 1 # Binarize
    epi.pp.tfidf(adata_peaks)
    print("--> TF-IDF normalization complete.")

    # ==============================================================================
    # 3. LINK PEAKS TO GENES VIA PROMOTERS
    # ==============================================================================
    print("\nStep 3: Linking peaks to gene promoters...")
    print("--> Parsing GTF with gtfparse...")
    gtf_df_raw = read_gtf(gtf_file)
    gtf = pr.PyRanges(gtf_df_raw.rename({'seqname': 'Chromosome','start': 'Start','end': 'End','strand': 'Strand'}).to_pandas())

    print("--> Standardizing chromosome names (NCBI to UCSC)...")
    # Note: This mapping is for GRCm38. It may need to be adapted for other genomes.
    ncbi_to_ucsc_map = {
        'NC_000067.6': 'chr1', 'NC_000068.7': 'chr2', 'NC_000069.6': 'chr3',
        'NC_000070.6': 'chr4', 'NC_000071.6': 'chr5', 'NC_000072.6': 'chr6',
        'NC_000073.6': 'chr7', 'NC_000074.6': 'chr8', 'NC_000075.6': 'chr9',
        'NC_000076.6': 'chr10', 'NC_000077.6': 'chr11', 'NC_000078.6': 'chr12',
        'NC_000079.6': 'chr13', 'NC_000080.6': 'chr14', 'NC_000081.6': 'chr15',
        'NC_000082.6': 'chr16', 'NC_000083.6': 'chr17', 'NC_000084.6': 'chr18',
        'NC_000085.6': 'chr19', 'NC_000086.7': 'chrX', 'NC_000087.7': 'chrY',
        'NC_005089.1': 'chrM'
    }
    gtf_df = gtf.df.copy()
    if any(c in ncbi_to_ucsc_map for c in gtf_df['Chromosome'].unique()):
        print("--> Detected NCBI accessions. Applying conversion.")
        gtf_df['Chromosome'] = gtf_df['Chromosome'].map(ncbi_to_ucsc_map)
        gtf_df.dropna(subset=['Chromosome'], inplace=True)
        gtf = pr.PyRanges(gtf_df)

    genes = gtf[gtf.feature == 'gene']

    print("--> Manually calculating TSS...")
    genes_df = genes.df
    tss_df = genes_df.copy()
    plus_strand = tss_df['Strand'] == '+'
    tss_df.loc[plus_strand, 'End'] = tss_df.loc[plus_strand, 'Start'] + 1
    minus_strand = tss_df['Strand'] == '-'
    tss_df.loc[minus_strand, 'Start'] = tss_df.loc[minus_strand, 'End'] - 1
    tss = pr.PyRanges(tss_df)

    print("--> Manually calculating promoter regions...")
    promoter_df = tss.df.copy()
    plus_mask = promoter_df['Strand'] == '+'
    promoter_df.loc[plus_mask, 'End'] = promoter_df.loc[plus_mask, 'Start'] + promoter_downstream
    promoter_df.loc[plus_mask, 'Start'] = promoter_df.loc[plus_mask, 'Start'] - promoter_upstream
    minus_mask = promoter_df['Strand'] == '-'
    promoter_df.loc[minus_mask, 'Start'] = promoter_df.loc[minus_mask, 'End'] - promoter_downstream
    promoter_df.loc[minus_mask, 'End'] = promoter_df.loc[minus_mask, 'End'] + promoter_upstream
    promoter_df['Start'] = promoter_df['Start'].clip(lower=0)
    promoters = pr.PyRanges(promoter_df)

    peak_pr_df = adata_peaks.var_names.to_series().str.split('[:-]', expand=True, regex=True)
    peak_pr_df.columns = ['Chromosome', 'Start', 'End']
    peak_pr_df['peak_name'] = adata_peaks.var_names
    peaks_pr = pr.PyRanges(peak_pr_df)

    print("--> Finding peak-promoter overlaps...")
    joined_pr = promoters.join(peaks_pr)
    if joined_pr.empty:
        raise ValueError("Join resulted in an empty set. Check chromosome names in both files.")
    joined_df = joined_pr.df

    if 'gene_name' in joined_df.columns:
        joined_df.dropna(subset=['gene_name'], inplace=True)
        gene_to_peaks = joined_df.groupby('gene_name')['peak_name'].apply(list).to_dict()
    elif 'gene' in joined_df.columns:
        print("--> 'gene_name' not found. Using 'gene' column as the identifier.")
        joined_df.dropna(subset=['gene'], inplace=True)
        gene_to_peaks = joined_df.groupby('gene')['peak_name'].apply(list).to_dict()
    else:
        raise KeyError("Could not find 'gene_name' or 'gene' in the GTF attributes.")
    print(f"--> Linked {len(gene_to_peaks)} genes to one or more peaks.")

    # ==============================================================================
    # 4. AGGREGATE PEAK COUNTS TO CREATE GENE ACTIVITY MATRIX
    # ==============================================================================
    print("\nStep 4: Aggregating peaks to create gene activity matrix...")
    peak_indices_map = {name: i for i, name in enumerate(adata_peaks.var_names)}
    genes_with_peaks = sorted(gene_to_peaks.keys())
    gene_indices_map = {name: i for i, name in enumerate(genes_with_peaks)}
    gene_activity_matrix = lil_matrix((adata_peaks.shape[0], len(genes_with_peaks)), dtype=np.float32)

    for gene_name, peak_list in tqdm(gene_to_peaks.items(), desc="Aggregating peaks"):
        gene_idx = gene_indices_map.get(gene_name)
        if gene_idx is not None:
            peak_indices = [peak_indices_map[p] for p in peak_list if p in peak_indices_map]
            if peak_indices:
                score = adata_peaks.X[:, peak_indices].sum(axis=1)
                gene_activity_matrix[:, gene_idx] = score

    # ==============================================================================
    # 5. CREATE FINAL ANNDATA OBJECT
    # ==============================================================================
    print("\nStep 5: Creating final AnnData object for gene activities...")
    adata_genes = ad.AnnData(
        X=gene_activity_matrix.tocsr(),
        obs=adata_peaks.obs.copy(),
        var=pd.DataFrame(index=genes_with_peaks)
    )
    
    if output_file:
        print(f"\n--> Saving final object to '{output_file}'")
        adata_genes.write(output_file)

    print(f"\n✅ Success! Final object shape: {adata_genes.shape}")
    return adata_genes

