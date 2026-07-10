# Remap NCBI RefSeq chromosome accessions (col 1) to UCSC chr names using MAP (2-col tsv);
# drop rows whose chromosome is not in the map (unplaced scaffolds). Usage:
#   awk -v MAP=ncbi2ucsc.tsv -f remap_chrom.awk fragments.bed
BEGIN { FS = OFS = "\t"; while ((getline line < MAP) > 0) { split(line, a, "\t"); m[a[1]] = a[2] } }
($1 in m) { $1 = m[$1]; print }
