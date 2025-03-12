### make a bed file with some of the AT chromosomes (manually edit output)
python get_chrom_sizes.py ../genomes/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz

### get the complex labels for the interval
python melt_gff.py AT_train_chrom.bed ../genomes/Arabidopsis_thaliana.TAIR10.60.gff3.gz

### combine the files in JGI_SCRATCH. (delete headers manually first)
cat 1_labels.tsv 4_labels.tsv 5_labels.tsv > labels.tsv
 (cat <(tail -n +2 1_labels.tsv) <(tail -n +2 4_labels.tsv) <(tail -n +2 5_labels.tsv)) > embeddings.tsv  
 (cat <(tail -n +2 TAIR10_embeddings_1.tsv) <(tail -n +2 TAIR10_embeddings_4.tsv) <(tail -n +2 TAIR10_embeddings_5.tsv)) > embeddings.tsv  



/clusterfs/jgi/scratch/science/wcplant/nberkowitz