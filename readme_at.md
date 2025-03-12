## Using GPN for functional predicion

This paper uses a machine learning model to predict genomic properties.

https://doi.org/10.1073/pnas.2311219120

One application is detecting genomic features e.g. exons, introns.

#### setup

clone this repo https://github.com/songlab-cal/gpn

set this to use scratch for intermediate files (managed by HuggingFace)

`export HF_HOME=$PSCRATCH`
conda environment: `gpn_conda.yaml`


#### genomes

##### arabidopsis from ENSEMBL
https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60/gff3/arabidopsis_thaliana/

repeats - this site also has poplar
http://ucsc.gao-lab.org/cgi-bin/hgTables?hgsid=167291_E9nY5UIAQRUOAR01xJAsum4vDukw

introns are transcripts minus exons (this will include the rare introns that occur in UTRs)
`zgrep exon Arabidopsis_thaliana.TAIR10.60.chromosome.4.gff3.gz > Arabidopsis_thaliana.TAIR10.60.chromosome.4.exon.gff3`
`zgrep mRNA Arabidopsis_thaliana.TAIR10.60.chromosome.4.gff3.gz > Arabidopsis_thaliana.TAIR10.60.chromosome.4.transcript.gff3`
`bedtools subtract -a Arabidopsis_thaliana.TAIR10.60.chromosome.4.transcript.gff3 -b Arabidopsis_thaliana.TAIR10.60.chromosome.4.exon.gff3 > Arabidopsis_thaliana.TAIR10.60.chromosome.4.intron.gff3`

the same pattern for poplar
`zgrep exon Ptrichocarpa_533_v4.1.gene_exons.gff3.gz > Ptrichocarpa_533_v4.1.exons.gff3`
`zgrep mRNA Ptrichocarpa_533_v4.1.gene_exons.gff3.gz > Ptrichocarpa_533_v4.1.transcripts.gff3`
`bedtools subtract -a Ptrichocarpa_533_v4.1.transcripts.gff3 -b Ptrichocarpa_533_v4.1.exons.gff3 > Ptrichocarpa_533_v4.1.introns.gff3`

give each bin a single region label if possible. Ignore bins that stradle regions or have conflicting regions on different strands
`python gff_to_window.py TAIR10.chr4.json > TAIR10.chr4.anno.csv`

#### steps

get average embeddings for each 100nt window
`python run_gpn.py TAIR10.chr4.json`

make umap plot colored by labels

#### result
Putting the labels on the umap doesn't label cluseters. 

<image>

#### TODO
Try with an arabidopsis chromosome
- get the chrom size ahead of time so the files are the same dimension
- limit the number of unlabeled bins to ~ the number of labeled bins
- discard ambiguous bins (see paper)

zgrep exon Ptrichocarpa_533_v4.1.gene_exons.gff3.gz > Ptrichocarpa_533_v4.1.exons.gff3
zgrep mRNA Ptrichocarpa_533_v4.1.gene_exons.gff3.gz > Ptrichocarpa_533_v4.1.transcripts.gff3
bedtools subtract -a Ptrichocarpa_533_v4.1.transcripts.gff3 -b Ptrichocarpa_533_v4.1.exons.gff3 > Ptrichocarpa_533_v4.1.introns.gff3

## arabidopsis whole genome
script now gets path info from json

`python gff_to_window.py TAIR10.allchrom.json`

`python run_gpn.py TAIR10.allchrom.json`

this makes one file per chromosome. cat them in order, skipping the headers

`awk FNR!=1 TAIR10_all_1_all_avg_embeddings.tsv TAIR10_all_2_all_avg_embeddings.tsv TAIR10_all_3_all_avg_embeddings.tsv TAIR10_all_4_all_avg_embeddings.tsv TAIR10_all_5_all_avg_embeddings.tsv TAIR10_all_Mt_all_avg_embeddings.tsv TAIR10_all_Pt_all_avg_embeddings.tsv > TAIR10_all_chrom_all_avg_embedding.tsv`


## eliminating cat step and attempting to use json inputs for everything
`python run_gpn.py TAIR10.allchrom.json`

`python gff_to_window.py TAIR10.allchrom.json`

`python pad_intergenic.py TAIR10.allchrom.json`

`python sk_umap.py TAIR10.allchrom.json`


## still need to awk these manually to avoid a full re-run. won't be necessary in the future


`awk FNR!=1 Ptrichocarpa.Chr??.avg_embeddings_N.tsv > Ptrichocarpa.avg_embeddings_N.tsv`

`python run_gpn.py Ptrichocarpa_533_v4.0.json`

`awk FNR!=1 Ptrichocarpa.Chr??.avg_embeddings_N.tsv > Ptrichocarpa.avg_embeddings_N.tsv`

`python gff_to_window.py Ptrichocarpa_533_v4.0.json`

`python pad_intergenic.py Ptrichocarpa_533_v4.0.json`

`python sk_umap.py Ptrichocarpa_533_v4.0.json`


## soybean.

`gff2bed < genomes/Gmax_275_Wm82.a2.v1.gene_exons.gff3 > genomes/Gmax_275_Wm82.a2.v1.gene_exons.bed`

`grep exon genomes/Gmax_275_Wm82.a2.v1.gene_exons.bed > genomes/Gmax_275_Wm82.a2.v1.exons.bed`

`grep mRNA genomes/Gmax_275_Wm82.a2.v1.gene_exons.bed > genomes/Gmax_275_Wm82.a2.v1.transcript.bed`

`bedtools subtract -b genomes/Gmax_275_Wm82.a2.v1.exons.bed -a genomes/Gmax_275_Wm82.a2.v1.transcript.bed > genomes/Gmax_275_Wm82.a2.v1.intron.bed`


# AT nat-si transcript analysis
natural anti-sense transcripts come in pairs of auto-complimentary transcripts.
a subset are "convergent" meaning they are near eachother on the chromosome and have promotors pointing at one another

this paper is about a databse for these transcripts
`https://pmc.ncbi.nlm.nih.gov/articles/PMC3245084/`

download all and get the convergent ones
`wget https://bis.zju.edu.cn/pnatdb/system/nat/ath.nat.final.out`

`cut -f4 convergent_NAT.tsv > ids1.txt`
`cut -f2 convergent_NAT.tsv > ids2.txt`
`cat ids* > nat_ids.txt`
`grep -f nat_ids.txt ../../genomes/TAIR10_transcript.gff > nat.gff`

### some bed arithmatic to get medium sized chuncks containing nats.
`bedtools slop -i nat.bed -g TAIR10.genome -b 1000 > nat_slop1000.bed`
`bedtools merge -i nat_slop1000.bed > nat_merge.bed`
`sort -sizeD -i nat_merge.bed | head -n10 > nat_examples.bed`





