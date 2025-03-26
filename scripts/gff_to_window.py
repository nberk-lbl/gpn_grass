import sys
import gzip
import json

"""
Get avergage embeddings for functionally labeled genomic windows

- (prepare files before running) define introns: transcript - CDS
- label 100nt bins 
-- based on GFF
-- based on derived intron GFF
-- based on repeat bed

"""


run_cfg = json.loads(open(sys.argv[1], 'r').read())
window_size = run_cfg['window_size']
threshold = run_cfg['threshold']
outfile = f"{run_cfg['output_dir']}/{run_cfg['out_pfx']}.labels.csv"

main_gff = run_cfg['gff_path']
intron_bed = run_cfg['intron_bed']

windows = {} # windows by chromosome, start position and strand
seq_order = []
tags = run_cfg["gff_tags"]

def handle_fields(fields):

    chrom = fields[0]
    label = fields[2]
    start = fields[3]
    end = fields[4] 
    strand = fields[6]

    # if a specific chrom/contig is specified, only get those features. otherwise use all chromosomes
    if chrom in run_cfg["chrom_size"]:
        if label in tags:
            if not chrom in windows:
                windows[chrom] = {}
                seq_order.append(chrom)
            
            # mark the covered nucleotide in a heirarchichal dict
            for nt in range(start, end):
                window_num = int((nt-start) / window_size)
                if not window_num in windows[chrom]:
                    windows[chrom][window_num] = {}
                if not strand in windows[chrom][window_num]:
                    windows[chrom][window_num][strand] = {}
                if not label in windows[chrom][window_num][strand]:
                    windows[chrom][window_num][strand][label] = [0] * window_size
                
                pos = nt % window_size
                windows[chrom][window_num][strand][label][pos] += 1


## parse the tagged features from the main gff
# the main gff does not contain introns or repeats

## TEST CASE
## fields = "1	CDS	CDS	7650	7761	.	-	.	ID=transcript:AT1G01020.2;Parent=gene:AT1G01020;Name=ARV1-202;biotype=protein_coding;transcript_id=AT1G01020.2".split("\t")
## handle_fields(fields)


print("parsing main gff")
print(main_gff)
with gzip.open(main_gff, 'rb') as gff:
    for gff_line in gff:
        g_line = gff_line.decode("utf-8").rstrip()      
        if g_line[0] != '#':
            fields = g_line.split("\t")
            fields[3] = int(fields[3]) - 1
            fields[4] = int(fields[4])
            handle_fields(fields)
            

## add introns
print("parsing intron bed")
tags = ['intron']
with open(intron_bed) as i_bed:
    for bed_line in i_bed:
        chrom, start, end, name, score, strand = bed_line.rstrip().split("\t")
        fields = [chrom, '', 'intron', int(start), int(end), '', strand]
        handle_fields(fields)

## add repeats
print("parsing repeat bed")
tags = ['repeat']
with open(run_cfg["repeat_bed"]) as bed:
    for bed_line in bed:
        chrom, start, end = bed_line.rstrip().split("\t")[0:3]
        fields = [chrom, '', 'repeat', int(start), int(end), '', strand]
        handle_fields(fields)

tags = run_cfg["gff_tags"] + ['intron', 'repeat']

print(outfile)
with open(outfile, 'w') as csv:

    for chrom in run_cfg['chrom_order']:
        if (chrom in run_cfg['chrom_size']):
            max_window = int(run_cfg['chrom_size'][chrom] / window_size)
            for b in range(max_window):
                label = 'intergenic'
                
                # a window is ambiguous unless all of its nucleotides belong to exactly one feature
                # bins with features on both strands are ambiguous even if they match
                if chrom in windows and b in windows[chrom]:
                    strands = list(windows[chrom][b].keys())
                    if len(strands) != 1:
                        label = 'ambiguous'
                    else:
                        strand = strands[0]
                        labels = list(windows[chrom][b][strand].keys())
                        if len(labels) > 1:
                            label = 'ambiguous'
                        else:
                            positions = windows[chrom][b][strand][labels[0]]
                            if sum(positions) != 100 or max(positions) != 1:
                                label = 'ambiguous'
                            else:
                                label = labels[0]

                print(f"{chrom},{b},{label}", file=csv)

                    
                    
                
                


