import sys
import gzip


regions = [ "CDS",
            "five_prime_UTR",
            "lnc_RNA",
            "miRNA",
            "ncRNA",
            "ncRNA_gene",
            "snoRNA",
            "snRNA",
            "three_prime_UTR",
            "tRNA"]

strands = ["+", "-"]
pfx = "TAIR10_chr_"


def init_region_labels(intervals):

    region_map = {}
    global regions
    global strands


    for chrom in intervals:

        if not chrom in region_map:
            region_map[chrom] = {}

        for interval in intervals[chrom]:
            region_map[chrom][(interval[0], interval[1])] = []

            for p in range(interval[0], interval[1]):
                pos_data = {}

                for strand in strands:
                    pos_data[strand] = {}

                    for region in regions:
                        pos_data[strand][region] = {'id':set(), 'name':set()}

                region_map[chrom][(interval[0], interval[1])].append(pos_data)

    return region_map

def update_region_labels(region_maps, gff_vals):
    
    chrom, start, end, strand, region, annot = gff_vals
    intervals = region_maps[chrom]

    for pos in range(start, end):
        for region_pos_tpl in region_maps[chrom]:

            if pos >= region_pos_tpl[0] and pos < region_pos_tpl[1]:
                relative_pos = pos - region_pos_tpl[0]
                
                feature_id, name = parse_id(annot)
                
                if region in regions:
                    region_maps[chrom][region_pos_tpl][relative_pos][strand][region]['id'].add(feature_id)
                    region_maps[chrom][region_pos_tpl][relative_pos][strand][region]['name'].add(name)

def make_files(region_map):

    global pfx

    for chrom in region_map:
        for region_pos_tpl in region_map[chrom]:
            with open(f"/clusterfs/jgi/scratch/science/wcplant/nberkowitz/{pfx}{chrom},{region_pos_tpl[0]},{region_pos_tpl[1]}_labels.tsv", "w") as label_file:
                interval_length = region_pos_tpl[1] - region_pos_tpl[0]
                for pos in range(interval_length):
                    region_strand_parts = set()
                    name_parts = set()
                    for strand in strands:
                        for region in regions:
                            
                            pos_data = region_map[chrom][region_pos_tpl][pos][strand][region]
                            name_set = pos_data['name']
                            id_set = pos_data['id']

                            name_parts = name_parts.union(name_set)
                            if len(id_set) > 0:
                                region_strand_parts.add(f"{region}({strand})")

                    region_strand_list = list(region_strand_parts)
                    region_strand_list.sort()
                    region_strand_concat = ",".join(region_strand_list)
                    
                    if "" in name_parts:
                        name_parts.remove("")
                    name_list = sorted(list(name_parts))
                    
                    name_concat = ""
                    if len(name_list) > 0:
                        name_concat = ",".join(name_list)
                        
                    print(f"{pos}\t{pos + region_pos_tpl[0]}\t{region_strand_concat}\t{name_concat}", file = label_file)


def parse_id(annot_concat):

    feature_id = ''
    name = ''
    fields = annot_concat.split(";")

    for f in fields:
        if (f[0:2] == "ID"):
            feature_id = f.split("=")[1]

        if (f[0:4] == "Name"):
            name = f.split("=")[1]
    return (feature_id, name)

if __name__ == "__main__":

    bed_intervals = {}
    chunk_size = 100000

    with open(sys.argv[1], 'r') as bed:
        for bed_line in bed:
            chrom, start, end = bed_line.split("\t")
            if not chrom in bed_intervals:
                bed_intervals[chrom] = []
            bed_intervals[chrom].append((int(start), int(end)))

    region_map = init_region_labels(bed_intervals)
    
    with gzip.open(sys.argv[2],'rt') as gff:
        for gff_line in gff:
            if gff_line[0] != '#':
                chrom, a, category, start, end, b, strand, c, annot = gff_line.rstrip().split("\t")
                
                if chrom in bed_intervals and strand in strands:
                    update_region_labels(region_map, [chrom, int(start), int(end), strand, category, annot])
    
    make_files(region_map)
