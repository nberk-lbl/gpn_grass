
valid_labels = [
    "chromosome(.)",
    "CDS(-),chromosome(.),exon(-),gene(-),mRNA(-)",
    "CDS(+),chromosome(.),exon(+),gene(+),mRNA(+)",
    "chromosome(.),gene(-),mRNA(-)",
    "chromosome(.),gene(+),mRNA(+)",
    "chromosome(.),exon(+),gene(+),mRNA(+),three_prime_UTR(+)",
    "chromosome(.),exon(-),gene(-),mRNA(-),three_prime_UTR(-)",
    "chromosome(.),exon(+),five_prime_UTR(+),gene(+),mRNA(+)",
    "chromosome(.),exon(-),five_prime_UTR(-),gene(-),mRNA(-)",
    "CDS(+),chromosome(.),exon(+),gene(+),mRNA(+),three_prime_UTR(+)",
    "CDS(-),chromosome(.),exon(-),gene(-),mRNA(-),three_prime_UTR(-)",
    "chromosome(.),exon(-),lnc_RNA(-),ncRNA_gene(-)",
    "chromosome(.),exon(+),lnc_RNA(+),ncRNA_gene(+)",
    "CDS(+),chromosome(.),exon(+),five_prime_UTR(+),gene(+),mRNA(+)",
    "CDS(-),chromosome(.),exon(-),five_prime_UTR(-),gene(-),mRNA(-)",
    "CDS(+),chromosome(.),exon(+),exon(-),gene(+),lnc_RNA(-),mRNA(+),ncRNA_gene(-)",
    "chromosome(.),exon(+),exon(-),gene(+),gene(-),mRNA(+),mRNA(-),three_prime_UTR(+),three_prime_UTR(-)",
    "CDS(-),chromosome(.),exon(+),exon(-),gene(-),lnc_RNA(+),mRNA(-),ncRNA_gene(+)",
    "CDS(+),chromosome(.),exon(+),exon(-),gene(+),gene(-),mRNA(+),mRNA(-),three_prime_UTR(-)",
    "CDS(-),chromosome(.),exon(+),exon(-),gene(+),gene(-),mRNA(+),mRNA(-),three_prime_UTR(+)"
]


feature_sets = {}
for v in valid_labels:
    features = set(v.split(","))
    feature_sets[v] = features

most_subbed = {}

def choose_label(label):

    # if it's a selected label return it
    if label in feature_sets:
        return label

    # if it's a subset of exactly one selected label, return the superset label
    label_features = set(label.split(","))
    matches = []

    for f in feature_sets:
        s = feature_sets[f]
        if len(label_features - s) == 0:
            matches.append(f)
    if len(matches) == 1:
        if not (label, f) in most_subbed.keys():
            most_subbed[(label, f)] = 0
        most_subbed[(label, f)] += 1
        return matches[0]

    # if it isn't covered by one of the other cases return other
    return "chromosome(.),other"


with open("/pscratch/sd/n/nberk/grass/labels.tsv") as label_file, open("/pscratch/sd/n/nberk/grass/labels_subset.tsv", "w") as subset_file :
    label_file.readline()
    for x in label_file:
        chrom, pos, label = x.rstrip().split("\t")
        super_label = choose_label(label)
        print(f"{chrom}\t{pos}\t{super_label}", file=subset_file)

sub_counts = {}
for s in most_subbed:
    sub_counts[most_subbed[s]] = s

sorted_counts = sorted(sub_counts.keys(), reverse=True)
with open("most_subbed.txt", "w") as ms:
    for k in sorted_counts:
        c = sub_counts[k]
        print(f"{k}\t{c[0]}\t{c[1]}", file=ms)