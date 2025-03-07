import sys

label_hist = {}
with open("/pscratch/sd/n/nberk/grass/labels.tsv") as label_file:
    label_file.readline()
    for x in label_file:
        chrom, pos, label = x.rstrip().split("\t")
        if not label in label_hist:
            label_hist[label] = 0
        label_hist[label] += 1

labels_by_count = {}
for label in label_hist.keys():
    count = label_hist[label]
    if not count in labels_by_count:
        labels_by_count[count] = []
    labels_by_count[count].append(label)

sorted_counts = sorted(labels_by_count.keys(), reverse=True)

with open("label_counts.tsv", 'w') as counts_file:
    for count in sorted_counts:
        for label in labels_by_count[count]:
            print(f"{count}\t{label}", file=counts_file)

