



"""
bins that have no annotations on either alelle are marked "intergenic" but these seem to bleed into
other regions

re-label intergenic bins as ambiguous unless they are flanked intergenic bins. ambiguous bins will
be dropped from the analysis downstream

"""

import json
import sys

windows_by_chrom = {}
chrom_names = []
flank_size = 10
run_cfg = json.loads(open(sys.argv[1], 'r').read())
pfx = run_cfg['out_pfx']

with open(f'data/{pfx}.windows.csv') as label_file:
    
    # read the left flank into the buffer and write those lines to the output
    for label_line in label_file:
        chrom, window, label = label_line.rstrip().split(",")
        if not chrom in windows_by_chrom:
            chrom_names.append(chrom)
            windows_by_chrom[chrom] = []
        windows_by_chrom[chrom].append([chrom, window, label])

with open(f'data/{pfx}.windows_intergenic_repeat_padded.csv', 'w') as label_file:        
    for chrom in chrom_names:
        windows = windows_by_chrom[chrom]
        w = 0

        # handle the upstream flank
        for i in range(flank_size):
            chrom, window, label = windows[w]
            if (label in ['intergenic', 'repeat']):
                print(f"{chrom},{window},ambiguous", file=label_file)
            else:
                print(f"{chrom},{window},{label}", file=label_file)
            w += 1

        for i in range(len(windows) - (flank_size * 2 + 1) + 1):
            chrom, window, label = windows[w]

            # if a window is labeled as intergenic, check that all flanking windows are intergenic
            # if not, lable it ambiguous
            if label in ['intergenic', 'repeat']:

                intergenic_count = 0
                for j in range(flank_size * 2 + 1):
                    window_in_flank = windows[w - flank_size + j]
                    if window_in_flank[2] in ['intergenic', 'repeat']:
                        intergenic_count += 1

                if intergenic_count == flank_size * 2 + 1:
                    print(f"{chrom},{window},{label}", file=label_file)
                else:
                    print(f"{chrom},{window},ambiguous", file=label_file)

            else:
                print(f"{chrom},{window},{label}", file=label_file)

            w += 1

        # handle the downstream flank
        for i in range(flank_size):
            chrom, window, label = windows[w]
            if (label in ['intergenic', 'repeat']):
                print(f"{chrom},{window},ambiguous", file=label_file)
            else:
                print(f"{chrom},{window},{label}", file=label_file)
            w += 1
