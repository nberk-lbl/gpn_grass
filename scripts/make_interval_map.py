

def make_interval_map(interval, interval_file):
    bases = set()
    
    with open(interval_file) as f:
        for g in f:
            e = g.rstrip().split()


            start = int(e[3])
            end = int(e[4])
            
            for i in range(start, end):
                bases.add(i)

    with (open(f"{interval_file}.map", "w")) as map_file:
        for i in range(interval[1], interval[2]):
            v = 0
            if i in bases:
                v = 1
            print(f"{i}\t{v}", file=map_file)



"""
Chr01   phytozomev13    exon    175798  175846  .       +       .       ID=Potri.001G002700.2.v4.1.exon.4;Parent=Potri.001G002700.2.v4.1;pacid=42791221
Chr01   phytozomev13    exon    175949  176003  .       +       .       ID=Potri.001G002700.2.v4.1.exon.5;Parent=Potri.001G002700.2.v4.1;pacid=42791221
Chr01   phytozomev13    exon    176235  176356  .       +       .       ID=Potri.001G002700.2.v4.1.exon.6;Parent=Potri.001G002700.2.v4.1;pacid=42791221

Chr01   phytozomev13    exon    175798  175846  .       +       .       ID=Potri.001G002700.2.v4.1.exon.4;Parent=Potri.001G002700.2.v4.1;pacid=42791221
Chr01   phytozomev13    exon    175949  176003  .       +       .       ID=Potri.001G002700.2.v4.1.exon.5;Parent=Potri.001G002700.2.v4.1;pacid=42791221
Chr01   phytozomev13    exon    176235  176356  .       +       .       ID=Potri.001G002700.2.v4.1.exon.6;Parent=Potri.001G002700.2.v4.1;pacid=42791221
Chr01   phytozomev13    exon    176220  176400  .       -       .       ID=Potri.001G002800.1.v4.1.exon.4;Parent=Potri.001G002800.1.v4.1;pacid=42793160


"""

at_region = ("5", 3566900, 3567600)
interval_file = "genomes/at.interval"

make_interval_map(at_region, interval_file)


pt_region = ("Chr01", 175700, 176400)
make_interval_map(pt_region, "genomes/pt.interval")

gm_region = ("Chr01", 181200, 181900)
make_interval_map(gm_region, "genomes/gm.interval")

