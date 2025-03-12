import gpn.model
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from Bio import SeqIO
import gzip

def get_seq(fa_gz_file, region):

    chrom, start, end = region
    seq = ""

    with gzip.open(fa_gz_file, "rt") as fa:
        for record in SeqIO.parse(fa, "fasta"):
            if record.id.split()[0] == chrom:
                seq = record.seq[start:end]
    return(str(seq))

def plot_region(tokenizer, seq, pfx):
    input_ids = tokenizer(seq, return_tensors="pt", return_attention_mask=False, return_token_type_ids=False)["input_ids"]
    #torch.Size([1, 700])
    #tensor([4, 5, 5, 5, 6, 6, 3, 3, 3, 3])

    with torch.no_grad():
        embedding = model(input_ids=input_ids).last_hidden_state
    embedding.shape

    embedding_df = pd.DataFrame(StandardScaler().fit_transform(embedding[0].numpy()))
    embedding_df.index.name = "Position"
    embedding_df.columns.name = "Embedding dimension"

    plt.figure(figsize=(10, 6))
    sns.heatmap(embedding_df.T, center=0, vmin=-3, vmax=3, cmap="coolwarm", square=True, xticklabels=100, yticklabels=100)

    plt.savefig(f'{pfx}.png')
    


if __name__ == "__main__":

    model_path = "songlab/gpn-brassicales"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.get_vocab()

    model = AutoModel.from_pretrained(model_path)
    model.eval()

    at_region = ("5", 3566900, 3567600)
    at_fa = "genomes/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz"

    seq = get_seq(at_fa, at_region)
    plot_region(tokenizer, seq, "At")

    """
    Chr01   phytozomev13    exon    174067  174321  .       +       .       ID=Potri.001G002700.2.v4.1.exon.1;Parent=Potri.001G002700.2.v4.1;pacid=42791221
    Chr01   phytozomev13    exon    174919  175248  .       +       .       ID=Potri.001G002700.2.v4.1.exon.2;Parent=Potri.001G002700.2.v4.1;pacid=42791221
    Chr01   phytozomev13    exon    175498  175586  .       +       .       ID=Potri.001G002700.2.v4.1.exon.3;Parent=Potri.001G002700.2.v4.1;pacid=42791221
    --
    Chr01   phytozomev13    exon    175798  175846  .       +       .       ID=Potri.001G002700.2.v4.1.exon.4;Parent=Potri.001G002700.2.v4.1;pacid=42791221
    Chr01   phytozomev13    exon    175949  176003  .       +       .       ID=Potri.001G002700.2.v4.1.exon.5;Parent=Potri.001G002700.2.v4.1;pacid=42791221
    Chr01   phytozomev13    exon    176235  176356  .       +       .       ID=Potri.001G002700.2.v4.1.exon.6;Parent=Potri.001G002700.2.v4.1;pacid=42791221
    --
    Chr01   phytozomev13    exon    176453  176512  .       +       .       ID=Potri.001G002700.2.v4.1.exon.7;Parent=Potri.001G002700.2.v4.1;pacid=42791221
    Chr01   phytozomev13    exon    176612  176752  .       +       .       ID=Potri.001G002700.2.v4.1.exon.8;Parent=Potri.001G002700.2.v4.1;pacid=42791221
    """    

    pt_fa = "genomes/Ptrichocarpa_533_v4.0.fa.gz"
    pt_region = ("Chr01", 175700, 176400)

    seq = get_seq(pt_fa, pt_region)
    plot_region(tokenizer, seq, "Pt_001G002700")

    """
		974		Chr01	phytozomev10	exon	181151	181290	.	+	.	ID=Glyma.01G000900.1.Wm82.a2.v1.exon.11;Parent=Glyma.01G000900.1.Wm82.a2.v1;pacid=30543461													
		274		Chr01	phytozomev10	exon	181425	181583	.	+	.	ID=Glyma.01G000900.1.Wm82.a2.v1.exon.12;Parent=Glyma.01G000900.1.Wm82.a2.v1;pacid=30543461													
		236		Chr01	phytozomev10	exon	181661	181764	.	+	.	ID=Glyma.01G000900.1.Wm82.a2.v1.exon.13;Parent=Glyma.01G000900.1.Wm82.a2.v1;pacid=30543461													
		268		Chr01	phytozomev10	exon	181929	182104	.	+	.	ID=Glyma.01G000900.1.Wm82.a2.v1.exon.14;Parent=Glyma.01G000900.1.Wm82.a2.v1;pacid=30543461													
		6173		Chr01	phytozomev10	exon	188102	188291	.	+	.	ID=Glyma.01G000900.1.Wm82.a2.v1.exon.15;Parent=Glyma.01G000900.1.Wm82.a2.v1;pacid=30543461													
		740		Chr01	phytozomev10	exon	188842	188941	.	+	.	ID=Glyma.01G000900.1.Wm82.a2.v1.exon.16;Parent=Glyma.01G000900.1.Wm82.a2.v1;pacid=30543461													
    """

    gm_fa = "genomes/Gmax_275_v2.0.fa.gz"
    gm_region = ("Chr01", 181200, 181900)

    seq = get_seq(gm_fa, gm_region)
    plot_region(tokenizer, seq, "Gm_01G000900")







