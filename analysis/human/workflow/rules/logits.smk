import bioframe as bf
from gpn.data import (
    filter_defined, filter_length, load_table, add_flank, get_annotation_features,
    add_jitter, get_promoters, get_random_intervals, union_intervals,
    intersect_intervals, intervals_size
)
import pandas as pd
import polars as pl


def find_positions(interval):
    df = pd.DataFrame(dict(pos=range(interval.start, interval.end)))
    df["chrom"] = interval.chrom
    df.pos += 1  # we'll treat as 1-based
    return df


rule make_positions_promoter:
    input:
        "results/annotation.gtf.gz",
        "results/rmsk_merged.parquet",
        "results/intervals/128/defined.parquet",
    output:
        "results/positions/promoter/positions.parquet",
    run:
        annotation = load_table(input[0])
        annotation = annotation[annotation.chrom.isin(CHROMS)]

        transcripts = annotation[annotation.feature.isin(["mRNA", "transcript"])]
        print(transcripts)
        transcripts = transcripts[
            transcripts['attribute'].str.contains('transcript_biotype "protein_coding"')
        ]
        intervals = get_promoters(transcripts, 1000, 1000)
        print(intervals)

        CDS = bf.merge(annotation.query("feature == 'CDS'")).drop(columns="n_intervals")
        intervals = bf.subtract(intervals, CDS)
        print(intervals)

        repeats = pd.read_parquet(input[1])
        repeats = repeats[repeats.chrom.isin(CHROMS)]
        intervals = bf.subtract(intervals, repeats)
        print(intervals)

        defined_intervals = pd.read_parquet(input[2])
        intervals = intersect_intervals(
            intervals, bf.expand(defined_intervals, pad=-WINDOW_SIZE//2)
        )
        print(intervals)

        intervals = filter_length(intervals, 30)
        intervals = intervals.sort_values(["chrom", "start"])
        print(intervals)

        positions = pd.concat(
            intervals.progress_apply(find_positions, axis=1).values, ignore_index=True
        )
        print(positions)
        positions.to_parquet(output[0], index=False)


rule make_positions_erap2:
    output:
        "results/positions/erap2/positions.parquet",
    run:
        intervals = pd.DataFrame(dict(chrom=["5"], start=[96896864], end=[96897364]))
        positions = pd.concat(
            intervals.progress_apply(find_positions, axis=1).values, ignore_index=True
        )
        print(positions)
        positions.to_parquet(output[0], index=False)


rule make_positions_chrom:
    input:
        f"results/intervals/{config['window_size']}/defined.parquet",
    output:
        "results/positions/{chrom}/positions.parquet",
    run:
        intervals = pd.read_parquet(input[0]).query(f"chrom == '{wildcards.chrom}'")
        intervals = bf.expand(intervals, pad=-config["window_size"]//2)
        intervals = filter_length(intervals, 1)
        positions = pd.concat(
            intervals.progress_apply(find_positions, axis=1).values, ignore_index=True
        )
        print(positions)
        positions.to_parquet(output[0], index=False)


# since introducing torchrun, it seems like it loads the "dataset" (coordinates)
# in each worker, which uses a lot of memory for gigantic chroms. thus, reduced
# dataloader num workers
rule get_logits:
    input:
        "{anything}/positions.parquet",
        "results/msa/{alignment}/{species}/all.zarr",
        "results/checkpoints/{alignment}/{species}/{window_size}/{model}",
    output:
        "{anything}/logits/{alignment,[A-Za-z0-9]+}/{species,[A-Za-z0-9]+}/{window_size,\d+}/{model}.parquet",
    threads:
        workflow.cores
    shell:
        """
        torchrun --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{{print NF}}') -m gpn.msa.inference logits {input[0]} {input[1]} {wildcards.window_size} {input[2]} {output} \
        --per_device_batch_size {config[per_device_batch_size]} --is_file \
        --dataloader_num_workers 4
        """


rule download_jaspar:
    output:
        "results/jaspar.meme",
    shell:
        "wget -O {output} https://jaspar.genereg.net/download/data/2022/CORE/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt"
    

# example:
# snakemake --cores all --use-conda --conda-frontend mamba 
rule run_modisco:
    input:
        "{anything}/positions.parquet",
        "{anything}/logits/{model}.parquet",
        "results/genome.fa.gz",
    output:
        "{anything}/modisco/{model}/results.h5",
    conda:
        "../envs/modisco-lite.yaml"
    script:
        "../scripts/modisco_run.py"


rule plot_modisco:
    input:
        "{anything}/modisco/{model}/results.h5",
        "results/jaspar.meme",
    output:
        directory("{anything}/modisco/{model}/report"),
    conda:
        "../envs/modisco-lite.yaml"
    script:
        "../scripts/modisco_report.py"


rule process_logits:
    input:
        "results/{anything}/positions.parquet",
        "results/{anything}/logits/{model}.parquet",
        "results/genome.fa.gz",
    output:
        "results/{anything}/processed_logits/{model}.parquet",
    threads:
        workflow.cores
    run:
        V1 = pl.read_parquet(input[0])[["chrom", "pos"]]
        V2 = pl.read_parquet(input[1])
        V = V1.hstack(V2)
        assert len(V["chrom"].unique()) == 1
        chrom = V["chrom"][0]
        seq = Genome(input[2])._genome[chrom].upper()
        seq = np.frombuffer(seq.encode("ascii"), dtype="S1")
        V = V.with_columns(ref=seq[V["pos"]-1])
        V = V.with_columns(ref=pl.col("ref").cast(str))
        # sorry, this is horrible, was more elegant in pandas
        V = V.with_columns(
            V.select(ref_logit=(
                pl.when(pl.col("ref") == "A").then(pl.col("A"))
                .when(pl.col("ref") == "C").then(pl.col("C"))
                .when(pl.col("ref") == "G").then(pl.col("G"))
                .when(pl.col("ref") == "T").then(pl.col("T"))
        )))
        V = V.with_columns(
            V[NUCLEOTIDES] - V["ref_logit"]
        )
        V = V.select(["chrom", "pos", "ref"] + NUCLEOTIDES)
        print(V)
        V.write_parquet(output[0])


rule get_llr:
    input:
        "results/{anything}/processed_logits/{model}.parquet",
    output:
        "results/{anything}/llr/{model}.parquet",
    threads:
        workflow.cores
    run:
        V = pl.read_parquet(
            input[0]
        ).melt(
            id_vars=["chrom", "pos", "ref"], value_vars=NUCLEOTIDES,
            variable_name="alt", value_name="score"
        ).sort(["chrom", "pos", "ref"]).filter(pl.col("ref") != pl.col("alt"))
        print(V)
        V.write_parquet(output[0])


ruleorder: logits_merge_chroms > bgzip


rule logits_merge_chroms:
    input:
        expand("results/positions/{chrom}/{{anything}}/{{model}}.tsv.bgz", chrom=CHROMS),
    output:
        "results/positions/merged/{anything}/{model}.tsv.bgz",
    wildcard_constraints:
        anything="processed_logits|probs|llr",
    shell:
        "cat {input} > {output}"


#ruleorder: logits_merge_chroms > process_logits
#
#
#rule logits_merge_chroms:
#    input:
#        expand("results/positions/{chrom}/{{anything}}/{{model}}.parquet", chrom=CHROMS),
#    output:
#        "results/positions/merged/{anything}/{model}.parquet",
#    wildcard_constraints:
#        anything="processed_logits|probs",
#    run:
#        V = pl.concat([pl.read_parquet(path) for path in tqdm(input)])
#        if wildcards.anything == "processed_logits":
#            V = V.select(["chrom", "pos", "ref"] + NUCLEOTIDES)
#        print(V)
#        V.write_parquet(output[0])
#        #V.to_pandas().to_parquet(output[0], index=False)


rule all3:
    input:
        #f"results/positions/merged/processed_logits/{best_model}.tsv.bgz.tbi",
        #f"results/positions/merged/llr/{best_model}.tsv.bgz.tbi",
        #f"results/positions/22/probs_wig/{best_model}/A.bw",
        expand("results/positions/merged/probs_wig/{model}/{nuc}.bw", model=[best_model], nuc=NUCLEOTIDES),


rule make_probs_wig:
    input:
        "results/positions/{chrom}/processed_logits/{model}.parquet",
    output:
        temp(expand("results/positions/{{chrom}}/probs_wig/{{model}}/{nuc}.wig", nuc=NUCLEOTIDES)),
    wildcard_constraints:
        chrom="|".join(CHROMS)
    threads: workflow.cores // 4
    run:
        V = pl.read_parquet(input[0])
        V = V.with_columns(
            pl.DataFrame(softmax(V.select(NUCLEOTIDES), axis=1), schema=NUCLEOTIDES)
        )
        V = (
            V.with_columns(
                entropy=entropy(V.select(NUCLEOTIDES), base=2, axis=1)
            )
            .with_columns(
                (pl.col(NUCLEOTIDES) * (2 - pl.col("entropy")))
            )
        )
        for nuc, path in zip(NUCLEOTIDES, output):
            with open(path, 'w') as f:
                f.write(f"variableStep chrom=chr{wildcards.chrom}\n")
            with open(path, 'ab') as f:
                V.select(["pos", nuc]).write_csv(
                    f, separator="\t", include_header=False, float_precision=2,
                )


rule wigToBigWig:
    input:
        "{anything}.wig",
        "results/chrom.sizes",
    output:
        temp("{anything}.bw"),
    shell:
        "wigToBigWig {input} {output} -keepAllChromosomes -fixedSummaries"


# didn't manage to install conda version, so downloaded binary
# wget https://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/bigWigCat
# chmod +x bigWigCat
rule bigWigCat:
    input:
        expand("results/positions/{chrom}/probs_wig/{{model}}/{{nuc}}.bw", chrom=CHROMS),
    output:
        "results/positions/merged/probs_wig/{model}/{nuc}.bw",
    shell:
        "./bigWigCat {output} {input}"


rule make_chrom_sizes:
    input:
        "results/genome.fa.gz",
    output:
        "results/chrom.sizes",
    run:
        intervals = Genome(input[0], subset_chroms=CHROMS).get_all_intervals()
        intervals.chrom = "chr" + intervals.chrom
        intervals.to_csv(
            output[0], sep="\t", index=False, header=False,
            columns=["chrom", "end"],
        )


rule convert_bed_to_bigwig:
    input:
        "{anything}.bed",
        "results/chrom.sizes",
    output:
        "{anything}.bw"
    shell:
        # using a local download because conda version didn't work
        "./bedGraphToBigWig {input} {output}"


rule bigwig_done:
    input:
        expand("{{anything}}/{nuc}.bw", nuc=NUCLEOTIDES),
    output:
        touch("{anything}/bigwig.done"),
