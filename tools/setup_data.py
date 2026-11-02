"""Build the data release files read by the ``BacterialExpression`` dataset.

Release files::

    read_count_<mode>.parquet            mode in debug / train / full
    feature_annotation[_debug].csv
    condition_annotation[_debug].csv

and the GenBank genomes, ``genomes.zip`` and ``genomes_debug.zip`` (the
``debug`` genome only).

``--release SRC`` converts the organizers' files found in ``SRC``
(``read_count_{debug,public,private}.csv.gz``, ``feature_annotation.csv``,
``condition_annotation.csv``, ``genomes.zip``, the curated conditions
``experiment.zip`` and ``run_to_experiment.csv``) into ``OUT/public`` (the
``debug`` and ``train`` counts without the test rows, the annotations, the
genomes and ``experiment.zip``) and ``OUT/private`` (the ``full`` counts),
ready to upload to the public and the restricted Zenodo records. Each
observation keeps only its region ``status`` (``Train``, ``eval_<region>``,
``test_<region>``). Read counts are streamed, so the multi-GB tables never
sit in memory.

``--demo`` writes a tiny synthetic ``debug`` and ``full`` release, genomes
included, straight into the benchmark data folder, so ``prepare`` runs
offline (CI, smoke tests).
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
import pyarrow.parquet as pq

DEMO_DIR = Path("benchmark/data/bacteria_raw")
# Run table column: (section, field) of an experiment in experiment.zip.
CONDITION_FIELDS = {
    "is_wild_type": ("strain", "is_wild_type"),
    "medium_category": ("growth_medium", "medium_category"),
    "medium_name": ("growth_medium", "base_medium_name"),
    "temperature_celsius": ("cultivation", "temperature_celsius"),
    "ph": ("cultivation", "ph"),
    "oxygen_condition": ("cultivation", "oxygen_condition"),
    "agitation_rpm": ("cultivation", "agitation_rpm"),
    "growth_mode": ("cultivation", "growth_mode"),
    "growth_phase": ("sampling_point", "growth_phase"),
    "od600": ("sampling_point", "optical_density_od600"),
    "has_perturbation": ("perturbation", "has_perturbation"),
    "perturbation_category": ("perturbation", "perturbation_category"),
    "perturbation_compound": ("perturbation", "stressor_or_compound"),
}
# Release mode: (organizers' file, record).
RELEASES = {"debug": ("debug", "public"), "train": ("public", "public"),
            "full": ("private", "private")}
COUNT_COLS = ["species", "genome_id", "feature_id", "run_accession",
              "readcount", "status"]


def convert_read_counts(src, dest, keep_test):
    """Stream a read-count CSV into a zstd Parquet file.

    String columns are dictionary-encoded, so they load as categories.
    """
    reader = pv.open_csv(
        src, read_options=pv.ReadOptions(block_size=1 << 26),
        convert_options=pv.ConvertOptions(
            include_columns=COUNT_COLS,
            column_types={"readcount": pa.float64()}))
    writer = None
    for batch in reader:
        table = pa.Table.from_batches([batch])
        if not keep_test:
            table = table.filter(
                pc.invert(pc.starts_with(table["status"], "test_")))
        table = pa.table({
            name: pc.dictionary_encode(col) if pa.types.is_string(col.type)
            else col for name, col in zip(table.column_names, table.columns)
        })
        if writer is None:
            writer = pq.ParquetWriter(dest, table.schema, compression="zstd")
        writer.write_table(table)
    writer.close()


def make_release(src, out):
    for mode, (name, access) in RELEASES.items():
        dest = out / access
        dest.mkdir(parents=True, exist_ok=True)
        convert_read_counts(src / f"read_count_{name}.csv.gz",
                            dest / f"read_count_{mode}.parquet",
                            keep_test=access == "private")
        print(f"[setup_data] wrote {mode} counts to {dest}")
    feat_df = pd.read_csv(src / "feature_annotation.csv").drop(
        columns="status")
    cond_df = add_conditions(
        pd.read_csv(src / "condition_annotation.csv").drop(columns="status"),
        src)
    debug_ids = pd.read_parquet(out / "public" / "read_count_debug.parquet",
                                columns=["genome_id"])["genome_id"].unique()
    for suffix, keep in (("", slice(None)),
                         ("_debug", cond_df["genome_id"].isin(debug_ids))):
        cond_df[keep].to_csv(
            out / "public" / f"condition_annotation{suffix}.csv", index=False)
    feat_df[feat_df["genome_id"].isin(debug_ids)].to_csv(
        out / "public" / "feature_annotation_debug.csv", index=False)
    feat_df.to_csv(out / "public" / "feature_annotation.csv", index=False)
    write_genomes(src, out / "public", debug_ids)
    shutil.copy(src / "experiment.zip", out / "public")


def add_conditions(cond_df, src):
    """Join the curated conditions of each run, ``unknown`` left empty.

    ``experiment.zip`` holds one JSON per BioProject, keyed by experiment
    accession, and ``run_to_experiment.csv`` maps the runs to them. An
    experiment listed in several BioProjects takes its run's.
    """
    rows = []
    with zipfile.ZipFile(src / "experiment.zip") as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            project = json.loads(z.read(name))
            for e in project["experiments"]:
                rows.append({
                    "experiment_accession": e["experiment_accession"],
                    "json_prj": project["bioproject_id"],
                    "condition_confidence": e["confidence_score"],
                    **{col: e[sec][key]
                       for col, (sec, key) in CONDITION_FIELDS.items()},
                })
    runs = pd.read_csv(src / "run_to_experiment.csv",
                       usecols=["run_accession", "experiment_accession"])
    meta = (cond_df[["run_accession", "prj_accession"]]
            .merge(runs.drop_duplicates("run_accession"), how="left")
            .merge(pd.DataFrame(rows).replace("unknown", None), how="left"))
    meta = (meta.assign(other=meta["json_prj"] != meta["prj_accession"])
            .sort_values("other", kind="stable")
            .drop_duplicates("run_accession")
            .drop(columns=["prj_accession", "json_prj", "other"]))
    return cond_df.merge(meta, on="run_accession", how="left",
                         validate="one_to_one")


def write_genomes(src, dest, ids):
    """Copy ``genomes.zip`` and extract the ``ids`` genomes to a zip."""
    shutil.copy(src / "genomes.zip", dest)
    with zipfile.ZipFile(src / "genomes.zip") as full, zipfile.ZipFile(
            dest / "genomes_debug.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for gid in ids:
            name = f"genomes/{gid}.gbff"
            z.writestr(name, full.read(name))


def make_demo(out, seed=0):
    """Three genomes x 40 genes, 80 runs, NB-like counts."""
    rng = np.random.default_rng(seed)
    species = ["Escherichia_coli", "Pseudomonas_putida", "Bacillus_subtilis"]
    genomes = pd.DataFrame({
        "species": species,
        "genome_id": [f"GCF_{i:09d}.1" for i in range(1, 4)],
    })
    length = rng.integers(300, 2500, 3 * 40)
    feat_df = genomes.loc[np.repeat(np.arange(3), 40)].reset_index(drop=True)
    feat_df["feature_id"] = [f"cds-WP_{i:05d}" for i in range(len(feat_df))]
    feat_df["Length"] = length.astype(float)
    feat_df["log_length_ratio"] = np.log(length) - np.log(length).mean()
    cond_df = genomes.loc[rng.integers(0, 3, 80)].reset_index(drop=True)
    cond_df["prj_accession"] = "PRJNA001"
    cond_df["run_accession"] = [f"SRR{i:07d}" for i in range(80)]
    cond_df["log_size_factor"] = rng.normal(0.0, 0.3, 80)
    cond_df["medium_category"] = rng.choice(
        ["complex_rich", "minimal_defined", None], 80)
    cond_df["temperature_celsius"] = rng.choice([30.0, 37.0, np.nan], 80)

    obs = cond_df.merge(feat_df, on=["species", "genome_id"])
    log_mu = 4.0 + obs["log_length_ratio"] + obs["log_size_factor"]
    mu = np.exp(log_mu + rng.normal(0, 0.2, len(obs)))
    obs["readcount"] = np.round(
        np.maximum(0.0, rng.normal(mu, np.sqrt(mu + 0.2 * mu**2))), 1)

    # Held-out level of each gene and run (0 train, 1 eval, 2 test), and of
    # each pair of seen ones (region A).
    p = [0.7, 0.15, 0.15]
    gene = pd.Series(rng.choice(3, len(feat_df), p=p), feat_df["feature_id"])
    run = pd.Series(rng.choice(3, len(cond_df), p=p), cond_df["run_accession"])
    g = gene[obs["feature_id"]].to_numpy()
    r = run[obs["run_accession"]].to_numpy()
    top = np.maximum(g, r)
    level = np.where(top > 0, top, rng.choice(3, len(obs), p=p))
    letter = np.select([top == 0, g < top, r < top], ["A", "B", "C"], "D")
    obs["status"] = np.where(
        level == 0, "Train",
        np.array(["", "eval_", "test_"])[level] + letter)
    obs = obs[COUNT_COLS]

    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out / "genomes.zip", "w") as z:
        for gid in genomes["genome_id"]:
            seq = "".join(rng.choice(list("acgt"), 600))
            z.writestr(f"genomes/{gid}.gbff",
                       f"LOCUS       contig_1{len(seq):>21} bp    DNA     "
                       "linear   BCT 01-JAN-2026\n"
                       f"ORIGIN\n        1 {seq}\n//\n")
    shutil.copy(out / "genomes.zip", out / "genomes_debug.zip")
    obs[~obs["status"].str.startswith("test_")].to_parquet(
        out / "read_count_debug.parquet")
    obs.to_parquet(out / "read_count_full.parquet")
    for suffix in ("", "_debug"):
        feat_df.to_csv(out / f"feature_annotation{suffix}.csv", index=False)
        cond_df.to_csv(out / f"condition_annotation{suffix}.csv", index=False)
    print(f"[setup_data] wrote demo release to {out}, regions: "
          f"{obs['status'].value_counts().to_dict()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--release", type=Path, metavar="SRC",
                       help="folder with the organizers' CSV tables")
    group.add_argument("--demo", action="store_true",
                       help="write a tiny synthetic release")
    parser.add_argument("--out", type=Path, default=None,
                        help="output folder (default: release/ or the "
                        "benchmark data folder for --demo)")
    args = parser.parse_args()

    if args.demo:
        make_demo(args.out or DEMO_DIR)
    else:
        make_release(args.release, args.out or Path("release"))


if __name__ == "__main__":
    main()
