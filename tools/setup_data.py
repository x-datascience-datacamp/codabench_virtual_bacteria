"""Build the data release files read by the ``BacterialExpression`` dataset.

Release files, for each ``mode`` in ``debug`` / ``train`` / ``full``::

    read_count_<mode>.parquet
    feature_annotation_<mode>.csv
    condition_annotation_<mode>.csv

``--release SRC`` converts the organizers' CSV tables found in ``SRC``
(``read_count_<mode>.csv`` and the two annotation CSVs) into ``OUT/public``
(``debug`` and ``train``, split statuses dropped) and ``OUT/private``
(``full``, with the real statuses), ready to upload to the public and the
restricted Zenodo records. Read counts are streamed, so the multi-GB tables
never sit in memory.

``--demo`` writes a tiny synthetic ``debug`` and ``full`` release, genomes
included, straight into the benchmark data folder, so ``prepare`` runs
offline (CI, smoke tests).
"""

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from benchmark_utils.split import (  # noqa: E402
    STATUS_COLS, assign_regions, simulate_status,
)

DEMO_DIR = Path("benchmark/data/bacteria_raw")
RELEASES = {"debug": "public", "train": "public", "full": "private"}


def convert_read_counts(src, dest, keep_status):
    """Stream a read-count CSV into a zstd Parquet file.

    String columns are dictionary-encoded, so they load as categories.
    """
    reader = pv.open_csv(src, read_options=pv.ReadOptions(block_size=1 << 26))
    drop = [] if keep_status else STATUS_COLS
    writer = None
    for batch in reader:
        table = pa.Table.from_batches([batch])
        table = table.drop_columns(
            [c for c in drop if c in table.schema.names])
        table = pa.table({
            name: pc.dictionary_encode(col) if pa.types.is_string(col.type)
            else col for name, col in zip(table.column_names, table.columns)
        })
        if writer is None:
            writer = pq.ParquetWriter(dest, table.schema, compression="zstd")
        writer.write_table(table)
    writer.close()


def make_release(src, out):
    for mode, access in RELEASES.items():
        keep_status = access == "private"
        dest = out / access
        dest.mkdir(parents=True, exist_ok=True)
        for kind in ("feature_annotation", "condition_annotation"):
            df = pd.read_csv(src / f"{kind}_{mode}.csv")
            if not keep_status:
                df = df.drop(columns=STATUS_COLS, errors="ignore")
            df.to_csv(dest / f"{kind}_{mode}.csv", index=False)
        convert_read_counts(src / f"read_count_{mode}.csv",
                            dest / f"read_count_{mode}.parquet", keep_status)
        print(f"[setup_data] wrote {mode} release to {dest}")


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

    obs = cond_df.merge(feat_df, on=["species", "genome_id"])
    log_mu = 4.0 + obs["log_length_ratio"] + obs["log_size_factor"]
    mu = np.exp(log_mu + rng.normal(0, 0.2, len(obs)))
    obs["readcount"] = np.round(
        np.maximum(0.0, rng.normal(mu, np.sqrt(mu + 0.2 * mu**2))), 1)
    obs = obs[["species", "genome_id", "feature_id", "run_accession",
               "readcount"]]

    # Real-like statuses for the full release, drawn with another seed
    # than the simulated debug split.
    feat_g = simulate_status(feat_df, seed=seed + 1)
    cond_g = simulate_status(cond_df, seed=seed + 1)
    keys = ["genome_id", "feature_id", "run_accession"]
    obs_g = assign_regions(obs.astype({k: "category" for k in keys}),
                           feat_g, cond_g)

    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out / "genomes.zip", "w") as z:
        for gid in genomes["genome_id"]:
            seq = "".join(rng.choice(list("acgt"), 600))
            z.writestr(f"genomes/{gid}.gbff",
                       f"LOCUS       contig_1{len(seq):>21} bp    DNA     "
                       "linear   BCT 01-JAN-2026\n"
                       f"ORIGIN\n        1 {seq}\n//\n")
    for mode, (o, f, c) in {"debug": (obs, feat_df, cond_df),
                            "full": (obs_g, feat_g, cond_g)}.items():
        o.to_parquet(out / f"read_count_{mode}.parquet")
        f.to_csv(out / f"feature_annotation_{mode}.csv", index=False)
        c.to_csv(out / f"condition_annotation_{mode}.csv", index=False)
    print(f"[setup_data] wrote demo release to {out}, regions: "
          f"{obs_g['status'].value_counts().to_dict()}")


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
