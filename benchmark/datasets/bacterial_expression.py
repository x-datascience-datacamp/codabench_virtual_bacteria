"""Multi-species bacterial transcriptomics dataset, A/B/C/D test regions.

Each region is scored on a public and a hidden private half
(``benchmark_utils.split.is_private``).

``prepare`` downloads the release files of ``mode`` and the genomes from
Zenodo, then writes the train / A / B / C / D split
(``benchmark_utils.split``)::

    benchopt prepare benchmark -d "BacterialExpression[mode=train]"

- ``debug``: small subsample, regions simulated from the training data;
- ``train``: all training data, regions simulated the same way;
- ``full``: evaluation data with the real regions. Its record is restricted:
  set ``BACTERIA_DATA_TOKEN`` to download it.

Raw release files go to ``get_data_path("bacteria_raw")``::

    read_count_<mode>.parquet          (gene x run) observations, ``readcount``
    feature_annotation_<mode>.csv      gene table, ``log_length_ratio``
    condition_annotation_<mode>.csv    run table, ``log_size_factor`` and
                                       the curated conditions (medium,
                                       temperature, pH, perturbation...)
    genomes.zip                        ``genomes/<genome_id>.gbff``
    genomes_debug.zip                  the ``debug`` genome only

The prepared layout, under
``get_data_path("bacteria") / mode`` (on the competition worker the image sets
``BENCHOPT_DATA_HOME`` and the ``full`` layout is staged there, read-only)::

    {train, A, B, C, D}.parquet       observations + offsets + ``readcount``
    feature_annotation.csv
    condition_annotation.csv

and the GenBank genomes, shared by every mode, in
``get_data_path("bacteria") / "genomes"``: one ``<genome_id>.gbff`` per
genome. The gene table's ``Chr`` is a record's VERSION (RefSeq) or LOCUS
name (``contig_<i>``, repeated across genomes).
"""

import os
import shutil
import zipfile

import pandas as pd
from benchopt import BaseDataset
from benchopt.config import get_data_path

from benchmark_utils.download import download_zenodo
from benchmark_utils.split import REGIONS, is_private, split_release

# Public record: debug + train releases, genomes and curated conditions,
# read with the token of its share link while it is a draft. Restricted
# record: full release.
PUBLIC_RECORD = "23751267"
PUBLIC_TOKEN = (
    "eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjlhNTk5ZDg1LTIzNWItNDQ1NS1hYzkyLWJhNzFjM"
    "zVkYTk4NiIsImRhdGEiOnt9LCJyYW5kb20iOiI2YWFkN2Y2ZGI3NTRiOWZkNDAyZDM2YWU"
    "zZGNlNmNmMiJ9.hMs5tbqvZArD8nfn0Cirx7dLFU99bLy2bH2lBNlFC5c6e0laOWZ2Qdyq"
    "vZZ95A6vRUztcTL2C0203vk5f39LBg"
)
PRIVATE_RECORD = "23751273"
TOKEN_ENV = "BACTERIA_DATA_TOKEN"
GENOMES = "genomes.zip"

ID_FEATURES = ["genome_id", "feature_id", "run_accession"]
NUMERIC_FEATURES = ["log_length_ratio", "log_size_factor"]
GENE_METADATA_FEATURES = [
    "species",
    "ref_genome_id",
    "ref_feature_id",
    "Chr",
    "Start",
    "End",
    "Strand",
    "Length",
    "homology",
    "on_contig_edge",
]


class Dataset(BaseDataset):

    name = "BacterialExpression"

    parameters = {
        "mode": ["debug"],
    }

    def prepare(self):
        names = [f"read_count_{self.mode}.parquet",
                 f"feature_annotation_{self.mode}.csv",
                 f"condition_annotation_{self.mode}.csv"]
        raw = get_data_path("bacteria_raw")
        if self.mode == "full":
            download_zenodo(PRIVATE_RECORD, names, raw,
                            os.environ.get(TOKEN_ENV))
        else:
            download_zenodo(PUBLIC_RECORD, names, raw, PUBLIC_TOKEN)

        frames, feat_df, cond_df = split_release(
            pd.read_parquet(raw / names[0]),
            pd.read_csv(raw / names[1]),
            pd.read_csv(raw / names[2]),
            simulate=self.mode != "full",
        )
        root = get_data_path("bacteria") / self.mode
        root.mkdir(parents=True, exist_ok=True)
        genomes = root.parent / "genomes"
        # debug only needs its own genome, not the 750 MB archive.
        zip_name = "genomes_debug.zip" if self.mode == "debug" else GENOMES
        if not all((genomes / f"{g}.gbff").exists()
                   for g in feat_df["genome_id"].unique()):
            download_zenodo(PUBLIC_RECORD, [zip_name], raw, PUBLIC_TOKEN)
            tmp = root.parent / "genomes.part"
            shutil.rmtree(tmp, ignore_errors=True)
            zipfile.ZipFile(raw / zip_name).extractall(tmp)
            genomes.mkdir(exist_ok=True)
            for path in (tmp / "genomes").iterdir():
                path.replace(genomes / path.name)
            shutil.rmtree(tmp)
        feat_df.to_csv(root / "feature_annotation.csv", index=False)
        cond_df.to_csv(root / "condition_annotation.csv", index=False)
        for name, df in frames.items():
            df.to_parquet(root / f"{name}.parquet")

    def get_data(self):
        root = get_data_path("bacteria") / self.mode
        if not (root / "train.parquet").exists():
            # prepare's cache does not check its output: --force if the
            # data was deleted or BENCHOPT_DATA_HOME changed.
            raise SystemExit(
                f"[BacterialExpression] no data at {root}. Run `benchopt "
                "prepare benchmark -d "
                f"\"BacterialExpression[mode={self.mode}]\" --force`."
            )

        # The public half of each region under its name, the private one
        # as ``<region>_private``.
        test_sets = {}
        for name in REGIONS:
            df = pd.read_parquet(root / f"{name}.parquet")
            private = is_private(df, name)
            for key, part in ((name, df[~private]),
                              (f"{name}_private", df[private])):
                part = part.reset_index(drop=True)
                test_sets[key] = (part.drop(columns="readcount"),
                                  part["readcount"].to_numpy(dtype=float))
        cond_df = pd.read_csv(root / "condition_annotation.csv")
        genomes_dir = root.parent / "genomes"
        return dict(
            train_df=pd.read_parquet(root / "train.parquet"),
            samples_df=cond_df,
            genomes_dir=genomes_dir if genomes_dir.is_dir() else None,
            test_sets=test_sets,
            feature_annotations=pd.read_csv(
                root / "feature_annotation.csv"),
            condition_annotations=cond_df,
            id_features=ID_FEATURES,
            numeric_features=NUMERIC_FEATURES,
            gene_metadata_features=GENE_METADATA_FEATURES,
            condition_metadata_features=[
                c for c in cond_df.columns
                if c not in ID_FEATURES + NUMERIC_FEATURES],
        )
