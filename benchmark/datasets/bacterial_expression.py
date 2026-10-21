"""Multi-species bacterial transcriptomics dataset, A/B/C/D test regions.

``prepare`` downloads the release files of ``mode`` from Zenodo, then writes
the train / A / B / C / D split (``benchmark_utils.split``)::

    benchopt prepare benchmark -d "BacterialExpression[mode=train]"

- ``debug``: small subsample, regions simulated from the training data;
- ``train``: all training data, regions simulated the same way;
- ``full``: evaluation data with the real regions. Its record is restricted:
  set ``BACTERIA_DATA_TOKEN`` to download it.

Raw release files go to ``get_data_path("bacteria_raw")``::

    read_count_<tag>.parquet          (gene x run) observations, ``readcount``
    feature_annotation_<tag>.csv      gene table, ``log_length_ratio``
    condition_annotation_<tag>.csv    run table, ``log_size_factor``

with ``tag`` = ``debug`` / ``train`` / ``global``. The prepared layout, under
``get_data_path("bacteria") / mode`` (on the competition worker the image sets
``BENCHOPT_DATA_HOME`` and the ``full`` layout is staged there, read-only)::

    {train, A, B, C, D}.parquet       observations + offsets + ``readcount``
    feature_annotation.csv
    condition_annotation.csv
    genomes/                          optional GenBank genomes
"""

import pandas as pd
from benchopt import BaseDataset
from benchopt.config import get_data_path

from benchmark_utils.download import download_zenodo
from benchmark_utils.split import REGIONS, split_release

# Public record: debug + train releases. Restricted record: global release.
PUBLIC_RECORD = "23105231"
PRIVATE_RECORD = "23105231"
TOKEN_ENV = "BACTERIA_DATA_TOKEN"
RELEASE_TAG = {"debug": "debug", "train": "train", "full": "global"}

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
CONDITION_METADATA_FEATURES = ["species", "prj_accession"]


class Dataset(BaseDataset):

    name = "BacterialExpression"

    parameters = {
        "mode": ["debug"],
    }

    # The split only depends on ``mode``; simulated regions use a fixed seed.
    prepare_cache_ignore = ("base_seed",)

    def prepare(self):
        tag = RELEASE_TAG[self.mode]
        names = [f"read_count_{tag}.parquet", f"feature_annotation_{tag}.csv",
                 f"condition_annotation_{tag}.csv"]
        raw = get_data_path("bacteria_raw")
        record = PRIVATE_RECORD if self.mode == "full" else PUBLIC_RECORD
        download_zenodo(record, names, raw, TOKEN_ENV)

        frames, feat_df, cond_df = split_release(
            pd.read_parquet(raw / names[0]),
            pd.read_csv(raw / names[1]),
            pd.read_csv(raw / names[2]),
            simulate=self.mode != "full",
        )
        root = get_data_path("bacteria") / self.mode
        root.mkdir(parents=True, exist_ok=True)
        feat_df.to_csv(root / "feature_annotation.csv", index=False)
        cond_df.to_csv(root / "condition_annotation.csv", index=False)
        for name, df in frames.items():
            df.to_parquet(root / f"{name}.parquet")

    def get_data(self):
        root = get_data_path("bacteria") / self.mode
        # ``benchopt run`` does not call ``prepare``: run it on first use.
        if not (root / "train.parquet").exists():
            self.prepare()

        test_sets = {}
        for name in REGIONS:
            df = pd.read_parquet(root / f"{name}.parquet")
            test_sets[name] = (df.drop(columns="readcount"),
                               df["readcount"].to_numpy(dtype=float))
        cond_df = pd.read_csv(root / "condition_annotation.csv")
        genomes_dir = root / "genomes"
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
            condition_metadata_features=CONDITION_METADATA_FEATURES,
        )
