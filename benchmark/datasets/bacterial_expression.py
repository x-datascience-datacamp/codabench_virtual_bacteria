"""Multi-species bacterial transcriptomics dataset, A/B/C/D regions.

Each observation's ``status`` puts it in training (``Train``) or in a held-out
region of the evaluation (``eval_*``) or test (``test_*``) split. The two
splits are drawn with the same criteria:

- A: gene and run both seen in training, never together;
- B: new run;
- C: new gene;
- D: new gene in a new run.

"Seen" means seen in the data the model fits on. ``prepare`` drops the
held-out observations whose region this data contradicts, such as an A
observation whose gene it never contains.

``prepare`` downloads the release files of ``mode`` and the genomes from
Zenodo, then writes the regions::

    benchopt prepare benchmark -d "BacterialExpression[mode=train]"

- ``debug``: small subsample (one genome), scored on its eval regions;
- ``train``: all the public data, fits on ``Train``, scored on the eval
  regions;
- ``full``: fits on ``Train`` and eval, scored on the test regions, each
  halved into a public and a hidden private part (``is_private``). Its
  record is restricted: set ``BACTERIA_DATA_TOKEN`` to download it.

Raw release files go to ``get_data_path("bacteria_raw")``::

    read_count_<mode>.parquet           observations, ``readcount``
                                        and ``status``
    feature_annotation[_debug].csv      gene table, ``log_length_ratio``
    condition_annotation[_debug].csv    run table, ``log_size_factor`` and
                                        the curated conditions (medium,
                                        temperature, pH, perturbation...)
    genomes.zip                         ``genomes/<genome_id>.gbff``
    genomes_debug.zip                   the ``debug`` genome only

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

import numpy as np
import pandas as pd
from benchopt import BaseDataset
from benchopt.config import get_data_path

from benchmark_utils.download import download_zenodo

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

REGIONS = ("A", "B", "C", "D")
GENE_KEYS = ["genome_id", "feature_id"]
RUN_KEYS = ["genome_id", "run_accession"]
# Private halves: by observation for A, by run for B, by gene for C and D, so
# a new gene falls in the same half in C and D.
HALF_KEYS = {"A": GENE_KEYS + ["run_accession"], "B": RUN_KEYS,
             "C": GENE_KEYS, "D": GENE_KEYS}

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
        raw = get_data_path("bacteria_raw")
        suffix = "_debug" if self.mode == "debug" else ""
        feat_name = f"feature_annotation{suffix}.csv"
        cond_name = f"condition_annotation{suffix}.csv"
        counts = f"read_count_{self.mode}.parquet"
        download_zenodo(PUBLIC_RECORD, [feat_name, cond_name], raw,
                        PUBLIC_TOKEN)
        if self.mode == "full":
            download_zenodo(PRIVATE_RECORD, [counts], raw,
                            os.environ.get(TOKEN_ENV))
        else:
            download_zenodo(PUBLIC_RECORD, [counts], raw, PUBLIC_TOKEN)

        feat_df = pd.read_csv(raw / feat_name)
        cond_df = pd.read_csv(raw / cond_name)
        obs = pd.read_parquet(raw / counts).astype(
            {k: "category" for k in GENE_KEYS + RUN_KEYS})
        obs = attach(obs, feat_df, GENE_KEYS, ["log_length_ratio"])
        obs = attach(obs, cond_df, RUN_KEYS, ["log_size_factor"])

        # Locally the eval regions score the model; on the platform it also
        # fits on them and the test regions score it.
        held_out = "test" if self.mode == "full" else "eval"
        parts = {"Train": "train",
                 **{f"{held_out}_{r}": r for r in REGIONS}}
        if self.mode == "full":
            parts.update({f"eval_{r}": "train" for r in REGIONS})
        part = obs.pop("status").map(parts).to_numpy()
        fit = part == "train"
        gene, run = pair_codes(obs, GENE_KEYS), pair_codes(obs, RUN_KEYS)
        new_gene, new_run = ~np.isin(gene, gene[fit]), ~np.isin(run, run[fit])
        region = np.select([new_gene & new_run, new_gene, new_run],
                           ["D", "C", "B"], "A")
        part = np.where(fit | (part == region), part, None)

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
        for name in ("train",) + REGIONS:
            obs[part == name].reset_index(drop=True).to_parquet(
                root / f"{name}.parquet")

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

        # On the platform, the public half of each region under its name,
        # the private one as ``<region>_private``.
        test_sets = {}
        for name in REGIONS:
            df = pd.read_parquet(root / f"{name}.parquet")
            private = (is_private(df, name) if self.mode == "full"
                       else np.zeros(len(df), dtype=bool))
            halves = [(name, df[~private])]
            if private.any():
                halves.append((f"{name}_private", df[private]))
            for key, part in halves:
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


def attach(obs, table, keys, cols):
    """Left-join ``cols`` of ``table`` onto ``obs``, matching category codes.

    Casting the keys to the observation dtypes keeps the join on integer codes
    instead of strings, which matters for the ~8e7-row releases.
    """
    table = table[keys + cols].astype({k: obs[k].dtype for k in keys})
    # Keys absent from obs cast to NaN. A duplicated key would duplicate its
    # observations.
    return obs.merge(table.dropna(subset=keys), on=keys, how="left",
                     validate="many_to_one")


def pair_codes(df, keys):
    """One integer per row for the pair of categorical ``keys``."""
    first, second = (df[k].cat.codes.to_numpy().astype(np.int64)
                     for k in keys)
    return first * (len(df[keys[1]].cat.categories) + 1) + second


def is_private(df, region):
    """Whether each observation of ``region`` falls in its private half."""
    keys = df[HALF_KEYS[region]]
    return pd.util.hash_pandas_object(keys, index=False).to_numpy() % 2 == 1
