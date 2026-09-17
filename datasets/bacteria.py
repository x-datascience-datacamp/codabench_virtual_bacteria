"""Real bacterial gene-expression dataset (loads the pre-split layout).

Reads the files produced by ``tools/setup_data.py`` under the benchmark data
path (``benchopt``'s ``get_data_path``; on the competition worker the image
sets ``BENCHOPT_DATA_HOME`` and the phase stages the data there, read-only)::

    <root>/gene_expression/
      train_data.tsv          observations incl. ``counts``
      samples.tsv             experimental metadata (join key ``Run``)
      genomes/<reference>.gb  GenBank genome per species
      public/{test_data.tsv, labels.csv}     new gene x new condition (known)
      private/{test_data.tsv, labels.csv}    held-out species (col ``group``:
                                             near/far) — sealed, absent from
                                             the public starting kit

The ``public`` labels ship publicly (local dev score); ``private`` is added to
the evaluation only where its files are present (the server). Feature columns
follow the getting-started notebook.
"""

import pandas as pd
from benchopt import BaseDataset
from benchopt.config import get_data_path

NUMERIC_FEATURES = ["Length", "log_length_ratio", "log_size_factor",
                    "pH", "Temperature (C)", "Growth Rate (1/hr)"]
CATEGORICAL_FEATURES = ["Base Media", "Culture Type"]


class Dataset(BaseDataset):

    name = "Bacteria"

    requirements = ["pip::biopython"]

    def get_data(self):
        root = get_data_path("gene_expression")
        if not root.exists():
            raise SystemExit(
                f"[Bacteria] no data at {root}. Generate it with "
                "`python tools/setup_data.py` (real data) or "
                "`python tools/setup_data.py --demo` (synthetic)."
            )

        samples_df = pd.read_csv(root / "samples.tsv", sep="\t")
        genomes_dir = root / "genomes"
        genomes_dir = genomes_dir if genomes_dir.exists() else None
        train_df = pd.read_csv(root / "train_data.tsv", sep="\t")

        test_sets = {}
        pub_dir = root / "public"
        pub_df = pd.read_csv(pub_dir / "test_data.tsv", sep="\t")
        pub_counts = pd.read_csv(pub_dir / "labels.csv")["counts"].to_numpy()
        test_sets["public"] = (pub_df, pub_counts)

        priv_dir = root / "private"
        if priv_dir.exists():
            priv_df = pd.read_csv(priv_dir / "test_data.tsv", sep="\t")
            priv_lab = pd.read_csv(priv_dir / "labels.csv")
            priv_counts = priv_lab["counts"].to_numpy()
            test_sets["private"] = (priv_df, priv_counts)
            group = priv_lab["group"].to_numpy()
            for name in ("near", "far"):
                mask = group == name
                if mask.any():
                    test_sets[f"private_{name}"] = (
                        priv_df[mask].reset_index(drop=True),
                        priv_counts[mask],
                    )

        return dict(
            train_df=train_df,
            samples_df=samples_df,
            genomes_dir=genomes_dir,
            test_sets=test_sets,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )
