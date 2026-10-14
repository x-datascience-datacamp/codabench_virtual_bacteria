"""Multi-species bacterial transcriptomics dataset with multi-quadrant splits.

Reads the files produced by ``tools/setup_data.py`` under the benchmark data
path (``benchopt``'s ``get_data_path("gene_expression")``)::

    <root>/gene_expression/
      train_data.tsv                 training observations with ``readcount``
      feature_annotations.tsv        full gene metadata table
      condition_annotations.tsv      full condition (sample) metadata table
      C/{test_data.tsv, labels.csv}  unseen (gene x condition) pairing
      A/{test_data.tsv, labels.csv}  unseen genes
      B/{test_data.tsv, labels.csv}  unseen conditions
      D/{test_data.tsv, labels.csv}  unseen genes x unseen conditions
      genomes/                       GenBank / FASTA genomes (optional)

Returns the observation tables and test sets to ``Objective.set_data``.
"""

from pathlib import Path
import pandas as pd
from benchopt import BaseDataset
from benchopt.config import get_data_path

ID_FEATURES = [
    "genome_id", "feature_id", "run_accession", "gene_uid", "condition_uid"
]
NUMERIC_FEATURES = ["log_length_ratio", "log_size_factor"]
GENE_METADATA_FEATURES = [
    "species", "ref_genome_id", "ref_feature_id", "Chr", "Start", "End",
    "Strand", "Length", "homology", "on_contig_edge"
]
CONDITION_METADATA_FEATURES = ["species", "prj_accession"]


class Dataset(BaseDataset):

    name = "BacterialExpression"

    requirements = ["pandas", "numpy", "scipy"]

    def get_data(self):
        root = get_data_path("gene_expression")
        if not root.exists():
            raise SystemExit(
                f"[BacterialExpression] no data at {root}. Generate it with "
                "`python tools/setup_data.py` (real data) or "
                "`python tools/setup_data.py --demo` (synthetic)."
            )

        train_path = root / "train_data.tsv"
        if not train_path.exists():
            raise SystemExit(
                f"[BacterialExpression] missing {train_path}. "
                "Run `python tools/setup_data.py` first."
            )
        train_df = pd.read_csv(train_path, sep="\t")

        cond_path = root / "condition_annotations.tsv"
        samples_df = pd.read_csv(cond_path, sep="\t") if cond_path.exists() else None

        feat_path = root / "feature_annotations.tsv"
        feat_df = pd.read_csv(feat_path, sep="\t") if feat_path.exists() else None

        genomes_dir = root / "genomes"
        genomes_dir = genomes_dir if (genomes_dir.exists() and any(genomes_dir.iterdir())) else None

        test_sets = {}
        for split_name in ("C", "A", "B", "D"):
            split_dir = root / split_name
            t_data = split_dir / "test_data.tsv"
            t_lab = split_dir / "labels.csv"
            if t_data.is_file() and t_lab.is_file():
                df_test = pd.read_csv(t_data, sep="\t")
                counts = pd.read_csv(t_lab)["readcount"].to_numpy(dtype=float)
                test_sets[split_name] = (df_test, counts)

        if not test_sets:
            raise SystemExit(
                f"[BacterialExpression] No test splits found in {root}."
            )

        return dict(
            train_df=train_df,
            samples_df=samples_df,
            genomes_dir=genomes_dir,
            test_sets=test_sets,
            feature_annotations=feat_df,
            condition_annotations=samples_df,
            id_features=ID_FEATURES,
            numeric_features=NUMERIC_FEATURES,
            gene_metadata_features=GENE_METADATA_FEATURES,
            condition_metadata_features=CONDITION_METADATA_FEATURES,
        )

