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
    "genome_id",
    "feature_id",
    "run_accession",
    "gene_uid",
    "condition_uid",
]
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

    def get_data(self):
        root = get_data_path("gene_expression")
        if not root.exists():
            local_fallback = (
                Path(__file__).resolve().parents[2]
                / "phases/main/input_data/data/gene_expression"
            )
            if local_fallback.exists():
                root = local_fallback
            else:
                raise SystemExit(
                    f"[BacterialExpression] no data at {root}. "
                    "Generate it with "
                    f"`python tools/setup_data.py --mode {self.mode}`."
                )

        mode_dir = root / self.mode
        data_dir = mode_dir if mode_dir.exists() else root

        train_path = data_dir / "train_data.tsv"
        if not train_path.exists():
            raise SystemExit(
                f"[BacterialExpression] missing {train_path} "
                f"for mode '{self.mode}'. "
                f"Run `python tools/setup_data.py --mode {self.mode}` first."
            )
        train_df = pd.read_csv(train_path, sep="\t")

        cond_path = data_dir / "condition_annotations.tsv"
        samples_df = (
            pd.read_csv(cond_path, sep="\t") if cond_path.exists() else None
        )

        feat_path = data_dir / "feature_annotations.tsv"
        feat_df = (
            pd.read_csv(feat_path, sep="\t") if feat_path.exists() else None
        )

        genomes_dir = data_dir / "genomes"
        if not (genomes_dir.exists() and any(genomes_dir.iterdir())):
            root_genomes = root / "genomes"
            genomes_dir = (
                root_genomes
                if (root_genomes.exists() and any(root_genomes.iterdir()))
                else None
            )

        test_sets = {}
        for split_name in ("A", "B", "C", "D"):
            split_dir = data_dir / split_name
            t_data = split_dir / "test_data.tsv"
            t_lab = split_dir / "labels.csv"
            if t_data.is_file() and t_lab.is_file():
                df_test = pd.read_csv(t_data, sep="\t")
                counts = pd.read_csv(t_lab)["readcount"].to_numpy(dtype=float)
                test_sets[split_name] = (df_test, counts)

        if not test_sets:
            raise SystemExit(
                f"[BacterialExpression] No test splits found in {data_dir} "
                f"for mode '{self.mode}'."
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
