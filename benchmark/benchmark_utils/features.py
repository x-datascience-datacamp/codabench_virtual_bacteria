"""Shared feature engineering for the gene-expression baseline.

Combines the three data sources into a numeric design matrix, mirroring the
getting-started notebook:

- **gene table** (``meta["feature_annotations"]``) joined on
  ``(genome_id, feature_id)``, and **sample metadata** joined on
  ``(genome_id, run_accession)`` (``run_accession`` -> ``Run`` on the
  synthetic dataset, whose gene columns sit in the observations);
- **genome-derived promoter features** — nucleotide composition of the region
  immediately upstream of each gene, read from ``<genome_id>.gbff`` (zeros
  when ``genomes_dir`` is ``None``, e.g. on the synthetic dataset);
- **tabular preprocessing** — median-impute + standardize numeric columns,
  one-hot encode the most frequent levels of categoricals, fit on train only.

Participants are free to ignore this and build richer representations (k-mers,
learned sequence encoders, more metadata) — it is only the baseline's recipe.
"""

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROMOTER_COLS = ["promoter_A", "promoter_C", "promoter_G", "promoter_T",
                 "promoter_GC"]
GENE_KEYS = ["genome_id", "feature_id"]
RUN_KEYS = ["genome_id", "run_accession"]
# Metadata columns that are coordinates or identifiers, not features.
NOT_FEATURES = ["Chr", "Start", "End", "ref_genome_id", "ref_feature_id",
                "prj_accession", "experiment_accession"]
# Categoricals with more levels in train (accessions, free text) are dropped.
MAX_LEVELS = 200


def join(df, table, keys):
    """Left-join the columns of ``table`` that ``df`` lacks."""
    cols = keys + [c for c in table.columns if c not in df.columns]
    return df.merge(table[cols], on=keys, how="left")


def merge_samples(df, samples_df):
    """Join experimental metadata onto the observations."""
    if samples_df is None:
        return df.copy()
    if "Run" in samples_df.columns:
        return df.merge(samples_df, left_on="run_accession", right_on="Run",
                        how="left")
    return join(df, samples_df, RUN_KEYS)


def sequence_features(seq):
    """Nucleotide-composition summary of a DNA string (notebook baseline)."""
    seq = (seq or "").upper()
    n = max(len(seq), 1)
    return {
        "promoter_A": seq.count("A") / n,
        "promoter_C": seq.count("C") / n,
        "promoter_G": seq.count("G") / n,
        "promoter_T": seq.count("T") / n,
        "promoter_GC": (seq.count("G") + seq.count("C")) / n,
    }


def _extract_upstream(row, records, upstream=200):
    """DNA immediately upstream of a gene, in its reading orientation.

    Handles semicolon-separated multi-interval genes and 1-based coordinates
    (the ``+`` strand reads before ``Start``; the ``-`` strand reads after
    ``End`` and is reverse-complemented). Returns ``""`` when the sequence is
    unavailable.
    """
    chrom = str(row["Chr"]).split(";")[0]
    record = records.get(chrom)
    if record is None:
        return ""
    start = int(str(row["Start"]).split(";")[0])
    end = int(str(row["End"]).split(";")[-1])
    strand = str(row["Strand"]).split(";")[0]
    if strand == "+":
        lo = max(start - 1 - upstream, 0)
        seq = record.seq[lo:start - 1]
    else:
        seq = record.seq[end:end + upstream].reverse_complement()
    return str(seq)


def promoter_features(genes, genomes_dir, upstream=200):
    """Promoter composition of each gene of the gene table.

    A gene's ``Chr`` names a record of its ``<genome_id>.gbff`` by VERSION
    (RefSeq) or LOCUS name (``contig_<i>``, repeated across genomes), so the
    records are looked up one genome at a time.
    """
    feats = pd.DataFrame(0.0, index=genes.index, columns=PROMOTER_COLS)
    if genomes_dir is None:
        return feats
    for gid, group in genes.groupby("genome_id", observed=True):
        path = genomes_dir / f"{gid}.gbff"
        if not path.exists():
            continue
        records = {}
        for record in SeqIO.parse(str(path), "genbank"):
            records[record.id] = records[record.name] = record
        seqs = group.apply(_extract_upstream, axis=1, records=records,
                           upstream=upstream)
        feats.loc[group.index] = [
            list(sequence_features(s).values()) for s in seqs]
    return feats


class FeatureBuilder:
    """Fit-once / transform design matrix over the three data sources.

    Without ``meta["categorical_features"]``, the features are the numeric
    offsets plus the gene and condition metadata columns, numeric or
    categorical by dtype.
    """

    def __init__(self, meta):
        self.numeric = list(meta["numeric_features"])
        self.categorical = meta.get("categorical_features")
        # ``species`` is both a gene and a condition column.
        self.metadata = list(dict.fromkeys(
            list(meta.get("gene_metadata_features", []))
            + list(meta.get("condition_metadata_features", []))))
        self.genes = meta.get("feature_annotations")

    def _assemble(self, df, samples_df):
        if self.genes_ is not None:
            df = join(df, self.genes_, GENE_KEYS)
        else:
            df = df.assign(**{c: 0.0 for c in PROMOTER_COLS})
        merged = merge_samples(df, samples_df)
        merged[self.categorical_] = merged[self.categorical_].astype(str)
        return merged

    def fit(self, df, samples_df, genomes_dir):
        self.genes_ = None
        if self.genes is not None:
            self.genes_ = pd.concat([self.genes, promoter_features(
                self.genes, genomes_dir)], axis=1)
        self.categorical_ = []
        merged = self._assemble(df, samples_df)
        numeric, categorical = self.numeric, self.categorical
        if categorical is None:
            cols = [c for c in self.metadata
                    if c in merged.columns and c not in NOT_FEATURES]
            is_num = {c: pd.api.types.is_numeric_dtype(merged[c])
                      and not pd.api.types.is_bool_dtype(merged[c])
                      for c in cols}
            numeric = numeric + [c for c in cols if is_num[c]]
            categorical = [c for c in cols if not is_num[c]
                           and merged[c].nunique() <= MAX_LEVELS]
        self.categorical_ = categorical
        merged[categorical] = merged[categorical].astype(str)
        self.preprocessor_ = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric + PROMOTER_COLS),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  max_categories=16, sparse_output=False),
             categorical),
        ])
        self.preprocessor_.fit(merged)
        return self

    def transform(self, df, samples_df):
        merged = self._assemble(df, samples_df)
        return self.preprocessor_.transform(merged).astype(np.float32)
