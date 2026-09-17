"""Shared feature engineering for the gene-expression baseline.

Combines the three data sources into a numeric design matrix, mirroring the
getting-started notebook:

- **sample metadata** merged on ``run_accession`` -> ``Run``;
- **genome-derived promoter features** — nucleotide composition of the region
  immediately upstream of each gene (needs the GenBank ``.gb`` genomes; skipped
  when ``genomes_dir`` is ``None``, e.g. on the synthetic dataset);
- **tabular preprocessing** — median-impute + standardize numeric columns,
  most-frequent-impute + one-hot encode categoricals, fit on train only.

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


def merge_samples(df, samples_df):
    """Join experimental metadata onto the observations."""
    if samples_df is None:
        return df.copy()
    return df.merge(samples_df, left_on="run_accession", right_on="Run",
                    how="left")


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


def _load_records(genomes_dir):
    """Map every GenBank record id in ``genomes_dir`` to its SeqRecord."""
    records = {}
    for path in sorted(genomes_dir.glob("*.gb")):
        for record in SeqIO.parse(str(path), "genbank"):
            records[record.id] = record
    return records


def promoter_features(df, genomes_dir, upstream=200):
    """Per-observation promoter composition (zeros when no genomes)."""
    if genomes_dir is None:
        return pd.DataFrame(0.0, index=df.index, columns=PROMOTER_COLS)
    records = _load_records(genomes_dir)
    # Compute once per unique gene, then broadcast back to observations.
    gene_cols = ["Geneid", "Chr", "Start", "End", "Strand"]
    genes = df[gene_cols].drop_duplicates("Geneid").copy()
    seqs = genes.apply(_extract_upstream, axis=1, records=records,
                       upstream=upstream)
    feats = pd.DataFrame(
        [sequence_features(s) for s in seqs], index=genes.index)
    feats["Geneid"] = genes["Geneid"].values
    merged = df[["Geneid"]].merge(feats, on="Geneid", how="left")
    return merged[PROMOTER_COLS].fillna(0.0).set_axis(df.index)


class FeatureBuilder:
    """Fit-once / transform design matrix over the three data sources."""

    def __init__(self, numeric_features, categorical_features):
        self.numeric = list(numeric_features)
        self.categorical = list(categorical_features)

    def _assemble(self, df, samples_df, genomes_dir):
        merged = merge_samples(df, samples_df)
        promo = promoter_features(merged, genomes_dir)
        merged = pd.concat([merged.reset_index(drop=True),
                            promo.reset_index(drop=True)], axis=1)
        return merged

    def fit(self, df, samples_df, genomes_dir):
        merged = self._assemble(df, samples_df, genomes_dir)
        numeric = self.numeric + PROMOTER_COLS
        self.preprocessor_ = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore",
                                          sparse_output=False)),
            ]), self.categorical),
        ])
        self.preprocessor_.fit(merged)
        return self

    def transform(self, df, samples_df, genomes_dir):
        merged = self._assemble(df, samples_df, genomes_dir)
        return self.preprocessor_.transform(merged).astype(np.float32)
