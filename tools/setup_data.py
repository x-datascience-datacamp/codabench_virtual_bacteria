"""Build the tiered competition splits from the bacterial transcriptomics data.

Processes raw or sample multi-species CSV tables:
- ``read_count_{debug|train|global}.csv``
- ``feature_annotation_{debug|train|global}.csv``
- ``condition_annotation_{debug|train|global}.csv``
- optional ``gbff/`` or ``genomes/`` directory.

Modes:
- ``debug``: small subsample for debugging, simulates A, B, C, D splits.
- ``train``: available training data, simulates A, B, C, D splits.
- ``full`` : final evaluation data, uses pre-existing A, B, C, D labels.

Outputs the layout read by ``BacterialExpression`` dataset:
    <out_dir>/<mode>/
      train_data.tsv
      feature_annotations.tsv
      condition_annotations.tsv
      A/{test_data.tsv, labels.csv}
      B/{test_data.tsv, labels.csv}
      C/{test_data.tsv, labels.csv}
      D/{test_data.tsv, labels.csv}
      genomes/ (optional .gbff / .gb files)
"""

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("phases/main/input_data/data/gene_expression")

STATUS_RULES = pd.DataFrame(
    [
        ("train", "train", "Train"),
        ("test_partial", "train", "Train"),
        ("train", "test_partial", "Train"),
        ("test_partial", "test_partial", "A"),
        ("test_partial", "test_full", "B"),
        ("train", "test_full", "B"),
        ("test_full", "test_partial", "C"),
        ("test_full", "train", "C"),
        ("test_full", "test_full", "D"),
    ],
    columns=["feature_status", "run_status", "status"],
)


def _find_file(data_dir: Path, candidate_names: list) -> Path:
    for name in candidate_names:
        p = data_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Could not find any of {candidate_names} in {data_dir}"
    )


def simulate_splits(
    feat_df: pd.DataFrame, cond_df: pd.DataFrame, seed: int = 42
):
    """Simulate test_partial and test_full splits for debug/train modes.

    Rule:
    - 20% randomly selected conditions: 80% test_partial, 20% test_full.
      Remaining 80% conditions: train.
    - 20% randomly selected features: 80% test_partial, 20% test_full.
      Remaining 80% features: train.
    """
    # 1. Conditions
    cond_df = cond_df.copy()
    n_cond = len(cond_df)
    n_test_cond = int(round(0.20 * n_cond))
    test_cond_idx = cond_df.sample(n=n_test_cond, random_state=seed).index

    n_full_cond = int(round(0.20 * n_test_cond))
    if n_test_cond > 0 and n_full_cond == 0 and 0.20 * n_test_cond > 0:
        n_full_cond = 1
    test_full_cond_idx = (
        cond_df.loc[test_cond_idx]
        .sample(n=n_full_cond, random_state=seed)
        .index
    )
    test_partial_cond_idx = test_cond_idx.difference(test_full_cond_idx)

    cond_df["status"] = "train"
    cond_df.loc[test_partial_cond_idx, "status"] = "test_partial"
    cond_df.loc[test_full_cond_idx, "status"] = "test_full"

    # 2. Features
    feat_df = feat_df.copy()
    n_feat = len(feat_df)
    n_test_feat = int(round(0.20 * n_feat))
    test_feat_idx = feat_df.sample(n=n_test_feat, random_state=seed).index

    n_full_feat = int(round(0.20 * n_test_feat))
    if n_test_feat > 0 and n_full_feat == 0 and 0.20 * n_test_feat > 0:
        n_full_feat = 1
    test_full_feat_idx = (
        feat_df.loc[test_feat_idx]
        .sample(n=n_full_feat, random_state=seed)
        .index
    )
    test_partial_feat_idx = test_feat_idx.difference(test_full_feat_idx)

    feat_df["status"] = "train"
    feat_df.loc[test_partial_feat_idx, "status"] = "test_partial"
    feat_df.loc[test_full_feat_idx, "status"] = "test_full"

    return feat_df, cond_df


def load_raw_data(
    data_dir: Path = Path("data"), mode: str = "debug", seed: int = 42
):
    """Load and merge tables using pandas."""
    if mode == "debug":
        rc_cands = [
            "read_count_debug.csv",
            "baseline_read_count_global.csv",
            "read_count.csv",
        ]
        feat_cands = [
            "feature_annotation_debug.csv",
            "baseline_feature_annotation_global.csv",
        ]
        cond_cands = [
            "condition_annotation_debug.csv",
            "baseline_condition_annotation_global.csv",
        ]
    elif mode == "train":
        rc_cands = ["read_count_train.csv", "read_count.csv"]
        feat_cands = ["feature_annotation_train.csv", "feature_annotation.csv"]
        cond_cands = [
            "condition_annotation_train.csv",
            "condition_annotation.csv",
        ]
    elif mode == "full":
        rc_cands = [
            "read_count_global.csv",
            "baseline_read_count_global.csv",
            "read_count.csv",
        ]
        feat_cands = [
            "feature_annotation_global.csv",
            "baseline_feature_annotation_global.csv",
        ]
        cond_cands = [
            "condition_annotation_global.csv",
            "baseline_condition_annotation_global.csv",
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    rc_file = _find_file(data_dir, rc_cands)
    feat_file = _find_file(data_dir, feat_cands)
    cond_file = _find_file(data_dir, cond_cands)

    rc_df = pd.read_csv(rc_file)
    feat_df = pd.read_csv(feat_file)
    cond_df = pd.read_csv(cond_file)

    if mode in ("debug", "train"):
        feat_df, cond_df = simulate_splits(feat_df, cond_df, seed=seed)

    # Build unique identifiers
    rc_df["gene_uid"] = (
        rc_df["genome_id"].astype(str) + "_" + rc_df["feature_id"].astype(str)
    )
    rc_df["condition_uid"] = (
        rc_df["genome_id"].astype(str)
        + "_"
        + rc_df["run_accession"].astype(str)
    )

    # Prepare feature annotations for merge
    feat_cols = [
        c for c in feat_df.columns if c not in ("species", "ref_species")
    ]
    feat_merge = feat_df[feat_cols].copy()
    if "status" in feat_merge.columns:
        feat_merge = feat_merge.rename(columns={"status": "feature_status"})

    # Prepare condition annotations for merge
    cond_cols = [c for c in cond_df.columns if c not in ("species",)]
    cond_merge = cond_df[cond_cols].copy()
    if "status" in cond_merge.columns:
        cond_merge = cond_merge.rename(columns={"status": "run_status"})

    # Clean existing status columns in rc_df for debug/train simulation
    if mode in ("debug", "train"):
        cols_to_drop = [
            c
            for c in ("feature_status", "run_status", "status")
            if c in rc_df.columns
        ]
        if cols_to_drop:
            rc_df = rc_df.drop(columns=cols_to_drop)

    # Merge feature metadata
    feat_overlap = [
        c
        for c in feat_merge.columns
        if c in rc_df.columns and c not in ("genome_id", "feature_id")
    ]
    if feat_overlap:
        feat_merge = feat_merge.drop(columns=feat_overlap)
    merged_df = rc_df.merge(
        feat_merge, on=["genome_id", "feature_id"], how="left"
    )

    # Merge condition metadata
    cond_overlap = [
        c
        for c in cond_merge.columns
        if c in merged_df.columns and c not in ("genome_id", "run_accession")
    ]
    if cond_overlap:
        cond_merge = cond_merge.drop(columns=cond_overlap)
    merged_df = merged_df.merge(
        cond_merge, on=["genome_id", "run_accession"], how="left"
    )

    # Assign observation status
    if mode in ("debug", "train"):
        merged_df = merged_df.merge(
            STATUS_RULES, on=["feature_status", "run_status"], how="left"
        )
        merged_df["status"] = merged_df["status"].fillna("Train")

    genomes_dir = None
    for cand in ("gbff", "genomes"):
        p = data_dir / cand
        if p.is_dir():
            genomes_dir = p
            break

    return merged_df, feat_df, cond_df, genomes_dir


def write_layout(
    out_dir: Path,
    obs_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    cond_df: pd.DataFrame,
    genomes_src: Path = None,
):
    """Write output TSV/CSV files using pandas."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train partition
    train_df = obs_df[obs_df["status"] == "Train"]
    if not train_df.empty:
        train_df.to_csv(out_dir / "train_data.tsv", sep="\t", index=False)

    # Annotations
    if not feat_df.empty:
        feat_df.to_csv(
            out_dir / "feature_annotations.tsv", sep="\t", index=False
        )
    if not cond_df.empty:
        cond_df.to_csv(
            out_dir / "condition_annotations.tsv", sep="\t", index=False
        )

    # Test splits: A, B, C, D
    counts = {}
    for split_name in ("A", "B", "C", "D"):
        split_df = obs_df[obs_df["status"] == split_name]
        counts[split_name] = len(split_df)
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        if not split_df.empty:
            feat_cols = [
                c for c in split_df.columns if c not in ("readcount", "status")
            ]
            split_df[feat_cols].to_csv(
                split_dir / "test_data.tsv", sep="\t", index=False
            )
            split_df[["readcount"]].to_csv(
                split_dir / "labels.csv", index=False
            )

    # Copy genomes if available
    if genomes_src is not None and genomes_src.is_dir():
        dst = out_dir / "genomes"
        dst.mkdir(exist_ok=True)
        for g_file in sorted(genomes_src.glob("*")):
            if g_file.suffix in (".gb", ".gbff", ".fna", ".fasta"):
                dst_file = dst / g_file.name
                if not dst_file.exists():
                    shutil.copyfile(g_file, dst_file)

    print(
        f"[setup_data] wrote {out_dir} (train={len(train_df)}, "
        f"{', '.join(f'{k}={v}' for k, v in counts.items())})"
    )


def make_demo_data(seed: int = 0):
    """Synthetic dataset for smoke tests (using numpy and pandas)."""
    rng = np.random.default_rng(seed)
    species_list = [
        "Escherichia_coli",
        "Pseudomonas_putida",
        "Bacillus_subtilis",
    ]
    genomes = [f"GCF_{i:09d}.1" for i in range(1, 4)]
    runs = [f"SRR{i:07d}" for i in range(1, 21)]

    # Conditions
    cond_species = [rng.choice(species_list) for _ in runs]
    cond_genomes = [genomes[species_list.index(sp)] for sp in cond_species]
    split_status = ["train", "test_partial", "test_full"]
    cond_status = rng.choice(split_status, len(runs), p=[0.5, 0.3, 0.2])
    log_sf = rng.normal(0.0, 0.3, len(runs))

    cond_df = pd.DataFrame(
        {
            "species": cond_species,
            "genome_id": cond_genomes,
            "prj_accession": "PRJNA001",
            "run_accession": runs,
            "status": cond_status,
            "log_size_factor": log_sf,
        }
    )

    # Features
    feat_records = []
    for g in genomes:
        sp = species_list[genomes.index(g)]
        for f_idx in range(1, 31):
            fid = f"cds-WP_{g}_{f_idx:03d}"
            length = int(rng.integers(300, 2500))
            status = rng.choice(split_status, p=[0.5, 0.3, 0.2])
            feat_records.append(
                {
                    "species": sp,
                    "genome_id": g,
                    "feature_id": fid,
                    "ref_genome_id": g,
                    "ref_feature_id": fid,
                    "Chr": f"NC_{g}",
                    "Start": f_idx * 1000,
                    "End": f_idx * 1000 + length,
                    "Strand": "+",
                    "Length": float(length),
                    "homology": 100.0,
                    "on_contig_edge": False,
                    "status": status,
                    "log_length_ratio": math.log(length) - 6.9,
                }
            )
    feat_df = pd.DataFrame(feat_records)

    # Observations
    obs_df = cond_df.merge(
        feat_df, on=["species", "genome_id"], suffixes=("_run", "_feat")
    )
    obs_df = obs_df.rename(
        columns={"status_feat": "feature_status", "status_run": "run_status"}
    ).merge(STATUS_RULES, on=["feature_status", "run_status"], how="left")
    obs_df["gene_uid"] = obs_df["genome_id"] + "_" + obs_df["feature_id"]
    obs_df["condition_uid"] = (
        obs_df["genome_id"] + "_" + obs_df["run_accession"]
    )

    log_mu = (
        4.0
        + obs_df["log_length_ratio"]
        + obs_df["log_size_factor"]
        + rng.normal(0, 0.2, len(obs_df))
    )
    mu = np.exp(np.clip(log_mu, 0.0, 10.0))
    counts = np.maximum(0.0, rng.normal(mu, np.sqrt(mu + 0.2 * mu * mu)))
    obs_df["readcount"] = np.round(counts, 1)

    return obs_df, feat_df, cond_df, None


def main():
    parser = argparse.ArgumentParser(description="Build competition splits")
    parser.add_argument(
        "--mode",
        choices=["debug", "train", "full", "all"],
        default="debug",
        help="Dataset mode to build: debug, train, full, or all "
        "(default: debug)",
    )
    parser.add_argument(
        "--demo", action="store_true", help="generate a tiny synthetic dataset"
    )
    parser.add_argument(
        "--data-dir", default="data", help="source raw data directory"
    )
    parser.add_argument(
        "--out",
        default=str(OUT_DIR),
        help="output data directory for benchmark",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for simulation (default: 42)",
    )
    args = parser.parse_args()

    out_base = Path(args.out)
    if args.demo:
        obs_df, feat_df, cond_df, genomes_dir = make_demo_data(seed=args.seed)
        write_layout(out_base, obs_df, feat_df, cond_df, genomes_dir)
        return

    modes = ["debug", "train", "full"] if args.mode == "all" else [args.mode]
    for m in modes:
        try:
            print(f"[setup_data] processing mode: {m}")
            obs_df, feat_df, cond_df, genomes_dir = load_raw_data(
                Path(args.data_dir), mode=m, seed=args.seed
            )
            target_dir = out_base / m if out_base.name != m else out_base
            write_layout(target_dir, obs_df, feat_df, cond_df, genomes_dir)
        except FileNotFoundError as e:
            if args.mode == "all":
                print(f"[setup_data] skipping mode '{m}': {e}")
            else:
                raise


if __name__ == "__main__":
    main()
