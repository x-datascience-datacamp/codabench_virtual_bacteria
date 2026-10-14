"""Build the tiered competition splits from the bacterial transcriptomics data.

Processes raw or sample multi-species CSV tables:
- ``read_count_global.csv`` (or ``baseline_read_count_global.csv``)
- ``feature_annotation_global.csv`` (or ``baseline_feature_annotation_global.csv``)
- ``condition_annotation_global.csv`` (or ``baseline_condition_annotation_global.csv``)
- optional ``gbff/`` or ``genomes/`` directory.

Partition tiers governed by observation ``status``:
- ``Train``  : training observations (L-shaped block)
- ``C``      : test set C (unseen gene x condition combination, seen entities)
- ``A``      : test set A (unseen genes, seen conditions)
- ``B``      : test set B (seen genes, unseen conditions)
- ``D``      : test set D (unseen genes x unseen conditions)

Outputs the layout read by ``BacterialExpression`` dataset:
    <out_dir>/
      train_data.tsv
      feature_annotations.tsv
      condition_annotations.tsv
      C/{test_data.tsv, labels.csv}
      A/{test_data.tsv, labels.csv}
      B/{test_data.tsv, labels.csv}
      D/{test_data.tsv, labels.csv}
      genomes/ (optional .gbff / .gb files)
"""

import argparse
import csv
import math
import shutil
from pathlib import Path

OUT_DIR = Path("phases/main/input_data/data/gene_expression")


def _find_file(data_dir: Path, primary_name: str, fallback_name: str) -> Path:
    p1 = data_dir / primary_name
    if p1.is_file():
        return p1
    p2 = data_dir / fallback_name
    if p2.is_file():
        return p2
    raise FileNotFoundError(
        f"Could not find {primary_name} or {fallback_name} in {data_dir}"
    )


def load_raw_csv(data_dir: Path = Path("data")):
    """Load and join tables using standard library csv (dependency-free)."""
    rc_file = _find_file(data_dir, "read_count_global.csv",
                         "baseline_read_count_global.csv")
    feat_file = _find_file(data_dir, "feature_annotation_global.csv",
                           "baseline_feature_annotation_global.csv")
    cond_file = _find_file(data_dir, "condition_annotation_global.csv",
                           "baseline_condition_annotation_global.csv")

    # Read feature annotations
    feat_dict = {}
    feat_rows = []
    with open(feat_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (r["genome_id"], r["feature_id"])
            feat_dict[key] = r
            feat_rows.append(r)

    # Read condition annotations
    cond_dict = {}
    cond_rows = []
    with open(cond_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (r["genome_id"], r["run_accession"])
            cond_dict[key] = r
            cond_rows.append(r)

    # Read and join read counts
    joined_rows = []
    with open(rc_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            g_id = r["genome_id"]
            f_id = r["feature_id"]
            run_id = r["run_accession"]

            feat = feat_dict.get((g_id, f_id), {})
            cond = cond_dict.get((g_id, run_id), {})

            row = dict(r)
            # Add feature fields (excluding species/status collision)
            for k, v in feat.items():
                if k not in ("species", "status", "ref_species"):
                    row[k] = v
            # Add condition fields
            for k, v in cond.items():
                if k not in ("species", "status"):
                    row[k] = v

            row["gene_uid"] = f"{g_id}_{f_id}"
            row["condition_uid"] = f"{g_id}_{run_id}"
            joined_rows.append(row)

    genomes_dir = None
    for cand in ("gbff", "genomes"):
        p = data_dir / cand
        if p.is_dir():
            genomes_dir = p
            break

    return joined_rows, feat_rows, cond_rows, genomes_dir


def write_layout_csv(out_dir: Path, joined_rows: list, feat_rows: list,
                     cond_rows: list, genomes_src: Path = None):
    """Write output TSV/CSV files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Partition by status
    train_rows = []
    test_splits = {"C": [], "A": [], "B": [], "D": []}

    for r in joined_rows:
        st = r.get("status")
        if st == "Train":
            train_rows.append(r)
        elif st in test_splits:
            test_splits[st].append(r)

    # Write train_data.tsv
    if train_rows:
        fieldnames = list(train_rows[0].keys())
        with open(out_dir / "train_data.tsv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(train_rows)

    # Write annotations
    if feat_rows:
        with open(out_dir / "feature_annotations.tsv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(feat_rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(feat_rows)

    if cond_rows:
        with open(out_dir / "condition_annotations.tsv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(cond_rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(cond_rows)

    # Write test splits
    for name, rows in test_splits.items():
        split_dir = out_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)
        if rows:
            feat_fields = [k for k in rows[0].keys() if k not in ("readcount", "status")]
            with open(split_dir / "test_data.tsv", "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=feat_fields, delimiter="\t", extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

            with open(split_dir / "labels.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["readcount"], extrasaction="ignore")
                writer.writeheader()
                for r in rows:
                    writer.writerow({"readcount": r["readcount"]})

    # Genomes copy if available
    if genomes_src is not None and genomes_src.is_dir():
        dst = out_dir / "genomes"
        dst.mkdir(exist_ok=True)
        for g_file in sorted(genomes_src.glob("*")):
            if g_file.suffix in (".gb", ".gbff", ".fna", ".fasta"):
                dst_file = dst / g_file.name
                if not dst_file.exists():
                    shutil.copyfile(g_file, dst_file)

    counts = {k: len(v) for k, v in test_splits.items()}
    print(f"[setup_data] wrote {out_dir} (train={len(train_rows)}, "
          f"{', '.join(f'{k}={v}' for k, v in counts.items())})")


def make_demo_csv():
    """Synthetic dataset for smoke tests (dependency-free)."""
    import random
    rng = random.Random(0)
    species_list = ["Escherichia_coli", "Pseudomonas_putida", "Bacillus_subtilis"]
    genomes = [f"GCF_{i:09d}.1" for i in range(1, 4)]
    runs = [f"SRR{i:07d}" for i in range(1, 21)]

    cond_rows = []
    for r in runs:
        g = rng.choice(genomes)
        sp = species_list[genomes.index(g)]
        status = "train" if rng.random() < 0.8 else "test_full"
        cond_rows.append({
            "species": sp, "genome_id": g, "prj_accession": "PRJNA001",
            "run_accession": r, "status": status,
            "log_size_factor": f"{rng.gauss(0.0, 0.3):.6f}",
        })

    feat_rows = []
    for g in genomes:
        sp = species_list[genomes.index(g)]
        for f_idx in range(1, 31):
            fid = f"cds-WP_{g}_{f_idx:03d}"
            length = rng.randint(300, 2500)
            status = "train" if rng.random() < 0.75 else "test_full"
            feat_rows.append({
                "species": sp, "genome_id": g, "feature_id": fid,
                "ref_genome_id": g, "ref_feature_id": fid,
                "Chr": f"NC_{g}", "Start": str(f_idx * 1000),
                "End": str(f_idx * 1000 + length), "Strand": "+",
                "Length": str(float(length)), "homology": "100.0",
                "on_contig_edge": "False", "status": status,
                "log_length_ratio": f"{math.log(length) - 6.9:.6f}",
            })

    obs_rows = []
    cond_map = {(r["genome_id"], r["run_accession"]): r for r in cond_rows}
    feat_map = {(r["genome_id"], r["feature_id"]): r for r in feat_rows}

    for c_row in cond_rows:
        g = c_row["genome_id"]
        run_id = c_row["run_accession"]
        for f_row in feat_rows:
            if f_row["genome_id"] != g:
                continue
            fid = f_row["feature_id"]
            fs = f_row["status"]
            cs = c_row["status"]

            if fs == "train" and cs == "train":
                st = "Train"
            elif fs == "test_full" and cs == "train":
                st = "A"
            elif fs == "train" and cs == "test_full":
                st = "B"
            elif fs == "test_full" and cs == "test_full":
                st = "D"
            else:
                st = "C"

            log_mu = (
                4.0 + float(f_row["log_length_ratio"])
                + float(c_row["log_size_factor"]) + rng.gauss(0, 0.2)
            )
            mu = math.exp(max(0.0, min(10.0, log_mu)))
            count = max(0.0, rng.gauss(mu, math.sqrt(mu + 0.2 * mu * mu)))

            row = {
                "species": c_row["species"],
                "genome_id": g,
                "feature_id": fid,
                "run_accession": run_id,
                "readcount": f"{count:.1f}",
                "feature_status": fs,
                "run_status": cs,
                "status": st,
                "gene_uid": f"{g}_{fid}",
                "condition_uid": f"{g}_{run_id}",
            }
            for k, v in f_row.items():
                if k not in ("species", "status", "ref_species"):
                    row[k] = v
            for k, v in c_row.items():
                if k not in ("species", "status"):
                    row[k] = v
            obs_rows.append(row)

    return obs_rows, feat_rows, cond_rows, None


def main():
    parser = argparse.ArgumentParser(description="Build competition splits")
    parser.add_argument("--demo", action="store_true",
                        help="generate a tiny synthetic dataset")
    parser.add_argument("--data-dir", default="data",
                        help="source raw data directory")
    parser.add_argument("--out", default=str(OUT_DIR),
                        help="output data directory for benchmark")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    if args.demo:
        obs, feat_rows, cond_rows, genomes_dir = make_demo_csv()
    else:
        obs, feat_rows, cond_rows, genomes_dir = load_raw_csv(Path(args.data_dir))

    write_layout_csv(out_dir, obs, feat_rows, cond_rows, genomes_dir)


if __name__ == "__main__":
    main()
