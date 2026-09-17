"""Build the tiered competition split from the raw multi-species table.

The whole split lives here (the benchmark only *loads* the result). One
``build_split`` code path serves both:

- ``python tools/setup_data.py``          — real data (fill ``load_raw`` +
  the species tiers below with the assembled files);
- ``python tools/setup_data.py --demo``   — a tiny synthetic dataset in the
  exact on-disk layout, to exercise the loader and the Codabench pipeline
  end-to-end with no real data.

Tiers (see ``design.md``):
- TRAIN         : training species, L-shaped region (not double-held-out).
- PUBLIC        : training species, ``feature_split==test AND
                  condition_split==test`` (new gene x new condition).
- PRIVATE near  : held-out species close to a training species.
- PRIVATE far   : held-out species far in phylogeny.

Leakage guards enforced below: (1) genes cut at the ortholog-group level so a
"new gene" is genuinely novel; (2) whole studies held out for conditions;
(3) normalization offsets are per-observation (never test-set aggregates).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("phases/main/input_data/data/gene_expression")

# Fill these with the real references (genome accessions) once the data lands.
TRAIN_SPECIES = ["GCF_train"]
NEAR_SPECIES = ["GCF_near"]
FAR_SPECIES = ["GCF_far"]

OBS_COLS = ["Geneid", "run_accession", "reference", "Chr", "Start", "End",
            "Strand", "Length", "log_length_ratio", "log_size_factor",
            "feature_split", "condition_split"]


def _held_out_block(obs):
    return ((obs["feature_split"] == "test")
            & (obs["condition_split"] == "test"))


def build_split(obs):
    """Partition a raw long table into the four tiers.

    ``obs`` is one row per (gene x sample) observation with at least the
    columns in ``OBS_COLS`` plus ``counts`` and (for guard 1) an
    ``ortholog_group`` and (for guard 2) a ``study`` column.
    """
    in_train = obs["reference"].isin(TRAIN_SPECIES)
    block = _held_out_block(obs)

    train = obs[in_train & ~block].copy()
    public = obs[in_train & block].copy()
    near = obs[obs["reference"].isin(NEAR_SPECIES)].copy()
    far = obs[obs["reference"].isin(FAR_SPECIES)].copy()
    near["group"] = "near"
    far["group"] = "far"
    private = pd.concat([near, far], ignore_index=True)

    _check_leakage(obs, train, public, private)
    return train, public, private


def _check_leakage(obs, train, public, private):
    """Fail loudly if a split leaks across the intended boundaries.

    The public block is the L-shape's ``test-gene x test-condition`` corner, so
    a test-gene *does* appear in train (under train conditions) and a
    test-condition *does* appear in train (for train genes) — that is the
    design. What must not leak is a (gene x sample) *cell*, and the split
    labels must be consistent per gene / per study.
    """
    # Guard 1: feature_split is a property of the gene (ortholog group).
    if "ortholog_group" in obs.columns:
        bad = obs.groupby("ortholog_group")["feature_split"].nunique()
        assert (bad <= 1).all(), (
            "feature_split varies within an ortholog group "
            "(guard 1: the gene set partition must be gene-level)."
        )
    # Guard 2: condition_split is a property of the study (study holdout).
    if "study" in obs.columns:
        bad = obs.groupby("study")["condition_split"].nunique()
        assert (bad <= 1).all(), (
            "condition_split varies within a study "
            "(guard 2: hold out whole studies for the condition axis)."
        )
    # No (gene x sample) cell shared between train and public.
    cell = ["Geneid", "run_accession"]
    shared = train[cell].merge(public[cell], on=cell)
    assert shared.empty, (
        f"{len(shared)} (gene x sample) cells shared by train and public.")
    # Public is exactly the double-held-out block on the training species.
    assert _held_out_block(public).all(), "public has non-held-out rows."
    assert not _held_out_block(train).any(), "train has held-out-block rows."
    assert not (set(train["reference"]) & set(private["reference"])), (
        "train and private share a species."
    )


def write_layout(out_dir, train, public, private, samples, genomes_src=None):
    """Write the on-disk layout the ``Bacteria`` dataset reads."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train_data.tsv", sep="\t", index=False)
    samples.to_csv(out_dir / "samples.tsv", sep="\t", index=False)

    (out_dir / "public").mkdir(exist_ok=True)
    public[OBS_COLS].to_csv(
        out_dir / "public" / "test_data.tsv", sep="\t", index=False)
    public[["counts"]].to_csv(out_dir / "public" / "labels.csv", index=False)

    (out_dir / "private").mkdir(exist_ok=True)
    priv_cols = OBS_COLS + ["group"]
    private[priv_cols].to_csv(
        out_dir / "private" / "test_data.tsv", sep="\t", index=False)
    private[["counts", "group"]].to_csv(
        out_dir / "private" / "labels.csv", index=False)

    if genomes_src is not None:
        import shutil
        dst = out_dir / "genomes"
        dst.mkdir(exist_ok=True)
        for gb in Path(genomes_src).glob("*.gb"):
            shutil.copyfile(gb, dst / gb.name)
    print(f"[setup_data] wrote {out_dir}  "
          f"(train={len(train)}, public={len(public)}, "
          f"private={len(private)})")


def load_raw():
    """Load the assembled multi-species table + samples + genomes dir.

    Fill this in once the real files are available (they are expected to match
    the getting-started notebook schema). Must return
    ``(obs_df, samples_df, genomes_dir)``.
    """
    raise NotImplementedError(
        "Point load_raw at the assembled files (train/test tables, "
        "samples.tsv, genomes/) and set TRAIN/NEAR/FAR_SPECIES."
    )


def make_demo(seed=0):
    """Tiny synthetic dataset in the exact layout, with small genomes."""
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    rng = np.random.default_rng(seed)
    references = TRAIN_SPECIES + NEAR_SPECIES + FAR_SPECIES
    media = ["M9", "LB", "MOPS"]
    culture = ["batch", "chemostat"]

    # Studies own their condition_split (whole-study holdout); samples belong
    # to a study.
    n_studies = 8
    study_split = {f"study{j}": ("test" if rng.random() < 0.4 else "train")
                   for j in range(n_studies)}
    n_samples = 90
    sample_study = {}
    samples = pd.DataFrame({
        "Run": [f"SRR{i:04d}" for i in range(n_samples)],
        "Base Media": rng.choice(media, n_samples),
        "Culture Type": rng.choice(culture, n_samples),
        "pH": rng.normal(7.0, 0.4, n_samples),
        "Temperature (C)": rng.choice([30.0, 37.0, 42.0], n_samples),
        "Growth Rate (1/hr)": np.abs(rng.normal(0.6, 0.2, n_samples)),
    })
    for run in samples["Run"]:
        sample_study[run] = f"study{int(rng.integers(0, n_studies))}"

    # Ortholog groups own their feature_split (gene-set partition); each is a
    # gene in every species (orthologs share the id across references).
    n_og = 150
    og_split = {f"OG{g:03d}": ("test" if rng.random() < 0.4 else "train")
                for g in range(n_og)}

    rows, genomes = [], {}
    for ref in references:
        glen = 20000
        genomes[ref] = SeqRecord(
            Seq("".join(rng.choice(list("ACGT"), glen))),
            id=f"{ref}.1", name=ref, description=f"demo {ref}",
            annotations={"molecule_type": "DNA"})
        for g in range(n_og):
            og = f"OG{g:03d}"
            length = int(rng.integers(200, 3000))
            start = int(rng.integers(300, glen - 3000))
            for _ in range(int(rng.integers(4, 10))):
                run = samples["Run"].iloc[int(rng.integers(0, n_samples))]
                study = sample_study[run]
                rows.append({
                    "Geneid": f"{ref}_{og}", "run_accession": run,
                    "reference": ref, "Chr": f"{ref}.1",
                    "Start": start, "End": start + length, "Strand":
                    rng.choice(["+", "-"]), "Length": float(length),
                    "log_length_ratio": float(np.log(length) - 7.0),
                    "log_size_factor": float(rng.normal(0, 0.3)),
                    "feature_split": og_split[og],
                    "condition_split": study_split[study],
                    "ortholog_group": og, "study": study,
                })
    obs = pd.DataFrame(rows).drop_duplicates(["Geneid", "run_accession"])
    obs = obs.reset_index(drop=True)

    # Counts: feature/sample driven, shifted for held-out species.
    s = samples.set_index("Run").loc[obs["run_accession"]]
    shift = np.where(obs["reference"].isin(FAR_SPECIES), 1.0,
                     np.where(obs["reference"].isin(NEAR_SPECIES), 0.3, 0.0))
    log_mu = (3.0 + shift
              + 0.9 * (obs["log_length_ratio"].values)
              + 1.5 * (s["Growth Rate (1/hr)"].values - 0.6)
              - 0.7 * (s["pH"].values - 7.0)
              + obs["log_size_factor"].values
              + rng.normal(0, 0.15, len(obs)))
    mu = np.exp(np.clip(log_mu, -2, 9))
    obs["counts"] = rng.negative_binomial(8.0, 8.0 / (8.0 + mu))
    return obs, samples, genomes


def main():
    parser = argparse.ArgumentParser(description="Build competition splits")
    parser.add_argument("--demo", action="store_true",
                        help="generate a tiny synthetic dataset")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out_dir = Path(args.out)

    if args.demo:
        obs, samples, genomes = make_demo(args.seed)
        train, public, private = build_split(obs)
        write_layout(out_dir, train, public, private, samples)
        # write the demo genomes
        from Bio import SeqIO
        (out_dir / "genomes").mkdir(exist_ok=True)
        for ref, rec in genomes.items():
            SeqIO.write(rec, out_dir / "genomes" / f"{ref}.gb", "genbank")
    else:
        obs, samples, genomes_dir = load_raw()
        train, public, private = build_split(obs)
        write_layout(out_dir, train, public, private, samples, genomes_dir)


if __name__ == "__main__":
    main()
