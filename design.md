# Design — Bacterial gene-expression competition (benchopt + Codabench)

Decision record and build plan. Modelled on the NeurIPS-2026 neural-interfaces
competition (`~/workspace/neural-compet`), which is the same architecture at
**4 tracks**; this competition is the **single-track** simplification of it.

## Task

Predict RNA-seq expression `counts` for a **gene × sample** observation, from
three input sources (already spelled out in `gene_expression_getting_started.ipynb`):

1. **Gene table** — one row per (gene × sample): genomic coordinates
   (`Chr`, `Start`, `End`, `Strand`), `Length`, normalization offsets
   (`log_length_ratio`, `log_size_factor`), foreign keys (`Geneid`,
   `run_accession`, `reference`).
2. **Genome** — per-species GenBank `.gb` files; coordinates index into the DNA
   (baseline uses the 200 bp upstream / promoter region).
3. **Sample metadata** (`samples.tsv`) — the environment: `Base Media`,
   `Culture Type`, `pH`, `Temperature`, `Growth Rate`, …

A submission outputs **(μ, σ) per observation**; ranking metric = mean
**Negative-Binomial NLL** of `counts ~ NB(μ, σ)` (lower is better), with
**MAE / R² on `log1p(counts)`** reported alongside.

## What we deliberately do NOT inherit from compet-eeg

The reference is heavy for two reasons, and **both are absent here** — this
competition is markedly simpler:

- **No multi-track machinery.** compet-eeg is 4 competitions in a monorepo
  (`tracks/<t>/…`, per-track bundles/pages/solutions). We have **one task** →
  one benchmark in a dedicated `benchmark/` folder, **no `track` objective
  parameter**, plain metric keys, one `competition.yaml`, one bundle.
- **No neuralbench/neuralset data stack.** compet-eeg's `nb_task.py` wraps
  neuralbench task configs, exca cluster infra, lazy EEG segmenters, one-hot
  target transforms, channel-map plumbing — hundreds of lines. Our data is
  **plain `.tsv` + GenBank `.gb`** loaded with pandas + Biopython in a short
  `datasets/bacteria.py`. (Note: their `nb_task.py` is *neuralbench*, not
  Negative-Binomial — naming coincidence; our NB loss is unrelated new code.)

What we *do* reuse, near-verbatim: the thin `benchopt run` **ingestion**, the
results-parquet **scoring**, the `CompetSolver` **contract**, the phase
`config.yaml` shape (native run config + `scoring.columns`), the single
**Docker image**, and `create_bundle.py`.

## Locked decisions (2026-09-17)

| # | Fork | Choice | Consequence |
|---|---|---|---|
| — | **Everything in a benchopt layer** | Yes | The benchmark *is* the competition; ingestion is a thin `benchopt run` wrapper; the metric lives in the Objective. |
| 1 | Platform trains? | **Yes — always train on platform** | Submission is *code* (a benchopt Solver) that engineers features + trains. The objective always hands the solver the train data and its `fit` always runs (no `training` toggle — locally and on the server we always train). |
| 2 | Phases | **Single phase, public + private columns** | One window, one leaderboard. Public metric shown live; private metric a `hidden: True` column used for the final rank. No switchover. Upgrade path: a sealed final phase (see Risks). |
| 3 | Metric location | **In the Objective** (`evaluate_result`) | `objective_*` columns land in the results parquet; the scoring program only maps columns → leaderboard keys. Participants get the exact number from a local `benchopt run`. |
| 4 | Workers | **Self-hosted GPU** | Deep sequence models (CNN/transformer promoter encoders) allowed; staged data mounted read-only. |
| 5 | Benchmark location | **Baked into the image** (`/compet/benchmark`) | Single track, single image: an image tag pins the exact evaluation, ingestion always resolves the benchmark, and the bundle carries only config + wrapper + pages. Updating the benchmark means an image rebuild. |
| 6 | Environment | **Frozen image** (pinned `requirements.txt`) | No install at submission time; a new dep = an image release. |

Data status: **multi-species data assembled** (on the microbiologists' side, not
yet in this repo). Submission contract: **code + raw data, participants own
feature engineering**.

## The split (the pedagogical core)

Three generalization axes → a difficulty spectrum that is *hard but doable*:

| Axis | Column | doable end | hard end |
|---|---|---|---|
| **Species** | `reference` (genome) | held-out species *close* to a train species (shared orthologs) | held-out species *far* in phylogeny |
| **Gene** | `feature_split` | seen gene | new gene (new ortholog group) |
| **Condition** | `condition_split` | seen environment | new environment |

Tiers produced offline by `tools/setup_data.py`:

- **TRAIN** — training species; the L-shaped region `NOT (feature_split==test AND
  condition_split==test)` (exactly today's `train_data.tsv`).
- **PUBLIC test** (live leaderboard feedback) — *same* training species, the
  double-held-out block (`feature_split==test AND condition_split==test`):
  new gene × new condition on a **known** species. Achievable, keeps students
  moving, immune to trivial memorization.
- **PRIVATE test** (final rank, hidden column) — **held-out species**, reported
  split into a **near** and a **far** group so the leaderboard breakdown shows
  *where* generalization holds.

Leakage guards `setup_data.py` enforces (`_check_leakage`). The public block is
the L-shape's *test-gene × test-condition* corner, so a test-gene **does**
appear in train (under train conditions) — that is the design, not a leak. What
must hold:
1. **`feature_split` is gene-level** — constant within an ortholog group, so a
   gene is a train-gene or a test-gene consistently (not half-and-half).
   (Near-species ortholog overlap across *species* is intended signal — what
   makes "near" doable.)
2. **`condition_split` is study-level** — whole studies held out, never split a
   study's replicates across train/test.
3. **No (gene × sample) cell** shared between train and public; train and
   private **species** disjoint.
4. **Normalization offsets** — `log_size_factor` / `log_length_ratio` must be
   per-observation from provided metadata only, never from test-set aggregates.

## Target layout (single-track distillation of neural-compet)

The benchopt benchmark lives in a dedicated `benchmark/` folder (clear
separation from the Codabench wrapper); the wrapper sits at the repo root.

```
benchmark/                           # the benchopt benchmark — `benchopt run benchmark`
  objective.py                       # Objective: NB-NLL + MAE/R² on public & private (near/far); always trains; get_one_result = constant baseline
  datasets/
    simulated.py                     # zero-dep smoke test — powers `benchopt run benchmark -d Simulated` + `benchopt test`
    bacteria.py                      # real: loads pre-split tables + samples + genomes; hands solver train+resources, keeps test counts
  solvers/
    baseline_nn.py                   # notebook baseline as a GeneCompetSolver (promoter composition + metadata → (μ,σ) MLP)
    constant.py                      # trivial floor (predict mean count, calibrated σ)
  test_config.py                     # CI hook: skip Bacteria (no data in CI)
  benchmark_utils/
    base_solver.py                   # GeneCompetSolver contract (get_model / fit; model.predict)
    nb_loss.py                       # Negative-Binomial NLL + MAE/R² on log1p — the metric
    baselines.py, features.py        # constant model; promoter extraction + tabular preprocessing (Biopython/sklearn)
competition.yaml                     # single phase, hidden private columns, self-hosted GPU docker_image
ingestion_program/{ingestion.py,metadata.yaml}   # thin `benchopt run` wrapper (benchmark from the image)
scoring_program/{scoring.py,metadata.yaml}        # results.parquet → scores.json
phases/main/input_data/config.yaml   # dataset: Bacteria; scoring.columns map
phases/reference_data/README.md
pages/{participate.md,timeline.md,terms.md}
solution/submission.py               # participant-facing baseline (a GeneCompetSolver)
tools/{Dockerfile,requirements.txt,create_bundle.py,setup_data.py,run_docker.py}
```

## Component contracts

**Data flow (benchopt):** `Dataset.get_data()` → `Objective.set_data()` →
`Objective.get_objective()` → `Solver.set_objective()` | `Solver.run()` |
`Solver.get_result()` → `Objective.evaluate_result()`.

- **`Bacteria.get_data()`** returns `train_df`, `samples_df`, `genomes_dir`,
  `public_test_df`, `public_counts`, `private_test_df`, `private_counts`
  (+ `private_group` near/far), `meta`.
- **`Objective.get_objective()`** hands the solver `train_df`, `samples_df`,
  `genomes_dir`, `meta`. Test frames and all counts stay in the objective.
- **`GeneCompetSolver`** (in `benchmark/benchmark_utils/base_solver.py`):
  - `get_model(self, meta)` — build the untrained model.
  - `fit(self, model, train_df, samples_df, genomes_dir)` — feature engineering
    + training (always runs, on the platform).
  - the returned model exposes
    `predict(test_df, samples_df, genomes_dir) -> np.ndarray (N, 2)` = (μ, σ).
- **`Objective.evaluate_result(model)`** — for each of {public, private_near,
  private_far}: `preds = model.predict(subset_df, samples_df, genomes_dir)`;
  compute NB-NLL + MAE + R² vs the kept counts. Fails loudly on wrong row count,
  NaN/inf, `μ<0`, `σ≤0`. Returns `nb_nll_public`, `nb_nll_private`,
  `nb_nll_private_near`, `nb_nll_private_far`, `mae_public`, `r2_public`, ….
- **Phase `config.yaml`** (native benchopt run config + one `scoring` key):
  ```yaml
  dataset: [Bacteria]
  seed: 42
  no_timeout: true
  scoring:
    columns:
      nb_nll:            objective_nb_nll_public        # shown, sorting asc
      mae:               objective_mae_public
      r2:                objective_r2_public
      nb_nll_private:      objective_nb_nll_private       # leaderboard column hidden:True
      nb_nll_private_near: objective_nb_nll_private_near
      nb_nll_private_far:  objective_nb_nll_private_far
  ```

## Implementation roadmap (status)

1. ✅ **benchopt layer on Simulated** — `objective.py`, `datasets/simulated.py`,
   `benchmark_utils/{nb_loss,base_solver,baselines}.py`, `solvers/constant.py`.
   `benchopt test . --skip-install` green, `flake8` clean, NB metric unit-tested.
2. ✅ **NN baseline** — `solvers/baseline_nn.py` + `benchmark_utils/features.py`
   (promoter extraction + tabular preprocessing). Beats the floor on Simulated
   (R² 0.93); deterministic (seeded).
3. ✅ **`datasets/bacteria.py`** — loads the staged layout (tables + genomes +
   sealed private). `test_config.py` skips it in CI.
4. ✅ **`tools/setup_data.py`** — one `build_split` (3 axes + `_check_leakage`)
   used by both the real `load_raw` (to fill in tomorrow) and a `--demo`
   generator; writes the on-disk layout incl. small GenBank files.
5. ✅ **Codabench wrapper** — `ingestion_program/` + `scoring_program/` (from
   neural-compet), `competition.yaml` (single phase, hidden private columns),
   phase `config.yaml`, pages, sample submission. End-to-end
   ingestion→scoring on demo data → `scores.json` with the expected keys.
6. ✅ **Image + bundle** — `tools/Dockerfile` (python:3.12 + torch cu126 +
   biopython/pandas/sklearn/scipy/benchopt), `requirements.txt`,
   `create_bundle.py` (follows symlinks), `run_docker.py`. CI: lint + benchopt
   test + bundle + native end-to-end + docker build.

**Remaining (needs the real data, tomorrow):** fill `load_raw` +
`TRAIN/NEAR/FAR_SPECIES` in `tools/setup_data.py`, confirm the assembled schema
matches the notebook columns (esp. `reference`, `feature_split`,
`condition_split`, an `ortholog_group` and `study` for the guards), then run
`python tools/setup_data.py` and validate a real `benchopt run benchmark -d Bacteria`.

## Risks / open points

- **Labels readable in the submission container.** Metric-in-objective (fork 3)
  means the private counts sit in the mounted phase data; a solver runs arbitrary
  code and could read them. Acceptable for an M2 class with a single phase +
  hidden column. Harden later by either (a) a **sealed final phase** with
  `hide_prediction_output`/`hide_score_output` (as neural-compet does) + a
  reproducibility audit of the top-k, or (b) moving *private* scoring into the
  scoring program with labels in `reference_data/` (not mounted into ingestion).
- **GPU worker cost.** Training on the platform × submission quota × GPU time.
  Cap `execution_time_limit`; set `max_submissions_per_day`.
- **Real-data schema drift.** The contract above follows the notebook columns;
  confirm the assembled multi-species tables match (esp. `reference` values,
  `feature_split`/`condition_split`, ortholog-group id for guard 1).
- **benchopt as the local dev loop.** Ship the starting kit so students run the
  exact server benchmark locally (`benchopt run benchmark -d Bacteria
  -s <their solver>`) — the one-code-path invariant.
```
