# Bacterial gene-expression competition (benchopt + Codabench)

A Codabench competition whose evaluation **is** a benchopt benchmark:
participants run the exact same code locally that the server runs. The task is
to predict RNA-seq expression `counts` for a gene × sample observation from its
genome sequence, gene description and experimental (environment) metadata, and
to **generalize to held-out bacterial species**.

See `gene_expression_getting_started.ipynb` for an end-to-end walkthrough.

## The benchmark (`benchmark/`)

The self-contained benchopt benchmark — run it directly with
`benchopt run benchmark`:

- `benchmark/objective.py` — the task + metric (Negative-Binomial NLL, MAE/R²)
  per region, on the public and hidden private halves; always trains.
- `benchmark/datasets/` — `simulated.py` (zero-dependency smoke test) and
  `bacterial_expression.py` (downloads the real data from Zenodo in
  `prepare` and writes its train / A / B / C / D regions).
- `benchmark/solvers/` — `constant.py` (floor), `baseline_nn.py` (NN).
- `benchmark/benchmark_utils/` — the metric, the `GeneCompetSolver` submission
  contract, feature engineering, shared helpers.

## The competition wrapper (repo root)

- `competition.yaml` — the single-phase competition (public columns +
  hidden private columns).
- `ingestion_program/` — a thin `benchopt run` wrapper (runs the submission).
- `scoring_program/` — parses the benchopt results into `scores.json`.
- `phases/main/input_data/config.yaml` — the phase config (a native benchopt
  run config + a `scoring` column map).
- `pages/`, `solution/submission.py` — participant pages and the sample
  submission.
- `tools/` — `setup_data.py` (build the release), `Dockerfile`, `requirements.txt`,
  `create_bundle.py`, `run_docker.py`.

## Quickstart

```bash
pip install -r requirements.txt

# download the data, fit on train and score on the eval regions
# (mode=train for all the public data)
benchopt prepare benchmark -d "BacterialExpression[mode=debug]"
benchopt run benchmark -d "BacterialExpression[mode=debug]" \
    -s solution/submission.py

# design-level tests + zero-dependency smoke test
benchopt test benchmark --skip-install
benchopt run benchmark -d Simulated -s Constant
```

## Build the Codabench bundle

```bash
python tools/create_bundle.py     # -> bundle.zip (upload to Codabench)
```

The bundle ships the config, programs, pages and the **benchmark** (in the
phase `input_data`, so a benchmark change needs no image rebuild). The
**data** (including the sealed private split) is staged separately onto the
compute worker at `$BENCHOPT_DATA_HOME` (`/app/data`), so it never enters the
public artifact.
