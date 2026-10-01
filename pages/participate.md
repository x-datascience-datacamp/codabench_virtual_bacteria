# How to participate

## The task

Predict the RNA-seq expression **`counts`** of a gene under a given
experimental condition, from three sources of information:

1. **gene table** — one row per (gene × sample): genomic coordinates,
   length, normalization offsets;
2. **genome** — GenBank `.gb` files; the coordinates index into the DNA
   sequence (e.g. the promoter region upstream of the gene);
3. **sample metadata** (`samples.tsv`) — the environment: growth medium,
   culture type, pH, temperature, growth rate.

Your model outputs two numbers per observation, **`mu`** (expected count,
≥ 0) and **`sigma`** (Negative-Binomial dispersion, > 0). Submissions are
ranked by the mean **Negative-Binomial negative log-likelihood** (lower is
better); MAE and R² on `log1p(counts)` are shown alongside.

## The split

- **Public columns** (live feedback): a *new gene × new condition* split on
  **known** species.
- **Private / final ranking** (hidden): **held-out species**, reported as
  *near* and *far* phylogenetic groups. Generalizing to distant species is
  the challenge — do not overfit the public columns.

## What you submit

A single **`submission.py`** defining `class Solver(GeneCompetSolver)`. All
your Python code must live in this one file: the platform loads it on its own,
so it cannot import other `.py` files you ship. You may add non-code files
next to it (model weights, configs, precomputed tables) and read them from
`meta["submission_dir"]` in `get_model`. You implement two methods:

```python
class Solver(GeneCompetSolver):
    name = "submission"

    def get_model(self, meta):
        # build your (untrained) model
        ...

    def fit(self, model, train_df, samples_df, genomes_dir):
        # engineer features from the three sources and train
        ...
```

Your trained model must expose
`predict(test_df, samples_df, genomes_dir) -> ndarray (N, 2)` returning
`(mu, sigma)`. The platform **trains your code** on identical hardware, then
evaluates it. See `solution/submission.py` for a working baseline
(feature builder + Ridge regression + dispersion calibration).

## Run it locally — same code as the server

```bash
pip install -r requirements.txt
python tools/setup_data.py --demo          # or the real data, when provided
BENCHOPT_DATA_HOME=phases/main/input_data/data \
    benchopt run benchmark -d Bacteria -s solution/submission.py
```

The number you get locally on the public split is the number the leaderboard
shows — the benchmark you run *is* the benchmark the server runs.

## Ideas to improve the baseline

Richer promoter representations (k-mers, learned sequence encoders), more
experimental metadata, a neural network (`benchmark/solvers/baseline_nn.py`), or
architectures that model gene × condition interactions.
