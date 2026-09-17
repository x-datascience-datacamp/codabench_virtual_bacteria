"""Submission contract for the gene-expression competition.

A submission is a **valid benchopt solver**: a folder with a ``submission.py``
defining ``class Solver(GeneCompetSolver)``. Participants write plain Python /
PyTorch — no benchopt internals required. They implement:

- ``get_model(self, meta)`` -> model (required). Build the (untrained) model.
  ``meta["submission_dir"]`` points at the submission folder if you ship extra
  files (precomputed tables, weights).
- ``fit(self, model, train_df, samples_df, genomes_dir)`` (required for the
  competition). Engineer features from the three data sources and train.
  Runs only when the objective's ``training`` parameter is selected — which
  the competition phase config does (``Objective[training=True]``). A plain
  local ``benchopt run`` is inference-only (``train_df=None``), mirroring a
  server that only runs prediction.

The trained model must expose::

    model.predict(test_df, samples_df, genomes_dir) -> ndarray, shape (N, 2)

with columns ``(mu, sigma)`` — see ``benchmark_utils.nb_loss``. The objective
calls it on the held-out test frames (features only; counts stay hidden).

``meta`` is a plain dict: the feature-column hints (``numeric_features``,
``categorical_features``), ``device``, ``submission_dir``. The three data
sources are pandas DataFrames / a genomes directory path:

- ``train_df`` / ``test_df`` : one row per (gene x sample) observation;
- ``samples_df``            : experimental metadata, join key ``Run``
                              matched to observations' ``run_accession``;
- ``genomes_dir``           : folder of GenBank ``.gb`` genome files (may be
                              ``None`` on the synthetic dataset).
"""

import inspect
import os
from pathlib import Path

from benchopt import BaseSolver


class GeneCompetSolver(BaseSolver):
    """Base class for all submissions (and the built-in baselines).

    numpy / pandas / scipy / scikit-learn / torch come from the benchmark
    environment (the objective's requirements); submissions declare only their
    own extras, e.g. ``requirements = ["pip::my-model-pkg"]``.
    """

    requirements = []

    sampling_strategy = "run_once"

    def set_objective(self, train_df, samples_df, genomes_dir, **meta):
        # train_df is None on inference-only runs (objective training=False);
        # the solver discovers it can train from the argument, not a flag.
        self.train_df = train_df
        self.samples_df = samples_df
        self.genomes_dir = Path(genomes_dir) if genomes_dir is not None \
            else None
        # Extra files ship next to submission.py; ingestion points
        # COMPET_SUBMISSION_DIR there, else default to the solver file's dir.
        submission_dir = os.environ.get("COMPET_SUBMISSION_DIR")
        if submission_dir is None:
            submission_dir = Path(inspect.getfile(type(self))).parent
        self.meta = {**meta, "submission_dir": Path(submission_dir)}
        self.model = self.get_model(self.meta)

    def run(self, _):
        if self.train_df is not None:
            self.fit(self.model, self.train_df, self.samples_df,
                     self.genomes_dir)

    def get_result(self):
        return dict(model=self.model)

    # --- to be implemented by the submission ---------------------------

    def get_model(self, meta):
        """Build and return the (untrained) model."""
        raise NotImplementedError

    def fit(self, model, train_df, samples_df, genomes_dir):
        """Engineer features and train the model in place."""
