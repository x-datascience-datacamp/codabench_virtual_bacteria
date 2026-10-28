"""Submission contract for the gene-expression competition.

A submission is a **valid benchopt solver**: a folder with a ``submission.py``
defining ``class Solver(GeneCompetSolver)``. All its Python code lives in that
single file (benchopt loads it by path, so sibling ``.py`` files cannot be
imported); the folder may also hold non-code files. Participants write plain
Python / PyTorch — no benchopt internals required. They implement:

- ``get_model(self, meta)`` -> model (required). Build the (untrained) model.
  ``meta["submission_dir"]`` points at the submission folder, to read shipped
  non-code files (weights, configs, precomputed tables).
- ``fit(self, model, train_df, samples_df, genomes_dir)`` (required). Engineer
  features from the three data sources and train the model. It runs on the
  platform before evaluation.

The trained model must expose::

    model.predict(test_df, samples_df, genomes_dir) -> ndarray, shape (N, 2)

with columns ``(mu, sigma)`` — see ``benchmark_utils.nb_loss``. The objective
calls it on the held-out test frames (features only; counts stay hidden).

``meta`` is a plain dict: the feature-column hints (``numeric_features``,
``categorical_features``), ``submission_dir``. The three data sources are
pandas DataFrames / a genomes directory path:

- ``train_df`` / ``test_df`` : one row per (gene x sample) observation;
- ``samples_df``            : experimental metadata, join key ``Run``
                              matched to observations' ``run_accession``;
- ``genomes_dir``           : folder of GenBank genomes, one
                              ``<genome_id>.gbff`` each (``None`` on the
                              synthetic dataset).
"""

import inspect
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
        self.train_df = train_df
        self.samples_df = samples_df
        self.genomes_dir = Path(genomes_dir) if genomes_dir is not None \
            else None
        # Non-code extra files ship next to submission.py.
        submission_dir = Path(inspect.getfile(type(self))).parent
        self.meta = {**meta, "submission_dir": submission_dir}
        self.model = self.get_model(self.meta)

    def run(self, _):
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
