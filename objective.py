"""Objective for the bacterial gene-expression competition.

Predict RNA-seq ``counts`` for a (gene x sample) observation from three data
sources (gene table, genome sequence, sample/environment metadata). A
submission's model returns ``(mu, sigma)`` per observation; the ranking metric
is the mean **Negative-Binomial NLL** (lower is better), with MAE / R^2 on
``log1p(counts)`` reported alongside.

Evaluation runs on several held-out subsets — ``public`` (new gene x new
condition on a *known* species) and ``private`` (a *held-out* species, further
split into ``near`` / ``far`` groups). Metrics are emitted per subset as
``objective_<metric>_<subset>``; the phase config maps them to leaderboard
keys and hides the private columns.

Training is opt-in via the ``training`` parameter: a plain ``benchopt run`` is
inference-only (``train_df=None``), the competition phase selects
``Objective[training=True]`` so the solver's ``fit`` runs on the platform.
"""

from benchopt import BaseObjective

from benchmark_utils.nb_loss import nb_nll, log_mae, log_r2, check_predictions


class Objective(BaseObjective):

    name = "gene-expression-nb"
    url = ("https://github.com/x-datascience-datacamp/"
           "codabench_virtual_bacteria")

    requirements = ["scipy", "scikit-learn"]

    min_benchopt_version = "1.9.2"

    # Each solver runs once to completion (no convergence curve).
    sampling_strategy = "run_once"

    # Inference-only by default; the competition phase selects training=True.
    parameters = {"training": [False]}

    # ``benchopt test`` exercises the full contract, training included.
    test_config = {"training": True}

    def set_data(self, train_df, samples_df, genomes_dir, test_sets, **meta):
        # test_sets: ordered dict ``name -> (features_df, counts_array)``;
        # the counts stay here so the objective owns evaluation.
        self.train_df = train_df
        self.samples_df = samples_df
        self.genomes_dir = genomes_dir
        self.test_sets = test_sets
        self.meta = meta

    def skip(self, train_df=None, **data):
        if self.training and train_df is None:
            return True, "training=True needs a dataset with a train split"
        return False, None

    def get_objective(self):
        # Train data + resources go to the solver (only when training); the
        # test frames and all counts stay here.
        return dict(
            train_df=self.train_df if self.training else None,
            samples_df=self.samples_df,
            genomes_dir=self.genomes_dir,
            **self.meta,
        )

    def evaluate_result(self, model):
        scores = {}
        for name, (features_df, counts) in self.test_sets.items():
            preds = model.predict(features_df, self.samples_df,
                                  self.genomes_dir)
            mu, sigma = check_predictions(preds, len(counts))
            scores[f"nb_nll_{name}"] = nb_nll(counts, mu, sigma)
            scores[f"mae_{name}"] = log_mae(counts, mu)
            scores[f"r2_{name}"] = log_r2(counts, mu)
        return scores

    def get_one_result(self):
        # A trivial constant model, used by ``benchopt test`` to validate the
        # metric computation without running a real solver.
        from benchmark_utils.baselines import ConstantModel
        return dict(model=ConstantModel())
