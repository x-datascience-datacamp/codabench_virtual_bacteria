"""Objective for the bacterial gene-expression competition.

Predict RNA-seq ``counts`` for a (gene x sample) observation from three data
sources (gene table, genome sequence, sample/environment metadata). A
submission's model returns ``(mu, sigma)`` per observation; the ranking metric
is the mean **Negative-Binomial NLL** (lower is better), with MAE / R^2 on
``log1p(counts)`` reported alongside.

Evaluation runs on the held-out regions A to D (``benchmark_utils.split``),
each halved into a public part and a hidden private part. Metrics are emitted
per region as ``objective_<metric>_<region>`` (public) and
``objective_<metric>_<region>_private``, plus the ``challenge`` mean over A,
B and C of each half. The phase config maps them to leaderboard keys.

The solver always trains on the platform: ``get_objective`` hands it the
train data and resources, and its ``fit`` runs before evaluation.
"""

from benchopt import BaseObjective

from benchmark_utils.nb_loss import nb_nll, log_mae, log_r2, check_predictions


class Objective(BaseObjective):

    name = "gene-expression-nb"
    url = (
        "https://github.com/x-datascience-datacamp/"
        "codabench_virtual_bacteria"
    )

    requirements = ["scipy", "scikit-learn"]

    min_benchopt_version = "1.9.2"

    # Each solver runs once to completion (no convergence curve).
    sampling_strategy = "run_once"

    def set_data(self, train_df, samples_df, genomes_dir, test_sets, **meta):
        # test_sets: ordered dict ``name -> (features_df, counts_array)``;
        # the counts stay here so the objective owns evaluation.
        self.train_df = train_df
        self.samples_df = samples_df
        self.genomes_dir = genomes_dir
        self.test_sets = test_sets
        self.meta = meta

    def get_objective(self):
        # Train data + resources go to the solver; the test frames and all
        # counts stay here.
        return dict(
            train_df=self.train_df,
            samples_df=self.samples_df,
            genomes_dir=self.genomes_dir,
            **self.meta,
        )

    def evaluate_result(self, model):
        scores = {}
        for name, (features_df, counts) in self.test_sets.items():
            preds = model.predict(
                features_df, self.samples_df, self.genomes_dir
            )
            mu, sigma = check_predictions(preds, len(counts))
            scores[f"nb_nll_{name}"] = nb_nll(counts, mu, sigma)
            scores[f"mae_{name}"] = log_mae(counts, mu)
            scores[f"r2_{name}"] = log_r2(counts, mu)

        # Challenge aggregation: equal weighting on A, B and C, per half.
        for half in ("", "_private"):
            regions = [k + half for k in ("A", "B", "C")
                       if f"nb_nll_{k}{half}" in scores]
            for metric in ("nb_nll", "mae", "r2") if regions else ():
                scores[f"{metric}_challenge{half}"] = float(
                    sum(scores[f"{metric}_{r}"] for r in regions)
                    / len(regions))
        if "nb_nll_challenge" in scores:
            scores["value"] = scores["nb_nll_challenge"]
        elif scores:
            scores["value"] = next(iter(scores.values()))

        return scores

    def get_one_result(self):
        # A trivial constant model, used by ``benchopt test`` to validate the
        # metric computation without running a real solver.
        from benchmark_utils.baselines import ConstantModel

        return dict(model=ConstantModel())
