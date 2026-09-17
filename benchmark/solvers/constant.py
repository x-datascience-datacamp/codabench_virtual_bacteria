"""Constant baseline — predicts the mean training count for every observation.

The leaderboard floor: any model that engineers features should beat it.
Depends only on the base stack, so it runs anywhere (``benchopt run .
-d Simulated -s Constant``).
"""

from benchmark_utils.base_solver import GeneCompetSolver
from benchmark_utils.baselines import ConstantModel


class Solver(GeneCompetSolver):

    name = "Constant"

    def get_model(self, meta):
        return ConstantModel()

    def fit(self, model, train_df, samples_df, genomes_dir):
        model.fit(train_df, samples_df, genomes_dir)
