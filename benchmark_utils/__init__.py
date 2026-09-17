"""Shared code for the gene-expression competition benchmark.

benchopt loads this package by path and makes it importable as
``from benchmark_utils import ...`` from any objective/dataset/solver, and
pickles it *by value* so it travels to remote/distributed workers. One module
per concern (``nb_loss``, ``base_solver``, ``baselines``, ``features``,
``data``) so importing one topic does not pull in the others' dependencies.
Keep this ``__init__`` empty.
"""
