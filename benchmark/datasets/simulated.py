"""Zero-dependency simulated dataset — the always-available smoke test.

Stands in for the real bacterial task so any model can be tested without data:
synthetic (gene x sample) observations whose ``counts`` are Negative-Binomial
draws driven by a few gene features, sample/environment features and a
per-observation size factor. It mirrors the real three-source schema (a gene
table, a ``samples_df`` joined on ``run_accession`` -> ``Run``, and an optional
genomes directory — ``None`` here) and the tiered split (a training tier plus
held-out ``public`` / ``private_near`` / ``private_far`` tiers with a
generalization gap: each held-out tier uses fresh genes and samples, and the
private tiers shift the count scale).

Only depends on numpy + pandas (the benchmark's base stack), so it powers
``benchopt run . -d Simulated`` and ``benchopt test``.
"""

import numpy as np
import pandas as pd
from benchopt import BaseDataset

NUM_GENE = ["Length", "log_length_ratio", "log_size_factor"]
NUM_SAMPLE = ["pH", "Temperature (C)", "Growth Rate (1/hr)"]
CAT = ["Base Media", "Culture Type"]
MEDIA = ["M9", "LB", "MOPS"]
CULTURE = ["batch", "chemostat"]


def _std(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


class Dataset(BaseDataset):

    name = "Simulated"

    requirements = []

    parameters = {"n_obs": [500]}
    test_parameters = {"n_obs": [80]}

    def _make_samples(self, rng, n, prefix):
        return pd.DataFrame({
            "Run": [f"{prefix}{i:04d}" for i in range(n)],
            "Base Media": rng.choice(MEDIA, n),
            "Culture Type": rng.choice(CULTURE, n),
            "pH": rng.normal(7.0, 0.4, n),
            "Temperature (C)": rng.choice([30.0, 37.0, 42.0], n),
            "Growth Rate (1/hr)": np.abs(rng.normal(0.6, 0.2, n)),
        })

    def _make_genes(self, rng, n, prefix, reference):
        length = rng.integers(200, 4000, n)
        start = rng.integers(1, 4_000_000, n)
        return pd.DataFrame({
            "Geneid": [f"{prefix}{i:05d}" for i in range(n)],
            "reference": reference,
            "Chr": "chr1",
            "Start": start,
            "End": start + length,
            "Strand": rng.choice(["+", "-"], n),
            "Length": length.astype(float),
            "log_length_ratio": np.log(length) - np.log(length.mean()),
        })

    def _make_tier(self, rng, n_obs, genes, samples, shift,
                   feature_split, condition_split):
        """Assemble ``n_obs`` observations and their NB counts for one tier."""
        obs = genes.sample(n_obs, replace=True, random_state=rng.integers(1e9))
        obs = obs.reset_index(drop=True)
        s = samples.sample(n_obs, replace=True,
                           random_state=rng.integers(1e9)).reset_index(
            drop=True)
        obs["run_accession"] = s["Run"].values
        # Same composite-ID columns as the real BacterialExpression schema
        obs["genome_id"] = obs["reference"]
        obs["feature_id"] = obs["Geneid"]
        obs["log_size_factor"] = rng.normal(0.0, 0.3, n_obs)
        obs["feature_split"] = feature_split
        obs["condition_split"] = condition_split

        # Count mean strongly driven by gene + sample features and the size
        # factor (high SNR so a feature model clearly beats the floor).
        log_mu = (
            3.0 + shift
            + 0.8 * _std(obs["log_length_ratio"].values)
            + 0.6 * _std(obs["Length"].values)
            - 1.0 * _std(s["pH"].values)
            + 1.2 * _std(s["Growth Rate (1/hr)"].values)
            + 0.6 * (s["Temperature (C)"].values == 37.0)
            + 0.8 * (s["Base Media"].values == "LB")
            + obs["log_size_factor"].values
            + rng.normal(0.0, 0.15, n_obs)
        )
        mu = np.exp(np.clip(log_mu, -2.0, 9.0))
        r = 8.0  # mild overdispersion so the conditional mean is informative
        counts = rng.negative_binomial(r, r / (r + mu)).astype(np.int64)
        return obs, counts

    def get_data(self):
        rng = np.random.default_rng(self.get_seed())
        n = self.n_obs

        # One sample pool + one gene pool per tier -> held-out tiers use fresh
        # genes and samples (a genuine new-gene x new-condition gap).
        s_train = self._make_samples(rng, 25, "SRRtr")
        s_pub = self._make_samples(rng, 15, "SRRpub")
        s_near = self._make_samples(rng, 15, "SRRnear")
        s_far = self._make_samples(rng, 15, "SRRfar")
        g_train = self._make_genes(rng, 60, "g_tr_", "GCF_train")
        g_pub = self._make_genes(rng, 40, "g_pub_", "GCF_train")
        g_near = self._make_genes(rng, 40, "g_near_", "GCF_near")
        g_far = self._make_genes(rng, 40, "g_far_", "GCF_far")

        train_obs, train_counts = self._make_tier(
            rng, n, g_train, s_train, shift=0.0,
            feature_split="train", condition_split="train")
        train_df = train_obs.copy()
        train_df["counts"] = train_counts

        pub_obs, pub_counts = self._make_tier(
            rng, n // 3, g_pub, s_pub, shift=0.0,
            feature_split="test", condition_split="test")
        near_obs, near_counts = self._make_tier(
            rng, n // 4, g_near, s_near, shift=0.3,
            feature_split="test", condition_split="test")
        far_obs, far_counts = self._make_tier(
            rng, n // 4, g_far, s_far, shift=1.0,
            feature_split="test", condition_split="test")

        # Held-out frames carry no counts (kept by the objective for scoring).
        private_obs = pd.concat([near_obs, far_obs], ignore_index=True)
        private_counts = np.concatenate([near_counts, far_counts])
        test_sets = {
            "public": (pub_obs, pub_counts),
            "private": (private_obs, private_counts),
            "private_near": (near_obs, near_counts),
            "private_far": (far_obs, far_counts),
        }

        samples_df = pd.concat([s_train, s_pub, s_near, s_far],
                               ignore_index=True)

        return dict(
            train_df=train_df,
            samples_df=samples_df,
            genomes_dir=None,
            test_sets=test_sets,
            numeric_features=NUM_GENE + NUM_SAMPLE,
            categorical_features=CAT,
        )
