"""Train / A / B / C / D regions of the (gene x run) observations.

Each gene and each run has a status: ``train``, ``test_partial`` (also seen
in training) or ``test_full`` (never seen). An observation's region follows
from the pair, see ``REGION_TABLE``:

- A: gene and run both seen in training, never together;
- B: new gene;
- C: new run;
- D: new gene in a new run.

The letters follow the difficulty: the genome describes a new gene, while a
new run only has its curated conditions.

The evaluation release ships the real statuses. The student releases ship
none, and ``simulate_status`` draws them with a fixed seed, so local scores
follow the same protocol.

Each region is then halved into a public part, shown on the leaderboard, and
a hidden private part, which ranks: by observation for A, whose genes and
runs are all seen in training, by gene for B and D, by run for C. A new gene
falls in the same half in B and D.
"""

import numpy as np
import pandas as pd

REGIONS = ("A", "B", "C", "D")
STATUSES = ["train", "test_partial", "test_full"]
REGION_TABLE = np.array([
    # run:  train    test_partial  test_full     # gene:
    ["Train", "Train", "C"],                     # train
    ["Train", "A", "C"],                         # test_partial
    ["B", "B", "D"],                             # test_full
])
STATUS_COLS = ["feature_status", "run_status", "status"]
GENE_KEYS = ["genome_id", "feature_id"]
RUN_KEYS = ["genome_id", "run_accession"]
HALF_KEYS = {"A": GENE_KEYS + ["run_accession"], "B": GENE_KEYS,
             "C": RUN_KEYS, "D": GENE_KEYS}


def simulate_status(df, seed=42):
    """Hold out 20% of the rows: 80% ``test_partial``, 20% ``test_full``."""
    rng = np.random.default_rng(seed)
    held_out = rng.permutation(len(df))[:int(round(0.2 * len(df)))]
    status = np.full(len(df), "train", dtype=object)
    status[held_out] = "test_partial"
    status[held_out[:int(np.ceil(0.2 * len(held_out)))]] = "test_full"
    return df.assign(status=status)


def attach(obs, table, keys, cols):
    """Left-join ``cols`` of ``table`` onto ``obs``, matching category codes.

    Casting the keys to the observation dtypes keeps the join on integer codes
    instead of strings, which matters for the ~1e8-row ``train`` release.
    """
    table = table[keys + cols].astype({k: obs[k].dtype for k in keys})
    # Keys absent from obs cast to NaN. A duplicated key would duplicate its
    # observations.
    return obs.merge(table.dropna(subset=keys), on=keys, how="left",
                     validate="many_to_one")


def assign_regions(obs, feat_df, cond_df):
    """Add the region of each observation as its ``status`` column.

    Uses the gene and run ``status`` of ``feat_df`` / ``cond_df``; genes or
    runs without one count as ``train``.
    """
    obs = attach(obs, feat_df.rename(columns={"status": "fs"}),
                 GENE_KEYS, ["fs"])
    obs = attach(obs, cond_df.rename(columns={"status": "rs"}),
                 RUN_KEYS, ["rs"])
    # Code -1 (no status) clips to 0, i.e. ``train``.
    fs = pd.Categorical(obs.pop("fs"), categories=STATUSES).codes
    rs = pd.Categorical(obs.pop("rs"), categories=STATUSES).codes
    obs["status"] = REGION_TABLE[np.maximum(fs, 0), np.maximum(rs, 0)]
    return obs


def split_release(obs, feat_df, cond_df, simulate):
    """Return ``{"train", "A", ..., "D": frame}`` and the annotations.

    Frames keep ``readcount`` and gain the ``log_length_ratio`` and
    ``log_size_factor`` offsets. Statuses are dropped from every output.
    """
    obs = obs.astype({k: "category" for k in GENE_KEYS + RUN_KEYS})
    if simulate:
        feat_df, cond_df = simulate_status(feat_df), simulate_status(cond_df)
        obs = assign_regions(obs.drop(columns=STATUS_COLS, errors="ignore"),
                             feat_df, cond_df)
    obs = attach(obs, feat_df, GENE_KEYS, ["log_length_ratio"])
    obs = attach(obs, cond_df, RUN_KEYS, ["log_size_factor"])

    region = obs.pop("status")
    obs = obs.drop(columns=STATUS_COLS, errors="ignore")
    frames = {name: obs[region == name].reset_index(drop=True)
              for name in ("Train",) + REGIONS}
    frames["train"] = frames.pop("Train")
    feat_df = feat_df.drop(columns=STATUS_COLS, errors="ignore")
    cond_df = cond_df.drop(columns=STATUS_COLS, errors="ignore")
    return frames, feat_df, cond_df


def is_private(df, region):
    """Whether each observation of ``region`` falls in its private half."""
    keys = df[HALF_KEYS[region]]
    return pd.util.hash_pandas_object(keys, index=False).to_numpy() % 2 == 1
