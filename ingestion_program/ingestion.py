"""Ingestion program for the gene-expression competition.

A thin wrapper around ``benchopt run``. A submission is a benchopt solver:
``submission.py`` (a ``GeneCompetSolver``) plus any extra files it ships.
This program copies the benchmark (baked into the image at
``/compet/benchmark``) to a writable workdir, drops in the submission
solver(s), then execs::

    benchopt run <workdir> --config <input_data>/config.yaml -s <solver> \
        -r 1 --no-cache --no-plot --output submission

The phase ``config.yaml`` is a **native benchopt run config** (``dataset``,
``seed``, ``no_timeout``, ...) plus one competition-only key stripped before
the run (benchopt rejects unknown options): ``scoring`` (parsed by scoring.py
— the config is forwarded with the raw results parquet; all evaluation happens
in the objective, which always trains the solver). The data location is the
image's $BENCHOPT_DATA_HOME (/app/data, where the compute worker mounts the
staged data read-only). $BENCHOPT_DEBUG ensures any solver error aborts the
run.
"""

import os
from pathlib import Path

os.environ["BENCHOPT_DEBUG"] = "true"
# scikit-learn array-API dispatch needs scipy's, read at scipy import time.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import argparse  # noqa: E402
import ast  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import yaml  # noqa: E402

# benchopt artefacts + downloaded data, never copied to the workdir.
IGNORE = shutil.ignore_patterns(
    "outputs", "__cache__", "__pycache__", ".pytest_cache", "data"
)


def _solver_name(path):
    """The ``name`` of a top-level ``class Solver`` in *path*, read via ast.

    Static parsing avoids importing (executing) the submission here; benchopt
    imports it during the run.
    """
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name == "Solver":
            for stmt in node.body:
                if (isinstance(stmt, ast.Assign)
                        and any(getattr(t, "id", None) == "name"
                                for t in stmt.targets)
                        and isinstance(stmt.value, ast.Constant)):
                    return stmt.value.value
    return None


def install_submission_solvers(submission_dir, workdir):
    """Copy the submission's Solver files into the benchmark; return names."""
    names = []
    for path in sorted(submission_dir.glob("*.py")):
        name = _solver_name(path)
        if name is not None:
            shutil.copyfile(
                path, workdir / "solvers" / f"_submission_{path.stem}.py")
            names.append(name)
    if not names:
        raise SystemExit(
            f"No `class Solver` with a `name` found in {submission_dir}.")
    return names


def main(submission_dir, output_dir, benchmark_dir, input_dir):
    # Point the solvers at the submission folder (shipped extra files).
    os.environ["COMPET_SUBMISSION_DIR"] = str(submission_dir)
    workdir = Path(tempfile.mkdtemp(prefix="compet_run_")) / "benchmark"
    shutil.copytree(benchmark_dir, workdir, ignore=IGNORE)

    config = input_dir / "config.yaml"
    cfg = yaml.safe_load(config.read_text())
    # benchopt rejects unknown config keys: strip the competition-only one.
    cfg = {k: v for k, v in cfg.items() if k != "scoring"}
    run_config = workdir.parent / "run_config.yml"
    run_config.write_text(yaml.safe_dump(cfg))

    cmd = [
        "benchopt", "run", str(workdir),
        "-r", "1", "--no-cache", "--no-plot", "--no-html", "--no-display",
        "--output", "submission", "--config", str(run_config),
    ]
    names = install_submission_solvers(submission_dir, workdir)
    for name in names:
        cmd += ["-s", name]

    print(f"[ingestion] evaluating {', '.join(names)}", flush=True)
    start = time.time()
    subprocess.run(cmd, check=True)

    # Raw results + config for the scoring program, which only parses them.
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = max((workdir / "outputs").glob("submission*.parquet"))
    shutil.copyfile(result_file, output_dir / "results.parquet")
    (output_dir / "metadata.json").write_text(
        json.dumps({"duration": time.time() - start}))
    shutil.copyfile(config, output_dir / "config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Competition ingestion")
    parser.add_argument("--submission-dir", default="/app/ingested_program")
    parser.add_argument("--output-dir", default="/app/output")
    parser.add_argument("--input-data", default="/app/input_data",
                        help="Phase dir holding config.yaml")
    parser.add_argument("--benchmark-dir", default="/compet/benchmark",
                        help="benchopt benchmark (baked into the image)")
    args = parser.parse_args()

    main(
        Path(args.submission_dir),
        Path(args.output_dir),
        Path(args.benchmark_dir),
        Path(args.input_data),
    )
