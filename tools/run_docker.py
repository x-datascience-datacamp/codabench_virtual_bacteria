"""Build the image and run ingestion + scoring in it (local platform test).

Mirrors the platform layout: the phase directory (config + benchmark) is
mounted as ``/app/input_data``, a submission as ``/app/ingested_program``, and
the staged data as ``/app/data`` (read-only). Generate the data first with
``python tools/setup_data.py --demo`` (or your real loader).

Usage
-----
    python tools/run_docker.py [--data <host-data-dir>] [--submission <dir>]
"""

import argparse
from pathlib import Path

try:
    import docker
except ImportError:
    raise ImportError(
        "The 'docker' package is required: pip install docker."
    )

REPO = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docker ingestion test")
    parser.add_argument("--data",
                        default=str(REPO / "phases/main/input_data/data"),
                        help="Host data dir mounted read-only as /app/data")
    parser.add_argument("--submission", default=str(REPO / "solution"),
                        help="Submission folder (default: solution/)")
    args = parser.parse_args()

    client = docker.from_env()
    image = "tommoral/bacteria-compet:dev"
    print(f"Building {image}...")
    client.images.build(path=str(REPO), dockerfile="tools/Dockerfile",
                        tag=image)

    # The benchmark is baked into the image; the phase only carries config.
    phase = REPO / "phases/main/input_data"
    out = REPO / "ingestion_res"
    out.mkdir(exist_ok=True)
    print("Running ingestion...")
    logs = client.containers.run(
        image=image, remove=True, name="ingestion", user="root",
        command="python3 /compet/ingestion_program/ingestion.py",
        volumes=[
            f"{phase}:/app/input_data",
            f"{args.submission}:/app/ingested_program",
            f"{out}:/app/output",
            f"{args.data}:/app/data:ro",
        ],
    )
    print(logs.decode("utf-8"))

    score = REPO / "scoring_res"
    score.mkdir(exist_ok=True)
    print("Running scoring...")
    logs = client.containers.run(
        image=image, remove=True, name="scoring", user="root",
        command="python3 /compet/scoring_program/scoring.py",
        volumes=[
            f"{out}:/app/input/res",
            f"{score}:/app/output",
        ],
    )
    print(logs.decode("utf-8"))
    print("Docker container ran successfully.")
