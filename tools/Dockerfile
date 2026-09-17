# One image for the competition: it carries the environment and the two
# programs (for local runs). The benchmark and its benchmark_utils travel in
# the phase bundle mounted on /app/input_data.
#
#     docker build -f tools/Dockerfile -t <registry>/bacteria-compet:v1 .
#
# The same image serves the Codabench workers and the participants:
#
#     # stage a phase's data into a host folder (bind-mounted read-only as
#     # /app/data for the evaluation runs)
#     docker run -v <host-data>:/app/data -v <phase-bundle>:/app/input_data \
#         <image> python3 tools/setup_data.py --demo   # or your real loader
#
#     # reproduce the platform evaluation on a submission folder
#     docker run --gpus all -v <host-data>:/app/data \
#         -v <phase-bundle>:/app/input_data \
#         -v <submission>:/app/ingested_program -v <results>:/app/output \
#         <image> python3 /compet/ingestion_program/ingestion.py
#
# Python 3.12 base; torch from the PyTorch wheel index (the wheels bundle the
# CUDA runtime, so GPU works through the nvidia container runtime without a
# CUDA base image).
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# torch from the official wheel index with an explicit CUDA channel: cu126
# runs on any R525+ driver. Installed first so requirements.txt is satisfied.
ARG TORCH_CHANNEL=cu126
RUN pip install -U pip && pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/${TORCH_CHANNEL}

# All Python dependencies a submission may import (no install at submission
# time). torch is already present, so pip keeps the CUDA build above.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The programs + the benchopt benchmark they run (baked in, so the phase
# bundle only carries config; an image tag pins the exact evaluation).
COPY ingestion_program /compet/ingestion_program
COPY scoring_program /compet/scoring_program
COPY benchmark /compet/benchmark

# Data is bind-mounted read-only at /app/data by the compute worker; the
# phase bundle lands on /app/input_data, benchmark included.
ENV BENCHOPT_DATA_HOME=/app/data
VOLUME /app/data

# Fail the build if the stack is incoherent.
RUN python -c "import benchopt, pandas, sklearn, scipy, Bio, torch"
