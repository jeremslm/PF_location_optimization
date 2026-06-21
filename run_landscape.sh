#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python landscape_scan_free_diiid.py \
    --mode orchestrator \
    --n 65536 \
    --nprocs 30 \
    --oft-threads 1 \
    --evals-per-chunk 100 \
    --weights 1e-4 \
    --sobol-seed 1
