#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python landscape_scan_free_diiid.py \
    --mode orchestrator \
    --n 65536 \
    --nprocs 5 \
    --oft-threads 1 \
    --weights 1e-4 \
    --evals-per-chunk 50
