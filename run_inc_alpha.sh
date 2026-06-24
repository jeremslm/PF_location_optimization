#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python opt_comp_comb_inc_alpha.py \
    --mode orchestrator \
    --weights 1e-4 1e-3 1e-2 1e-1 \
    --coils 2 3 4 5 \
    --ncpus 16 \
    --lambda 1e-6 \
    --nthreads 1 \
    --max-evals 10080 \
    --max-time 604800 \
    --random-state 2 \
    --folder comb_inc_alpha