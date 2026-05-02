"""
Free-boundary cost-landscape Sobol scan, parallelized.

Constraints: theta1 = 20 deg, mu1 = mu2 = mu3 = 0.
Sobol-samples (theta2, theta3) at N points (default 262144).
Sweeps weight_fb. Each worker runs its own OFT_env / TokaMaker.
Checkpoints every CHECKPOINT_EVERY completions (resumable).

Run:
  python landscape_scan_free_diiid.py --nprocs 20
  python landscape_scan_free_diiid.py --nprocs 20 --n 1024 --weights 1e-2
"""

import argparse
import os
import shutil
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np
from scipy.stats import qmc

home_dir = os.path.expanduser("~")
oft_root_path = os.path.join(home_dir, "OpenFUSIONToolkit/install_release")
os.environ["OFT_ROOTPATH"] = oft_root_path
tokamaker_python_path = os.getenv("OFT_ROOTPATH")
if tokamaker_python_path is not None:
    sys.path.append(os.path.join(tokamaker_python_path, "python"))

from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk

from helper_fct import resize_polygon, update_boundary
from opt_comp_combined_boundary import _free_boundary_cost

NUM_COILS = 3
N_SAMPLES = 262144
WEIGHTS_FB = [1e-4, 1e-3, 1e-2, 1e-1]
THETA1_FIXED = 20.0
ANGULAR_BOUNDS = (10, 170)
SOBOL_SEED = 42
CHECKPOINT_EVERY = 200
OUT_DIR = "examples/comparisons/free_boundary_DIIID/landscape/coils:3"
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_W = {}


def _worker_init(eqdsk_path, oft_threads):
    """Run once per worker process. Builds OFT_env and shared geometry."""
    pid = os.getpid()
    tmp_dir = os.path.join(_BASE_DIR, "tmp", f"landscape_free_w{pid}")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    os.chdir(tmp_dir)

    eqdsk = read_eqdsk(eqdsk_path)
    fixed_LCFS = eqdsk["rzout"].copy()
    lim = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)

    myOFT = OFT_env(nthreads=oft_threads)

    fixed_mag_axis = np.array([1.77764093, -0.04014656])

    lim1 = update_boundary(r0=1.69, z0=0, a0=0.67, kappa=2, delta=0.8, squar=0.15, npts=1700)
    cand1 = resize_polygon(lim1, dx=0.1)
    lim2 = update_boundary(r0=1.94, z0=0, a0=0.95, kappa=1.55, delta=0.8, squar=0.15, npts=1700)
    cand2 = resize_polygon(lim2, dx=0.15)

    _W["myOFT"] = myOFT
    _W["eqdsk"] = eqdsk
    _W["fixed_mag_axis"] = fixed_mag_axis
    _W["fixed_LCFS"] = fixed_LCFS
    _W["lim"] = lim
    _W["cand1"] = cand1
    _W["cand2"] = cand2


def _worker_eval(args):
    idx, weight_fb, params = args
    try:
        c = _free_boundary_cost(
            params, _W["myOFT"], _W["eqdsk"], _W["fixed_mag_axis"], _W["fixed_LCFS"],
            _W["cand1"], _W["cand2"], _W["lim"], weight_fb, NUM_COILS,
        )
        return idx, float(c)
    except Exception:
        traceback.print_exc()
        return idx, 1e6


def _save_final(out_path, samples, cost, weight_fb, n_samples):
    valid = ~np.isnan(cost) & (cost < 1e6)
    if valid.any():
        masked = np.where(valid, cost, np.inf)
        idx_min = int(np.argmin(masked))
        min_theta2 = float(samples[idx_min, 0])
        min_theta3 = float(samples[idx_min, 1])
        min_cost = float(cost[idx_min])
    else:
        min_theta2 = float("nan")
        min_theta3 = float("nan")
        min_cost = float("nan")
    np.savez(
        out_path,
        samples=samples,
        cost=cost,
        theta1_anchor=THETA1_FIXED,
        mu_anchor=0.0,
        min_theta2=min_theta2,
        min_theta3=min_theta3,
        min_cost=min_cost,
        weight_fb=weight_fb,
        num_coils=NUM_COILS,
        angular_bounds=np.array(ANGULAR_BOUNDS),
        n_samples=n_samples,
        sobol_seed=SOBOL_SEED,
    )


def main(n_samples, weights, n_procs, oft_threads):
    eqdsk_path = os.path.join(_BASE_DIR, "examples/data/eqdsk/g192185.02440")
    os.makedirs(OUT_DIR, exist_ok=True)

    th_lo, th_hi = ANGULAR_BOUNDS
    sampler = qmc.Sobol(d=2, scramble=True, seed=SOBOL_SEED)
    samples = th_lo + sampler.random(n_samples) * (th_hi - th_lo)

    for w in weights:
        w_key = f"{w:.0e}"
        out_path = os.path.join(OUT_DIR, f"weight:{w_key}.npz")
        ckpt_path = out_path + ".ckpt.npz"

        if os.path.exists(ckpt_path):
            ck = np.load(ckpt_path, allow_pickle=False)
            cost = ck["cost"].copy()
            n_done = int((~np.isnan(cost)).sum())
            print(f"resuming weight={w_key}: {n_done}/{n_samples} already done")
        elif os.path.exists(out_path):
            print(f"weight={w_key} already complete at {out_path}, skipping")
            continue
        else:
            cost = np.full(n_samples, np.nan)

        pending_idx = np.where(np.isnan(cost))[0]
        if len(pending_idx) == 0:
            _save_final(out_path, samples, cost, w, n_samples)
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
            print(f"weight={w_key} done")
            continue

        tasks = [(int(i), float(w), np.array([THETA1_FIXED, samples[i, 0], samples[i, 1], 0.0, 0.0, 0.0])) for i in pending_idx]
        print(f"weight={w_key}: dispatching {len(tasks)} tasks across {n_procs} procs (oft_threads={oft_threads})")

        t0 = time.time()
        n_session = 0
        with Pool(processes=n_procs, initializer=_worker_init, initargs=(eqdsk_path, oft_threads)) as pool:
            for idx, c in pool.imap_unordered(_worker_eval, tasks, chunksize=4):
                cost[idx] = c
                n_session += 1
                if n_session % CHECKPOINT_EVERY == 0:
                    np.savez(ckpt_path, samples=samples, cost=cost)
                    elapsed = time.time() - t0
                    rate = n_session / elapsed
                    eta = (len(tasks) - n_session) / rate if rate > 0 else float("inf")
                    done_mask = ~np.isnan(cost)
                    n_failed = int((cost[done_mask] >= 1e6).sum())
                    print(f"  weight={w_key} {n_session}/{len(tasks)} fails={n_failed} rate={rate:.3f}/s elapsed={elapsed:.0f}s eta={eta:.0f}s")

        _save_final(out_path, samples, cost, w, n_samples)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        elapsed = time.time() - t0
        n_failed_final = int((cost[~np.isnan(cost)] >= 1e6).sum())
        print(f"weight={w_key} done in {elapsed:.0f}s, fails={n_failed_final}/{n_samples}, saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_SAMPLES)
    parser.add_argument("--nprocs", type=int, default=20)
    parser.add_argument("--oft-threads", type=int, default=1, help="OFT threads per worker (procs * threads should not exceed CPU count)")
    parser.add_argument("--weights", type=str, default=None, help="comma-separated, e.g. 1e-2,1e-3")
    args = parser.parse_args()

    weights = [float(x) for x in args.weights.split(",")] if args.weights else WEIGHTS_FB
    main(args.n, weights, args.nprocs, args.oft_threads)
