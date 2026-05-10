"""
Memory-efficient free-boundary cost-landscape Sobol scan.

Self-dispatching (mirror of mem_eff_comp_convergence.py):
  --mode orchestrator (default): Pool of N watchdogs. Each watchdog runs a
    subprocess chunk that does up to EVALS_PER_CHUNK evals then exits.
    Process exit reclaims OFT/Fortran heap so peak RAM stays bounded.
  --mode chunk: takes a JSON task file (list of {idx, theta2, theta3}) and
    a single weight_fb. Builds physics once, evaluates each task, prints
    one "RESULT idx cost reason" line per task, exits.

Constraints: theta1 = 20 deg, mu1 = mu2 = mu3 = 0. Sobol-samples (theta2, theta3).
Sweeps weight_fb. Checkpoints every CHECKPOINT_EVERY completions.
Output schema matches landscape_scan_diiid.py: samples, cost, failed (bool), ...
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import shutil
import subprocess
import sys
import time
import uuid
from multiprocessing import Pool

import numpy as np
from scipy.stats import qmc

home_dir = os.path.expanduser("~")
oft_root_path = os.path.join(home_dir, "OpenFUSIONToolkit/install_release")
os.environ["OFT_ROOTPATH"] = oft_root_path
tokamaker_python_path = os.getenv("OFT_ROOTPATH")
if tokamaker_python_path is not None:
    sys.path.append(os.path.join(tokamaker_python_path, "python"))

NUM_COILS = 3
N_SAMPLES = 262144
WEIGHTS_FB = [1e-4, 1e-3, 1e-2, 1e-1]
THETA1_FIXED = 20.0
ANGULAR_BOUNDS = (10, 170)
SOBOL_SEED = 42
CHECKPOINT_EVERY = 200
EVALS_PER_CHUNK = 100
CHUNK_TIMEOUT_S = 10800
OUT_DIR = "examples/comparisons/free_boundary_DIIID/landscape/coils:3"
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _build_physics_isolated(oft_threads, tmp_suffix):
    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk
    from helper_fct import resize_polygon, update_boundary

    eqdsk_path = os.path.join(_BASE_DIR, "examples/data/eqdsk/g192185.02440")
    tmp_dir = os.path.join(_BASE_DIR, "tmp", f"landscape_free_{tmp_suffix}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
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

    return myOFT, eqdsk, fixed_mag_axis, fixed_LCFS, lim, cand1, cand2, tmp_dir


def chunk_main(args):
    from opt_comp_combined_boundary import _free_boundary_cost

    weight = args.weight
    task_file = args.task_file
    oft_threads = args.oft_threads

    with open(task_file) as f:
        tasks = json.load(f)

    tmp_suffix = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    myOFT, eqdsk, fixed_mag_axis, fixed_LCFS, lim, cand1, cand2, tmp_dir = _build_physics_isolated(oft_threads, tmp_suffix)

    try:
        for t in tasks:
            idx = int(t["idx"])
            t2 = float(t["theta2"])
            t3 = float(t["theta3"])
            params = np.array([THETA1_FIXED, t2, t3, 0.0, 0.0, 0.0])
            cost_val, timing = _free_boundary_cost(
                params, myOFT, eqdsk, fixed_mag_axis, fixed_LCFS,
                cand1, cand2, lim, weight, NUM_COILS,
            )
            if timing is None:
                print(f"RESULT {idx} 1e6 fb_failure", flush=True)
            else:
                print(f"RESULT {idx} {float(cost_val):.17g} ok", flush=True)
    finally:
        os.chdir(_BASE_DIR)
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0


def _watchdog_eval(args):
    chunk_id, weight, idx_list, t2_list, t3_list, oft_threads = args
    tmp_tasks_dir = os.path.join(_BASE_DIR, "tmp")
    os.makedirs(tmp_tasks_dir, exist_ok=True)
    task_file = os.path.join(tmp_tasks_dir, f"landscape_chunk_tasks_{os.getpid()}_{chunk_id}.json")
    with open(task_file, "w") as f:
        json.dump([{"idx": int(i), "theta2": float(t2), "theta3": float(t3)}
                   for i, t2, t3 in zip(idx_list, t2_list, t3_list)], f)

    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--mode", "chunk",
        "--weight", repr(float(weight)),
        "--task-file", task_file,
        "--oft-threads", str(oft_threads),
    ]
    log_chunks = os.path.join(_BASE_DIR, OUT_DIR, "chunks.log")
    os.makedirs(os.path.dirname(log_chunks), exist_ok=True)
    received = {}
    timed_out = False
    stdout = ""
    stderr = ""
    rc = 0
    try:
        try:
            ret = subprocess.run(cmd, capture_output=True, text=True, timeout=CHUNK_TIMEOUT_S)
            stdout = ret.stdout or ""
            stderr = ret.stderr or ""
            rc = ret.returncode
        except subprocess.TimeoutExpired as e:
            so = e.stdout
            se = e.stderr
            stdout = (so.decode() if isinstance(so, bytes) else (so or "")) if so else ""
            stderr = (se.decode() if isinstance(se, bytes) else (se or "")) if se else ""
            rc = -9
            timed_out = True

        for line in stdout.splitlines():
            if line.startswith("RESULT "):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        ridx = int(parts[1])
                        rcost = float(parts[2])
                    except ValueError:
                        continue
                    rreason = parts[3]
                    received[ridx] = (rcost, rreason)

        with open(log_chunks, "a") as logf:
            logf.write(f"\n===== chunk_id={chunk_id} weight={weight:.0e} pid={os.getpid()} rc={rc} timed_out={timed_out} n_tasks={len(idx_list)} n_received={len(received)} {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            if stdout:
                logf.write(stdout)
            if stderr:
                logf.write("\n----- stderr -----\n")
                logf.write(stderr)

        results = []
        for i in idx_list:
            i_int = int(i)
            if i_int in received:
                results.append((i_int, received[i_int][0], received[i_int][1]))
            elif timed_out:
                results.append((i_int, 1e6, "chunk_timeout"))
            else:
                results.append((i_int, 1e6, "chunk_crash"))
        return results
    finally:
        if os.path.exists(task_file):
            os.remove(task_file)


def _save_final(out_path, samples, cost, failed, weight_fb, n_samples):
    valid = (~failed) & (~np.isnan(cost))
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
        failed=failed,
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


def orchestrator_main(ns):
    n_samples = ns.n
    weights = ns.weights
    n_procs = ns.nprocs
    oft_threads = ns.oft_threads
    evals_per_chunk = ns.evals_per_chunk

    out_dir_abs = os.path.join(_BASE_DIR, OUT_DIR)
    os.makedirs(out_dir_abs, exist_ok=True)

    th_lo, th_hi = ANGULAR_BOUNDS

    for w in weights:
        w_key = f"{w:.0e}"
        out_path = os.path.join(out_dir_abs, f"weight:{w_key}.npz")
        ckpt_path = out_path + ".ckpt.npz"

        if os.path.exists(ckpt_path):
            ck = np.load(ckpt_path, allow_pickle=False)
            samples = ck["samples"].copy()
            cost = ck["cost"].copy()
            n_samples_eff = samples.shape[0]
            if n_samples_eff != n_samples:
                print(f"weight={w_key}: ckpt has n_samples={n_samples_eff} but --n={n_samples}; using ckpt size to preserve work", flush=True)
            if "failed" in ck.files:
                failed = ck["failed"].copy().astype(bool)
            else:
                failed = (cost >= 1e6) & (~np.isnan(cost))
            n_done = int((~np.isnan(cost)).sum())
            print(f"resuming weight={w_key}: {n_done}/{n_samples_eff} already done (fb_fail={int(failed.sum())})", flush=True)
        elif os.path.exists(out_path):
            print(f"weight={w_key} already complete at {out_path}, skipping", flush=True)
            continue
        else:
            sampler = qmc.Sobol(d=2, scramble=True, seed=SOBOL_SEED)
            samples = th_lo + sampler.random(n_samples) * (th_hi - th_lo)
            cost = np.full(n_samples, np.nan)
            failed = np.zeros(n_samples, dtype=bool)
            n_samples_eff = n_samples

        pending_idx = np.where(np.isnan(cost))[0]
        if len(pending_idx) == 0:
            _save_final(out_path, samples, cost, failed, w, n_samples_eff)
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
            print(f"weight={w_key} done", flush=True)
            continue

        chunks = [pending_idx[i:i + evals_per_chunk] for i in range(0, len(pending_idx), evals_per_chunk)]
        tasks = [
            (cid, float(w),
             [int(i) for i in idx_arr],
             [float(samples[i, 0]) for i in idx_arr],
             [float(samples[i, 1]) for i in idx_arr],
             oft_threads)
            for cid, idx_arr in enumerate(chunks)
        ]
        print(f"weight={w_key}: dispatching {len(tasks)} chunks ({len(pending_idx)} evals) across {n_procs} procs (oft_threads={oft_threads}, evals_per_chunk={evals_per_chunk})", flush=True)

        n_done_initial = n_samples_eff - len(pending_idx)
        t0 = time.time()
        n_session = 0
        with Pool(processes=n_procs) as pool:
            for chunk_results in pool.imap_unordered(_watchdog_eval, tasks, chunksize=1):
                for idx, cost_val, reason in chunk_results:
                    cost[idx] = cost_val
                    failed[idx] = cost_val >= 1e6
                    n_session += 1
                    if n_session % CHECKPOINT_EVERY == 0:
                        np.savez(ckpt_path, samples=samples, cost=cost, failed=failed)
                        elapsed = time.time() - t0
                        rate = n_session / elapsed if elapsed > 0 else 0.0
                        eta = (len(pending_idx) - n_session) / rate if rate > 0 else float("inf")
                        n_failed = int(failed.sum())
                        n_total = n_done_initial + n_session
                        print(f"  weight={w_key} session={n_session}/{len(pending_idx)} total={n_total}/{n_samples_eff} fails={n_failed} rate={rate:.3f}/s elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

        _save_final(out_path, samples, cost, failed, w, n_samples_eff)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        elapsed = time.time() - t0
        n_failed_final = int(failed.sum())
        print(f"weight={w_key} done in {elapsed:.0f}s, fails={n_failed_final}/{n_samples_eff}, saved {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["orchestrator", "chunk"], default="orchestrator")
    parser.add_argument("--n", type=int, default=N_SAMPLES)
    parser.add_argument("--nprocs", type=int, default=20)
    parser.add_argument("--oft-threads", type=int, default=1, dest="oft_threads")
    parser.add_argument("--weights", type=str, default=None, help="comma-separated, e.g. 1e-2,1e-3")
    parser.add_argument("--evals-per-chunk", type=int, default=EVALS_PER_CHUNK, dest="evals_per_chunk")
    parser.add_argument("--weight", type=float, default=None, help="chunk mode: weight_fb")
    parser.add_argument("--task-file", type=str, default=None, dest="task_file", help="chunk mode: JSON task file")
    args = parser.parse_args()

    if args.mode == "chunk":
        if args.weight is None or args.task_file is None:
            parser.error("--mode chunk requires --weight and --task-file")
        sys.exit(chunk_main(args))

    args.weights = [float(x) for x in args.weights.split(",")] if args.weights else WEIGHTS_FB
    orchestrator_main(args)


if __name__ == "__main__":
    main()
