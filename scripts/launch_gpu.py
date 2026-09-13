#!/usr/bin/env python3
"""Run a list of cells across GPUs, one process per GPU, round-robin queue.

    python3 scripts/launch_gpu.py configs/gpu_matrix.txt --gpus 0,1,2,3,4,5,6,7 [--dry]
    python3 scripts/launch_gpu.py configs/gpu_matrix.txt --gpus 0-15 --out runs/gpu

Each non-comment line of the cell file is `<name> <run_matrix.py arguments>`; `--out` is added automatically as
<out>/<name>. Logs go to <out>/<name>/stdout.log. Cells already holding an eval.json are skipped (resume-safe).
The matrix is embarrassingly parallel: each cell is one GPU, no DDP needed (TimesFM-3 at context 256 fits in
under 10 GB with batch 32 and bf16).
"""
from __future__ import annotations

import argparse, os, subprocess, sys, time


def parse_gpus(s):
    out = []
    for part in s.split(","):
        if "-" in part: a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part.strip(): out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cells"); ap.add_argument("--gpus", default="0"); ap.add_argument("--out", default="runs/gpu"); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--extra", default="", help="arguments appended to every cell, e.g. '--steps 4000 --amp'")
    a = ap.parse_args(); gpus = parse_gpus(a.gpus); root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cells = []
    for line in open(a.cells):
        line = line.split("#")[0].strip()
        if not line: continue
        name, *args = line.split(); out = os.path.join(a.out, name)
        if os.path.exists(os.path.join(out, "eval.json")): print("skip (done):", name); continue
        cells.append((name, args + a.extra.split() + ["--out", out]))
    print(f"{len(cells)} cells on GPUs {gpus}"); running = {}
    while cells or running:
        for gpu in gpus:
            if gpu in running and running[gpu][1].poll() is not None:
                name, p = running.pop(gpu); print(f"[{time.strftime('%H:%M')}] gpu{gpu} done {name} rc={p.returncode}", flush=True)
            if gpu not in running and cells:
                name, args = cells.pop(0); os.makedirs(os.path.join(a.out, name), exist_ok=True)
                cmd = [sys.executable, "scripts/run_matrix.py", *args]; print(f"[{time.strftime('%H:%M')}] gpu{gpu} start {name}: {' '.join(cmd)}", flush=True)
                if a.dry: continue
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": root}
                running[gpu] = (name, subprocess.Popen(cmd, cwd=root, env=env, stdout=open(os.path.join(a.out, name, "stdout.log"), "a"), stderr=subprocess.STDOUT))
        if a.dry and not cells: break
        time.sleep(10)
    print("all cells finished")


if __name__ == "__main__":
    main()
