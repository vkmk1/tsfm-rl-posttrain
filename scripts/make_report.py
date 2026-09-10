#!/usr/bin/env python3
"""Collect runs/**/eval.json into one markdown table:  python scripts/make_report.py runs/synth runs/etth1 > docs/results.md"""
import glob, json, os, sys

rows = []
for root in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(root, "*", "eval.json"))):
        e = json.load(open(f)); c = e["cfg"]; p, r = e["policy"], e["reference"]
        rows.append((os.path.basename(root), os.path.basename(os.path.dirname(f)), c["algo"], c["reward"], c["steps"], p, r))
if not rows: print("no eval.json found"); sys.exit()
keys = ("mase", "crps", "coverage80", "width", "dir_acc", "newsvendor", "r2_cond")
print("| env | run | algo | reward | steps | " + " | ".join(f"{k} (policy / native)" for k in keys) + " |")
print("|---|---|---|---|---|" + "---|" * len(keys))
for env, run, algo, reward, steps, p, r in rows:
    cells = " | ".join(f"{p.get(k, float('nan')):.3f} / {r.get(k, float('nan')):.3f}" for k in keys)
    print(f"| {env} | {run} | {algo} | {reward} | {steps} | {cells} |")
