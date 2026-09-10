#!/usr/bin/env python3
"""Fetch Time-Series-Library CSVs from the Hugging Face mirror (thuml/Time-Series-Library, CC BY 4.0).
    python scripts/download_data.py electricity weather traffic ETTh1 ETTm1
"""
import os, shutil, sys
from huggingface_hub import hf_hub_download

FILES = {"ETTh1": "ETT-small/ETTh1.csv", "ETTh2": "ETT-small/ETTh2.csv", "ETTm1": "ETT-small/ETTm1.csv", "ETTm2": "ETT-small/ETTm2.csv",
         "weather": "weather/weather.csv", "electricity": "electricity/electricity.csv", "traffic": "traffic/traffic.csv"}
os.makedirs("data", exist_ok=True)
for name in sys.argv[1:] or ["electricity"]:
    dst = f"data/{name}.csv"
    if os.path.exists(dst): print("have", dst); continue
    p = hf_hub_download("thuml/Time-Series-Library", FILES[name], repo_type="dataset"); shutil.copy(p, dst); print("got", dst)
