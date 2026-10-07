import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import json
import numpy as np

split = json.load(open("split.json"))
hr = {}
for g in ["train", "val", "test"]:
    hr[g] = {n: float(np.load(f"cache/{n}.npz")["hr"]) for n in split[g]}

for g in ["train", "val", "test"]:
    v = np.array(list(hr[g].values()))
    print(f"{g:5s}: {len(v)} people | mean {v.mean():.1f} | min {v.min():.0f} | max {v.max():.0f} | "
          f"above 105 bpm: {(v > 105).sum()} | above 115 bpm: {(v > 115).sum()}")

print("\nTraining people sorted by average heart rate (bpm):")
for n, v in sorted(hr["train"].items(), key=lambda kv: kv[1]):
    print(f"  {n:10s} {v:6.1f}")

print("\nsubject11 (validation):", round(hr["val"].get("subject11", float("nan")), 1))