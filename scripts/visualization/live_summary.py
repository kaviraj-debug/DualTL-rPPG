import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import argparse, csv
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ref", type=float, required=True,
                help="reference heart rate in bpm, measured during the run")
args = ap.parse_args()

t, sm = [], []
for row in csv.DictReader(open("outputs/live_log.csv")):
    t.append(float(row["seconds_since_start"]))
    sm.append(float(row["bpm_smoothed"]))
t, sm = np.array(t), np.array(sm)
err = sm - args.ref

print(f"estimates: {len(sm)} | duration: {t[-1] - t[0]:.0f} s")
print(f"median {np.median(sm):.1f} | std {sm.std():.1f} | range {sm.min():.0f}-{sm.max():.0f} bpm")
print(f"reference {args.ref:.1f} | MAE {np.abs(err).mean():.1f} bpm | "
      f"within 5 bpm: {100 * np.mean(np.abs(err) <= 5):.0f}% | "
      f"within 10 bpm: {100 * np.mean(np.abs(err) <= 10):.0f}%")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 3.5))
    plt.plot(t, sm, label="Dual-TL live estimate")
    plt.axhline(args.ref, color="k", ls="--", label=f"reference ({args.ref:.0f} bpm)")
    plt.xlabel("time (s)"); plt.ylabel("heart rate (bpm)"); plt.legend(); plt.tight_layout()
    plt.savefig("outputs/plots/fig6_live_hr.png", dpi=150)
    print("Saved outputs/plots/fig6_live_hr.png")
except Exception as e:
    print("Plot skipped:", e)