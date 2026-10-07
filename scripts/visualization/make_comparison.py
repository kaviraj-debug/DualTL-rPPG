import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import csv
from pathlib import Path
import numpy as np

Path("outputs/plots").mkdir(parents=True, exist_ok=True)

files = {"Dual-TL (ours)": "outputs/test_windows.csv"}
for p in sorted(Path("outputs/compare").glob("*_windows.csv")):
    files[p.name.replace("_windows.csv", "")] = str(p)

data = {}
for name, path in files.items():
    d = {}
    for r in csv.DictReader(open(path)):
        d[(r["subject"], int(r["start_frame"]))] = (float(r["hr_true_bpm"]), float(r["hr_pred_bpm"]))
    data[name] = d

num = lambda s: int("".join(c for c in s if c.isdigit()))
keys = sorted(set.intersection(*[set(d) for d in data.values()]), key=lambda k: (num(k[0]), k[1]))
print("methods:", list(data))
print("windows common to all methods:", len(keys), "(should be 837)")

ref = "Dual-TL (ours)"
hg = np.array([data[ref][k][0] for k in keys])
diff = max(np.abs(np.array([data[m][k][0] for k in keys]) - hg).max() for m in data)
print(f"largest ground-truth difference between methods: {diff:.3f} bpm (should be about 0)")

subs_of = np.array([k[0] for k in keys])
subs = sorted(set(subs_of), key=num)

res, per = {}, {}
for m in data:
    hp = np.array([data[m][k][1] for k in keys])
    e = hp - hg
    res[m] = (np.abs(e).mean(), np.sqrt((e ** 2).mean()), np.corrcoef(hp, hg)[0, 1])
    per[m] = {s: np.abs(e[subs_of == s]).mean() for s in subs}
order = sorted(res, key=lambda m: res[m][0])

print("\nTEST people, same windows for every method")
print(f"{'method':16s} | MAE (bpm) | RMSE (bpm) | Pearson r")
with open("outputs/compare_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "mae_bpm", "rmse_bpm", "pearson_r"])
    for m in order:
        print(f"{m:16s} | {res[m][0]:9.2f} | {res[m][1]:10.2f} | {res[m][2]:9.2f}")
        w.writerow([m, round(res[m][0], 2), round(res[m][1], 2), round(res[m][2], 2)])

print("\nper-subject MAE (bpm)")
print("subject    | " + " | ".join(f"{m[:10]:>10s}" for m in order))
with open("outputs/compare_per_subject.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["subject"] + order)
    for s in subs:
        print(f"{s:10s} | " + " | ".join(f"{per[m][s]:10.1f}" for m in order))
        w.writerow([s] + [round(per[m][s], 2) for m in order])

print()
for m in order:
    if m != ref:
        wins = sum(per[m][s] < per[ref][s] for s in subs)
        print(f"{m}: lower error than Dual-TL for {wins} of {len(subs)} people")

published = [("CHROM (paper's table)", 2.37, 4.91, 0.89), ("PulseGAN", 1.19, 2.10, 0.98),
             ("Dual-GAN", 0.44, 0.67, 0.99), ("Dual-TL (paper)", 0.17, 0.41, 0.99)]
with open("outputs/published_ubfc_from_paper.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "mae_bpm", "rmse_bpm", "pearson_r"])
    w.writerows(published)
print("\nReported in the paper for UBFC-rPPG (different protocol, NOT directly comparable):")
for m, a, b, c in published:
    print(f"  {m:22s} MAE {a:.2f} | RMSE {b:.2f} | r {c:.2f}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, idx, title in zip(ax, [0, 1], ["MAE (bpm)", "RMSE (bpm)"]):
        vals = [res[m][idx] for m in order]
        colors = ["tab:red" if m == ref else "tab:blue" for m in order]
        a.barh(range(len(order)), vals, color=colors)
        a.set_yticks(range(len(order))); a.set_yticklabels(order); a.invert_yaxis()
        for i, v in enumerate(vals):
            a.text(v + 0.05, i, f"{v:.2f}", va="center")
        a.set_title(title + ", lower is better")
    plt.tight_layout()
    plt.savefig("outputs/plots/fig7_model_comparison_test.png", dpi=150)
    plt.close()
    print("\nSaved outputs/plots/fig7_model_comparison_test.png")
except Exception as e:
    print("Plot skipped:", e)