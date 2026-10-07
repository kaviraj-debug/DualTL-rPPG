import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import csv, json
from pathlib import Path
import numpy as np
import torch
from scipy.signal import butter, filtfilt
from dataset import make_datasets
from model import DualTL

dev = "cuda"
Path("outputs/plots").mkdir(parents=True, exist_ok=True)

# ---- load the final model (fall back to the original checkpoint name) ----
ckpt = Path("checkpoints/FINAL_dualtl.pt")
if not ckpt.exists():
    ckpt = Path("checkpoints/split_filtered_D128L2_best.pt")
print("Using checkpoint:", ckpt)
model = DualTL(D=128, L=2).to(dev)
model.load_state_dict(torch.load(ckpt, map_location=dev))
model.eval()

_, _, test_ds = make_datasets(mode="filtered", step_train=15)

# ---- predict on the TEST subjects ----
names, starts, P, G = [], [], [], []
with torch.no_grad():
    for i in range(0, len(test_ds), 64):
        batch = [test_ds[j] for j in range(i, min(i + 64, len(test_ds)))]
        x = torch.stack([b[0] for b in batch]).to(dev)
        P.append(model(x).float().cpu().numpy())
        G.append(torch.stack([b[1] for b in batch]).numpy())
        names += [b[2] for b in batch]
        starts += [b[3] for b in batch]
P, G, names = np.concatenate(P), np.concatenate(G), np.array(names)

_fps = {}
def fps_of(n):
    if n not in _fps:
        _fps[n] = float(np.load(f"cache/{n}.npz")["fps"])
    return _fps[n]

def hr_fft(x, fs, lo=0.75, hi=2.5, nfft=4096):
    x = (x - x.mean()) * np.hanning(len(x))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    p = np.abs(np.fft.rfft(x, nfft)) ** 2
    m = (f >= lo) & (f <= hi)
    return 60 * f[m][np.argmax(p[m])]

Pf, hp, hg = [], [], []
for p, g, n in zip(P, G, names):
    fs = fps_of(n)
    b, a = butter(1, [0.75, 2.5], btype="bandpass", fs=fs)
    pf = filtfilt(b, a, p)
    Pf.append(pf)
    hp.append(hr_fft(pf, fs))
    hg.append(hr_fft(g, fs))
Pf, hp, hg = np.array(Pf), np.array(hp), np.array(hg)

# ---- tables (saved BEFORE plotting) ----
num = lambda s: int("".join(c for c in s if c.isdigit()))
subs = sorted(set(names), key=num)

with open("outputs/test_windows.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["subject", "start_frame", "hr_true_bpm", "hr_pred_bpm"])
    for n, s, t, p in zip(names, starts, hg, hp):
        w.writerow([n, s, round(float(t), 2), round(float(p), 2)])

maes = {}
with open("outputs/test_per_subject.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["subject", "windows", "hr_true_bpm", "hr_pred_bpm", "mae_bpm"])
    for n in subs:
        m = names == n
        maes[n] = float(np.abs(hp[m] - hg[m]).mean())
        w.writerow([n, int(m.sum()), round(float(hg[m].mean()), 1),
                    round(float(hp[m].mean()), 1), round(maes[n], 2)])

err = hp - hg
mae, rmse, r = np.abs(err).mean(), np.sqrt((err ** 2).mean()), np.corrcoef(hp, hg)[0, 1]
print(f"TEST: MAE {mae:.2f} | RMSE {rmse:.2f} | r {r:.2f}")
print("Saved outputs/test_windows.csv and outputs/test_per_subject.csv")

# ---- plots ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    print("Plotting is blocked or unavailable:", e)
    print("The CSV tables were saved. Tell me and we will plot another way.")
    raise SystemExit

# 1. training curves
hist = json.load(open("outputs/history_split_filtered_D128L2.json"))
ep = [h["epoch"] for h in hist]
tr = [h["train_loss"] for h in hist]
va = [h["val_loss"] for h in hist]
best = int(np.argmin(va))
plt.figure(figsize=(6, 4))
plt.plot(ep, tr, label="train loss")
plt.plot(ep, va, label="validation loss")
plt.axvline(ep[best], ls="--", color="gray", label=f"best epoch ({ep[best]})")
plt.xlabel("epoch"); plt.ylabel("negative Pearson loss"); plt.legend(); plt.tight_layout()
plt.savefig("outputs/plots/fig1_training_curves.png", dpi=150); plt.close()

# 2. predicted vs true heart rate
plt.figure(figsize=(5, 5))
plt.scatter(hg, hp, s=8, alpha=0.4)
lo, hi = min(hg.min(), hp.min()) - 3, max(hg.max(), hp.max()) + 3
plt.plot([lo, hi], [lo, hi], "k--", lw=1)
plt.xlim(lo, hi); plt.ylim(lo, hi)
plt.xlabel("true HR (bpm)"); plt.ylabel("predicted HR (bpm)")
plt.title(f"Test subjects: MAE {mae:.2f} bpm, r = {r:.2f}")
plt.tight_layout(); plt.savefig("outputs/plots/fig2_hr_scatter.png", dpi=150); plt.close()

# 3. example waveforms: best and worst test subject
best_s, worst_s = min(maes, key=maes.get), max(maes, key=maes.get)
z = lambda v: (v - v.mean()) / (v.std() + 1e-8)
fig, axes = plt.subplots(2, 1, figsize=(9, 5))
for ax, s, label in zip(axes, [best_s, worst_s], ["lowest error", "highest error"]):
    idx = np.where(names == s)[0]
    k = idx[len(idx) // 2]
    t = np.arange(P.shape[1]) / fps_of(s)
    ax.plot(t, z(G[k]), label="ground truth PPG")
    ax.plot(t, z(Pf[k]), label="Dual-TL prediction")
    ax.set_title(f"{s} ({label}): true {hg[k]:.0f} bpm, predicted {hp[k]:.0f} bpm")
    ax.set_xlabel("time (s)"); ax.set_ylabel("normalized")
    ax.legend(loc="upper right")
plt.tight_layout(); plt.savefig("outputs/plots/fig3_waveforms.png", dpi=150); plt.close()

# 4. per-subject MAE
plt.figure(figsize=(7, 4))
plt.bar(range(len(subs)), [maes[n] for n in subs])
plt.axhline(mae, ls="--", color="k", label=f"overall MAE {mae:.2f}")
plt.xticks(range(len(subs)), subs, rotation=45)
plt.ylabel("MAE (bpm)"); plt.legend(); plt.tight_layout()
plt.savefig("outputs/plots/fig4_per_subject_mae.png", dpi=150); plt.close()

# 5. model comparison (VALIDATION results from our earlier runs)
labels = ["Filtered input,\nplain loss", "Paper-style\ninput", "Filtered input,\nshift-10 loss", "Guess the\naverage"]
vals = [3.72, 7.96, 8.95, 15.49]
plt.figure(figsize=(7, 4))
plt.bar(labels, vals)
for i, v in enumerate(vals):
    plt.text(i, v + 0.2, f"{v:.2f}", ha="center")
plt.ylabel("MAE (bpm), validation subjects"); plt.tight_layout()
plt.savefig("outputs/plots/fig5_model_comparison.png", dpi=150); plt.close()

print("Saved 5 figures in outputs/plots/")