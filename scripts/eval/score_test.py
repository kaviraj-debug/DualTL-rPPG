import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import csv, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from dataset import UBFCWindows
from model import DualTL

dev = "cuda"
Path("outputs").mkdir(exist_ok=True)


class CNN1D(nn.Module):
    def __init__(self, c_in=189, dropout=0.2):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 7, padding=3), nn.BatchNorm1d(o),
                                 nn.ReLU(), nn.Dropout(dropout))
        self.net = nn.Sequential(block(c_in, 64), block(64, 64), block(64, 32), nn.Conv1d(32, 1, 1))

    def forward(self, m):
        return self.net(m.flatten(1, 2)).squeeze(1)


class GRUNet(nn.Module):
    def __init__(self, c_in=189, h=64, dropout=0.2):
        super().__init__()
        self.inp = nn.Linear(c_in, h)
        self.gru = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.out = nn.Linear(2 * h, 1)

    def forward(self, m):
        x = m.flatten(1, 2).transpose(1, 2)
        y, _ = self.gru(torch.relu(self.inp(x)))
        return self.out(y).squeeze(-1)


split = json.load(open("split.json"))
test_ds = UBFCWindows([Path("cache") / f"{n}.npz" for n in split["test"]], mode="filtered")
print("test windows:", len(test_ds))

_fps = {}
def get_fs(n):
    if n not in _fps:
        _fps[n] = float(np.load(f"cache/{n}.npz")["fps"])
    return _fps[n]


def hr_fft(x, fs, lo=0.75, hi=2.5, nfft=4096):
    x = (x - x.mean()) * np.hanning(len(x))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    p = np.abs(np.fft.rfft(x, nfft)) ** 2
    m = (f >= lo) & (f <= hi)
    return 60 * f[m][np.argmax(p[m])]


def predict(model):
    model.eval()
    names, starts, P, G = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(test_ds), 64):
            batch = [test_ds[j] for j in range(i, min(i + 64, len(test_ds)))]
            x = torch.stack([b[0] for b in batch]).to(dev)
            P.append(model(x).float().cpu().numpy())
            G.append(torch.stack([b[1] for b in batch]).numpy())
            names += [b[2] for b in batch]; starts += [b[3] for b in batch]
    P, G = np.concatenate(P), np.concatenate(G)
    out = {}
    for p, g, n, s in zip(P, G, names, starts):
        fs = get_fs(n)
        b, a = butter(1, [0.75, 2.5], btype="bandpass", fs=fs)
        out[(n, s)] = (hr_fft(g, fs), hr_fft(filtfilt(b, a, p), fs))
    return out


SPECS = [
    ("Dual-TL (+stretch)", lambda: DualTL(D=128, L=2), "checkpoints/split_filtered_D128L2_stretch25_best.pt"),
    ("Dual-TL (no aug)", lambda: DualTL(D=128, L=2), "checkpoints/FINAL_dualtl.pt"),
    ("1D-CNN (+stretch)", CNN1D, "checkpoints/baseline_1D-CNN_stretch25.pt"),
    ("1D-CNN (no aug)", CNN1D, "checkpoints/baseline_1D-CNN_nostretch.pt"),
    ("GRU (+stretch)", GRUNet, "checkpoints/baseline_GRU_stretch25.pt"),
    ("GRU (no aug)", GRUNet, "checkpoints/baseline_GRU_nostretch.pt"),
]
results = {}
for name, ctor, path in SPECS:
    if not Path(path).exists():
        print("missing:", path, "-> skipped")
        continue
    m = ctor().to(dev)
    m.load_state_dict(torch.load(path, map_location=dev))
    results[name] = predict(m)
    print("scored", name)

for name in ["CHROM", "POS", "GREEN"]:
    p = Path(f"outputs/compare/{name}_windows.csv")
    if p.exists():
        results[name] = {(r["subject"], int(r["start_frame"])): (float(r["hr_true_bpm"]), float(r["hr_pred_bpm"]))
                         for r in csv.DictReader(open(p))}

ref = "Dual-TL (+stretch)"
if ref not in results:
    raise SystemExit("The Dual-TL (+stretch) checkpoint is missing, so nothing can be compared.")

num = lambda s: int("".join(c for c in s if c.isdigit()))
keys = sorted(set.intersection(*[set(d) for d in results.values()]), key=lambda k: (num(k[0]), k[1]))
print("windows common to all methods:", len(keys), "(should be 837)")
hg = np.array([results[ref][k][0] for k in keys])
gap = max(np.abs(np.array([results[m][k][0] for k in keys]) - hg).max() for m in results)
print(f"largest ground-truth difference between methods: {gap:.3f} bpm (should be about 0)")

subj = np.array([k[0] for k in keys])
people = sorted(set(subj), key=num)
names = list(results)
pred = {m: np.array([results[m][k][1] for k in keys]) for m in names}
abs_err = {m: np.abs(pred[m] - hg) for m in names}

sums = np.array([[abs_err[m][subj == p].sum() for p in people] for m in names])
counts = np.array([(subj == p).sum() for p in people])
rng = np.random.default_rng(0)
B = 5000
boot = np.empty((B, len(names)))
for b in range(B):
    idx = rng.integers(0, len(people), len(people))
    boot[b] = sums[:, idx].sum(1) / counts[idx].sum()
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
ri = names.index(ref)

rows = []
for i, m in enumerate(names):
    e = pred[m] - hg
    rows.append((m, abs_err[m].mean(), lo[i], hi[i], np.sqrt((e ** 2).mean()),
                 np.corrcoef(pred[m], hg)[0, 1], float(np.mean(boot[:, ri] < boot[:, i]))))
rows.sort(key=lambda r: r[1])

print("\nTEST people (8), same windows for every method")
print(f"{'method':20s} | MAE (bpm) | 95% interval for MAE | RMSE | r    | chance Dual-TL(+stretch) is better")
with open("outputs/final_models_test.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "mae", "mae_ci_low", "mae_ci_high", "rmse", "pearson_r", "p_dualtl_stretch_better"])
    for m, mae, l, h, rmse, r, pb in rows:
        extra = "-" if m == ref else f"{pb:.2f}"
        print(f"{m:20s} | {mae:9.2f} | {l:6.2f} to {h:6.2f}      | {rmse:4.2f} | {r:.2f} | {extra}")
        w.writerow([m, round(mae, 3), round(l, 3), round(h, 3), round(rmse, 3), round(r, 3), round(pb, 3)])

with open("outputs/final_per_subject.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["subject"] + names)
    for p in people:
        w.writerow([p] + [round(abs_err[m][subj == p].mean(), 2) for m in names])
print("\nSaved outputs/final_models_test.csv and outputs/final_per_subject.csv")