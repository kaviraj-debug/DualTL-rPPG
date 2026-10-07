import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import argparse
import numpy as np
import torch
from scipy.signal import butter, filtfilt
from dataset import make_datasets
from model import DualTL

ap = argparse.ArgumentParser()
ap.add_argument("--test", action="store_true", help="score the TEST people (use only once, at the very end)")
args = ap.parse_args()

dev = "cuda"
CKPTS = {
    "seed 0": "checkpoints/split_filtered_D128L2_stretch25_best.pt",
    "seed 1": "checkpoints/split_filtered_D128L2_s25seed1_best.pt",
    "seed 2": "checkpoints/split_filtered_D128L2_s25seed2_best.pt",
}

_, val_ds, test_ds = make_datasets(mode="filtered", step_train=15)
ds = test_ds if args.test else val_ds
print("Scoring on", "TEST" if args.test else "VALIDATION", "people |", len(ds), "windows")

names = [ds.items[i][0] for i in range(len(ds))]
G = np.stack([ds[i][1].numpy() for i in range(len(ds))])
X = torch.stack([ds[i][0] for i in range(len(ds))])

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


def predict(path):
    model = DualTL(D=128, L=2).to(dev)
    model.load_state_dict(torch.load(path, map_location=dev))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            out.append(model(X[i:i + 64].to(dev)).float().cpu().numpy())
    return np.concatenate(out)


def to_hr(P):
    hp = []
    for p, n in zip(P, names):
        fs = get_fs(n)
        b, a = butter(1, [0.75, 2.5], btype="bandpass", fs=fs)
        hp.append(hr_fft(filtfilt(b, a, p), fs))
    return np.array(hp)


hg = np.array([hr_fft(g, get_fs(n)) for g, n in zip(G, names)])
names_arr = np.array(names)
num = lambda s: int("".join(c for c in s if c.isdigit()))
people = sorted(set(names), key=num)

zs = lambda P: (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-8)
preds = {}
for k, path in CKPTS.items():
    preds[k] = zs(predict(path))
preds["average of 3"] = zs(np.mean([preds[k] for k in CKPTS], axis=0))

print(f"\n{'model':14s} | MAE (bpm) | RMSE (bpm) | Pearson r")
hrs = {}
for k, P in preds.items():
    hp = to_hr(P)
    hrs[k] = hp
    e = hp - hg
    print(f"{k:14s} | {np.abs(e).mean():9.2f} | {np.sqrt((e ** 2).mean()):10.2f} | {np.corrcoef(hp, hg)[0, 1]:9.2f}")

print("\nper-person MAE (bpm)")
print("subject    | " + " | ".join(f"{k:>12s}" for k in preds))
for p in people:
    m = names_arr == p
    print(f"{p:10s} | " + " | ".join(f"{np.abs(hrs[k][m] - hg[m]).mean():12.2f}" for k in preds))