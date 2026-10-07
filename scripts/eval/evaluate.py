import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import argparse, json
import numpy as np
import torch
from scipy.signal import butter, filtfilt
from dataset import make_datasets
from model import DualTL

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="filtered")
ap.add_argument("--D", type=int, default=128)
ap.add_argument("--L", type=int, default=2)
ap.add_argument("--shift", type=int, default=0)
ap.add_argument("--name", default="", help="suffix used when the model was trained")
ap.add_argument("--test", action="store_true", help="evaluate on the TEST subjects (use only once, at the very end)")
args = ap.parse_args()

dev = "cuda"
tag = f"split_{args.mode}_D{args.D}L{args.L}" + (f"_S{args.shift}" if args.shift else "")
if args.name:
    tag += "_" + args.name
_, val_ds, test_ds = make_datasets(mode=args.mode, step_train=15)
ds = test_ds if args.test else val_ds
print("Evaluating on", "TEST" if args.test else "VALIDATION", "subjects |", len(ds), "windows | checkpoint:", tag)

model = DualTL(D=args.D, L=args.L).to(dev)
model.load_state_dict(torch.load(f"checkpoints/{tag}_best.pt", map_location=dev))
model.eval()

fps = {}
def get_fs(name):
    if name not in fps:
        fps[name] = float(np.load(f"cache/{name}.npz")["fps"])
    return fps[name]

def hr_fft(x, fs, lo=0.75, hi=2.5, nfft=4096):
    x = (x - x.mean()) * np.hanning(len(x))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    p = np.abs(np.fft.rfft(x, nfft)) ** 2
    m = (f >= lo) & (f <= hi)
    return 60 * f[m][np.argmax(p[m])]

names, P, G = [], [], []
with torch.no_grad():
    for i in range(0, len(ds), 64):
        batch = [ds[j] for j in range(i, min(i + 64, len(ds)))]
        x = torch.stack([b[0] for b in batch]).to(dev)
        P.append(model(x).float().cpu().numpy())
        G.append(torch.stack([b[1] for b in batch]).numpy())
        names += [b[2] for b in batch]
P, G = np.concatenate(P), np.concatenate(G)

hp, hg = [], []
for p, g, n in zip(P, G, names):
    fs = get_fs(n)
    bb, aa = butter(1, [0.75, 2.5], btype="bandpass", fs=fs)
    hp.append(hr_fft(filtfilt(bb, aa, p), fs))
    hg.append(hr_fft(g, fs))
hp, hg, names = np.array(hp), np.array(hg), np.array(names)

train_names = json.load(open("split.json"))["train"]
base = np.mean([float(np.load(f"cache/{n}.npz")["hr"]) for n in train_names])

print("\nsubject    | windows | HR true | HR pred |  MAE")
for n in sorted(set(names), key=lambda s: int("".join(c for c in s if c.isdigit()))):
    m = names == n
    print(f"{n:10s} | {m.sum():7d} | {hg[m].mean():7.1f} | {hp[m].mean():7.1f} | {np.abs(hp[m] - hg[m]).mean():5.1f}")

err = hp - hg
print("-" * 52)
print(f"OVERALL: MAE {np.abs(err).mean():.2f} bpm | RMSE {np.sqrt((err ** 2).mean()):.2f} bpm | Pearson r {np.corrcoef(hp, hg)[0, 1]:.2f}")
print(f"BASELINE (always guess {base:.1f} bpm, the training-subject average): MAE {np.abs(base - hg).mean():.2f} bpm")