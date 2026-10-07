import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import csv, json
from pathlib import Path
import numpy as np
import torch
from scipy.signal import butter, filtfilt
from dataset import UBFCWindows, bandpass, WIN, STEP
from model import DualTL

dev = "cuda"
split = json.load(open("split.json"))
MODELS = {
    "Dual-TL stretch 0.25": "checkpoints/split_filtered_D128L2_stretch25_best.pt",
    "Dual-TL stretch 0.4": "checkpoints/split_filtered_D128L2_stretch40_best.pt",
}


def hr_fft(x, fs, lo=0.75, hi=2.5, nfft=4096):
    x = (x - x.mean()) * np.hanning(len(x))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    p = np.abs(np.fft.rfft(x, nfft)) ** 2
    m = (f >= lo) & (f <= hi)
    return 60 * f[m][np.argmax(p[m])]


# corrected baseline: average PPG-based heart rate of the TRAINING people
train_hr = []
for n in split["train"]:
    d = np.load(f"cache/{n}.npz")
    fs = float(d["fps"])
    ppg = bandpass(d["ppg"] - d["ppg"].mean(), fs)
    train_hr += [hr_fft(ppg[s:s + WIN], fs) for s in range(0, len(ppg) - WIN + 1, STEP)]
base = float(np.mean(train_hr))

test_ds = UBFCWindows([Path("cache") / f"{n}.npz" for n in split["test"]], mode="filtered")
names = [test_ds.items[i][0] for i in range(len(test_ds))]
X = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
G = np.stack([test_ds[i][1].numpy() for i in range(len(test_ds))])
fps = {n: float(np.load(f"cache/{n}.npz")["fps"]) for n in set(names)}
hg = np.array([hr_fft(g, fps[n]) for g, n in zip(G, names)])
print(f"test windows: {len(X)} | corrected baseline: always guess {base:.1f} bpm")

rows = []
for name, path in MODELS.items():
    m = DualTL(D=128, L=2).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev))
    m.eval()
    with torch.no_grad():
        P = np.concatenate([m(X[i:i + 64].to(dev)).float().cpu().numpy() for i in range(0, len(X), 64)])
    hp = []
    for p, n in zip(P, names):
        b, a = butter(1, [0.75, 2.5], btype="bandpass", fs=fps[n])
        hp.append(hr_fft(filtfilt(b, a, p), fps[n]))
    e = np.array(hp) - hg
    rows.append((name, np.abs(e).mean(), np.sqrt((e ** 2).mean()), np.corrcoef(hp, hg)[0, 1]))
eb = base - hg
rows.append(("Guess the average (corrected)", np.abs(eb).mean(), np.sqrt((eb ** 2).mean()), float("nan")))

print(f"\n{'model':30s} | MAE (bpm) | RMSE (bpm) | Pearson r")
with open("outputs/extra_test_rows.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "mae", "rmse", "pearson_r"])
    for name, mae, rmse, r in rows:
        print(f"{name:30s} | {mae:9.2f} | {rmse:10.2f} | {r:9.2f}")
        w.writerow([name, round(mae, 3), round(rmse, 3), round(r, 3)])
print("\nSaved outputs/extra_test_rows.csv")