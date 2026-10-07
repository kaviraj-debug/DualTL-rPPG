import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import json
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from dataset import UBFCWindows, zscore_rows
from model import DualTL

dev = "cuda"
FACTORS = [1.0, 0.8, 0.65]      # 1.0 = original pulse speed; 0.65 = pulse slowed to 65%
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


def bp(x, fs, order):
    b, a = butter(order, [0.75, 2.5], btype="bandpass", fs=fs)
    return filtfilt(b, a, x, axis=-1)


def yuv_to_rgb(yuv):
    Y, U, V = yuv[:, 0], yuv[:, 1] - 128.0, yuv[:, 2] - 128.0
    B = Y + U / 0.492
    R = Y + V / 0.877
    G = (Y - 0.299 * R - 0.114 * B) / 0.587
    return np.stack([R, G, B], axis=1)


def chrom(rgb, fs):
    n = rgb / rgb.mean(0)
    xs = 3 * n[:, 0] - 2 * n[:, 1]
    ys = 1.5 * n[:, 0] + n[:, 1] - 1.5 * n[:, 2]
    xf, yf = bp(xs, fs, 2), bp(ys, fs, 2)
    return xf - (xf.std() / (yf.std() + 1e-8)) * yf


split = json.load(open("split.json"))
files = [Path("cache") / f"{n}.npz" for n in split["val"]]
ds = UBFCWindows(files, mode="filtered")          # filtered input and label, as in training
raw = {}
for f in files:
    d = np.load(f)
    raw[f.stem] = (yuv_to_rgb(d["mst"][:, 62, :]), float(d["fps"]))   # unfiltered, for CHROM

nets = {}
for name, path in MODELS.items():
    m = DualTL(D=128, L=2).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev))
    m.eval()
    nets[name] = m

print(f"{'pulse speed':12s} | windows | mean true HR | {'CHROM':>7s} | " + " | ".join(f"{k:>20s}" for k in MODELS))
for fac in FACTORS:
    L = int(round(300 * fac))
    old, new = np.arange(L), np.linspace(0, L - 1, 300)
    xs, true, ch, subj = [], [], [], []
    for name in ds.data:
        mst, ppg = ds.data[name]["mst"], ds.data[name]["ppg"]
        rgb, fs = raw[name]
        for s in range(0, len(mst) - L + 1, 30):
            seg = interp1d(old, mst[s:s + L].reshape(L, -1), axis=0)(new)
            t = hr_fft(interp1d(old, ppg[s:s + L])(new), fs)
            if t < 50:                     # too close to the edge of the 45-150 bpm band
                continue
            xs.append(zscore_rows(seg.reshape(300, 63, 3).transpose(1, 2, 0)).astype(np.float32))
            true.append(t)
            ch.append(hr_fft(chrom(interp1d(old, rgb[s:s + L], axis=0)(new), fs), fs))
            subj.append(fs)
    X = torch.from_numpy(np.stack(xs))
    true, ch = np.array(true), np.array(ch)
    res = {}
    for k, net in nets.items():
        with torch.no_grad():
            P = np.concatenate([net(X[i:i + 64].to(dev)).float().cpu().numpy() for i in range(0, len(X), 64)])
        res[k] = np.array([hr_fft(bp(p, fs, 1), fs) for p, fs in zip(P, subj)])
    print(f"{fac:12.2f} | {len(true):7d} | {true.mean():12.1f} | {np.abs(ch - true).mean():7.2f} | "
          + " | ".join(f"{np.abs(res[k] - true).mean():20.2f}" for k in MODELS))