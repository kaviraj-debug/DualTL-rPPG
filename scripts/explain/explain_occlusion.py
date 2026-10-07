import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import numpy as np
import torch
from scipy.signal import butter, filtfilt
from dataset import make_datasets
from model import DualTL

dev = "cuda"
CKPT = "checkpoints/split_filtered_D128L2_stretch25_best.pt"
REGIONS = ["forehead", "upper cheek L", "lower cheek L", "upper cheek R", "lower cheek R", "chin"]
CHANNELS = ["Y", "U", "V"]
COMBO = np.array([[(k >> i) & 1 for i in range(6)] for k in range(1, 64)], np.float32)   # (63, 6)

model = DualTL(D=128, L=2).to(dev)
model.load_state_dict(torch.load(CKPT, map_location=dev))
model.eval()

_, val_ds, _ = make_datasets(mode="filtered", step_train=15)
names = [val_ds.items[i][0] for i in range(len(val_ds))]
X = torch.stack([val_ds[i][0] for i in range(len(val_ds))])          # (N, 63, 3, 300)
G = np.stack([val_ds[i][1].numpy() for i in range(len(val_ds))])
print("occlusion test on", len(X), "VALIDATION windows with", CKPT)

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


hg = np.array([hr_fft(g, get_fs(n)) for g, n in zip(G, names)])


def score(Xm):
    P = []
    with torch.no_grad():
        for i in range(0, len(Xm), 64):
            P.append(model(Xm[i:i + 64].to(dev)).float().cpu().numpy())
    P = np.concatenate(P)
    hp = []
    for p, n in zip(P, names):
        fs = get_fs(n)
        b, a = butter(1, [0.75, 2.5], btype="bandpass", fs=fs)
        hp.append(hr_fft(filtfilt(b, a, p), fs))
    e = np.array(hp) - hg
    return np.abs(e).mean(), np.sqrt((e ** 2).mean())


base_mae, base_rmse = score(X)
print(f"\n{'condition':32s} | MAE (bpm) | RMSE (bpm) | MAE change")
print(f"{'nothing removed':32s} | {base_mae:9.2f} | {base_rmse:10.2f} | {'-':>10s}")

print("\nremove one face region (the 32 combinations that include it)")
for r, name in enumerate(REGIONS):
    Xm = X.clone()
    Xm[:, torch.from_numpy(COMBO[:, r] == 1)] = 0
    mae, rmse = score(Xm)
    print(f"{name:32s} | {mae:9.2f} | {rmse:10.2f} | {mae - base_mae:+10.2f}")

rng = np.random.default_rng(0)
maes, rmses = [], []
for _ in range(5):
    Xm = X.clone()
    Xm[:, torch.from_numpy(rng.permutation(63) < 32)] = 0
    m1, m2 = score(Xm)
    maes.append(m1); rmses.append(m2)
print(f"{'CONTROL: 32 random combinations':32s} | {np.mean(maes):9.2f} | {np.mean(rmses):10.2f} | "
      f"{np.mean(maes) - base_mae:+10.2f}   (range {min(maes):.2f} to {max(maes):.2f})")

print("\nremove one colour channel")
for c, name in enumerate(CHANNELS):
    Xm = X.clone()
    Xm[:, :, c, :] = 0
    mae, rmse = score(Xm)
    print(f"{'without ' + name:32s} | {mae:9.2f} | {rmse:10.2f} | {mae - base_mae:+10.2f}")