import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import csv, json, warnings
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt
from dataset import WIN, STEP, bandpass

warnings.filterwarnings("ignore")
Path("outputs/compare").mkdir(parents=True, exist_ok=True)
test_names = json.load(open("split.json"))["test"]


def hr_fft(x, fs, lo=0.75, hi=2.5, nfft=4096):
    x = (x - x.mean()) * np.hanning(len(x))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    p = np.abs(np.fft.rfft(x, nfft)) ** 2
    m = (f >= lo) & (f <= hi)
    fm, pm = f[m], p[m]
    k = int(np.argmax(pm))
    near = np.abs(fm - fm[k]) <= 0.1
    return 60 * float(fm[k]), float(pm[near].sum() / (pm.sum() + 1e-12))


def bp(x, fs):
    b, a = butter(2, [0.75, 2.5], btype="bandpass", fs=fs)
    return filtfilt(b, a, x)


def yuv_to_rgb(yuv):
    """Invert OpenCV's BGR->YUV (BT.601). Returns (T, 3) as R, G, B."""
    Y, U, V = yuv[:, 0], yuv[:, 1] - 128.0, yuv[:, 2] - 128.0
    B = Y + U / 0.492
    R = Y + V / 0.877
    G = (Y - 0.299 * R - 0.114 * B) / 0.587
    return np.stack([R, G, B], axis=1)


def green(rgb, fs):
    return bp(rgb[:, 1] - rgb[:, 1].mean(), fs)


def chrom(rgb, fs):
    n = rgb / rgb.mean(0)
    xs = 3 * n[:, 0] - 2 * n[:, 1]
    ys = 1.5 * n[:, 0] + n[:, 1] - 1.5 * n[:, 2]
    xf, yf = bp(xs, fs), bp(ys, fs)
    return xf - (xf.std() / (yf.std() + 1e-8)) * yf


def pos(rgb, fs):
    T, l = len(rgb), int(1.6 * fs)
    P = np.array([[0, 1, -1], [-2, 1, 1]], float)
    H = np.zeros(T)
    for n in range(l - 1, T):
        m = n - l + 1
        c = rgb[m:n + 1] / rgb[m:n + 1].mean(0)
        s = P @ c.T
        h = s[0] + (s[0].std() / (s[1].std() + 1e-8)) * s[1]
        H[m:n + 1] += h - h.mean()
    return bp(H, fs)


try:
    from sklearn.decomposition import FastICA

    def ica(rgb, fs):
        X = (rgb - rgb.mean(0)) / (rgb.std(0) + 1e-8)
        S = FastICA(n_components=3, random_state=0, max_iter=500).fit_transform(X)
        best, best_q = None, -1
        for k in range(3):
            s = bp(S[:, k], fs)
            q = hr_fft(s, fs)[1]
            if q > best_q:
                best, best_q = s, q
        return best

    METHODS = {"GREEN": green, "ICA": ica, "CHROM": chrom, "POS": pos}
except Exception as e:
    print("scikit-learn not available, ICA skipped:", e)
    METHODS = {"GREEN": green, "CHROM": chrom, "POS": pos}

rows = {k: [] for k in METHODS}
for name in test_names:
    d = np.load(f"cache/{name}.npz")
    fs = float(d["fps"])
    rgb_all = yuv_to_rgb(d["mst"][:, 62, :])                 # combination 63 = all 6 regions
    ppg = bandpass(d["ppg"] - d["ppg"].mean(), fs)           # same ground truth as for Dual-TL
    for s in range(0, len(rgb_all) - WIN + 1, STEP):
        win = rgb_all[s:s + WIN]
        hg = hr_fft(ppg[s:s + WIN], fs)[0]
        for mname, fn in METHODS.items():
            rows[mname].append((name, s, hg, hr_fft(fn(win, fs), fs)[0]))
    print("done", name)


def metrics(hg, hp):
    e = hp - hg
    return np.abs(e).mean(), np.sqrt((e ** 2).mean()), np.corrcoef(hp, hg)[0, 1]


results = {}
for mname, r in rows.items():
    with open(f"outputs/compare/{mname}_windows.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "start_frame", "hr_true_bpm", "hr_pred_bpm"])
        for n, s, t, p in r:
            w.writerow([n, s, round(t, 2), round(p, 2)])
    arr = np.array([(t, p) for _, _, t, p in r])
    results[mname] = metrics(arr[:, 0], arr[:, 1])

if Path("outputs/test_windows.csv").exists():
    d = list(csv.DictReader(open("outputs/test_windows.csv")))
    hg = np.array([float(x["hr_true_bpm"]) for x in d])
    hp = np.array([float(x["hr_pred_bpm"]) for x in d])
    results["Dual-TL (ours)"] = metrics(hg, hp)

print(f"\nTEST people: {len(rows[next(iter(rows))])} windows each")
print(f"{'method':16s} | MAE (bpm) | RMSE (bpm) | Pearson r")
for k, (mae, rmse, r) in sorted(results.items(), key=lambda kv: kv[1][0]):
    print(f"{k:16s} | {mae:9.2f} | {rmse:10.2f} | {r:9.2f}")