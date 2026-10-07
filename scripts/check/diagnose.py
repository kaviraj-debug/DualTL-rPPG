import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt

def bp(x, fs, lo=0.75, hi=2.5):
    b, a = butter(1, [lo, hi], btype="bandpass", fs=fs)
    return filtfilt(b, a, x, axis=0)

def hr_fft(x, fs, lo=0.75, hi=2.5):
    x = x - x.mean()
    f = np.fft.rfftfreq(len(x), 1 / fs)
    p = np.abs(np.fft.rfft(x)) ** 2
    m = (f >= lo) & (f <= hi)
    return 60 * f[m][np.argmax(p[m])]

print("subject | best |r| (lag 0) | combo,chan | best lag (frames) | HR gt-ppg | HR video | HR file")
for npz in sorted(Path("cache").glob("*.npz")):
    d = np.load(npz)
    mst, ppg, fs, hr_file = d["mst"], d["ppg"], float(d["fps"]), float(d["hr"])
    gt = bp(ppg, fs)
    rows = mst.reshape(len(mst), -1)                  # (T, 189), index = combo*3 + channel
    rows_bp = bp(rows - rows.mean(0), fs)
    r = np.array([np.corrcoef(rows_bp[:, k], gt)[0, 1] for k in range(rows_bp.shape[1])])
    best = int(np.argmax(np.abs(r)))
    x = rows_bp[:, best]

    def corr_at(l):
        a, b = (x[l:], gt[:len(gt) - l]) if l >= 0 else (x[:l], gt[-l:])
        return abs(np.corrcoef(a, b)[0, 1])

    lags = list(range(-45, 46))
    best_lag = lags[int(np.argmax([corr_at(l) for l in lags]))]
    print(f"{npz.stem:10s} | {abs(r[best]):.2f} | {best // 3:2d},{'YUV'[best % 3]} | {best_lag:+3d} | "
          f"{hr_fft(gt, fs):5.1f} | {hr_fft(x, fs):5.1f} | {hr_file:5.1f}")