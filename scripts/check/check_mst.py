import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt

def bandpass(x, fs, lo=0.75, hi=2.5):
    b, a = butter(1, [lo, hi], btype="bandpass", fs=fs)
    return filtfilt(b, a, x)

def fft_hr(x, fs, lo=0.75, hi=2.5):
    x = x - x.mean()
    f = np.fft.rfftfreq(len(x), 1 / fs)
    p = np.abs(np.fft.rfft(x)) ** 2
    m = (f >= lo) & (f <= hi)
    return 60 * f[m][np.argmax(p[m])]

for npz in sorted(Path("cache").glob("*.npz")):
    d = np.load(npz)
    mst, ppg, fs, hr_gt = d["mst"], d["ppg"], float(d["fs"] if "fs" in d else d["fps"]), float(d["hr"])
    all_roi = mst[:, 62, :]                       # combination 63 = all 6 ROIs
    est = [fft_hr(bandpass(all_roi[:, c], fs), fs) for c in range(3)]
    print(f"{npz.stem:12s} GT HR={hr_gt:5.1f} | from Y={est[0]:5.1f} U={est[1]:5.1f} V={est[2]:5.1f}")

# visual check on the first cached subject
d = np.load(sorted(Path("cache").glob("*.npz"))[0])
mst, ppg, fs = d["mst"], d["ppg"], float(d["fps"])
n = int(10 * fs)
fig, ax = plt.subplots(3, 1, figsize=(10, 7))
ax[0].plot(np.arange(n) / fs, bandpass(mst[:, 62, 0], fs)[:n]); ax[0].set_title("Y channel, all ROIs, band-passed")
ax[1].plot(np.arange(n) / fs, bandpass(mst[:, 62, 1], fs)[:n]); ax[1].set_title("U channel, all ROIs, band-passed")
ax[2].plot(np.arange(n) / fs, ppg[:n]); ax[2].set_title("Ground-truth PPG")
plt.tight_layout(); plt.savefig("outputs/plots/mst_check.png")
print("Saved outputs/plots/mst_check.png")