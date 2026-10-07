import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import numpy as np
from scipy.signal import butter, filtfilt
from dataset import bandpass, WIN, STEP

CHECK = ["subject20", "subject11", "subject24", "subject18", "subject3"]


def hr_fft(x, fs, lo=0.75, hi=2.5, nfft=4096):
    x = (x - x.mean()) * np.hanning(len(x))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    p = np.abs(np.fft.rfft(x, nfft)) ** 2
    m = (f >= lo) & (f <= hi)
    return 60 * f[m][np.argmax(p[m])]


def bp(x, fs):
    b, a = butter(2, [0.75, 2.5], btype="bandpass", fs=fs)
    return filtfilt(b, a, x)


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


def power_near(x, fs, center_bpm, width=12, nfft=2 ** 15):
    """Power of the whole-video spectrum near center_bpm, relative to its overall peak."""
    x = (x - x.mean()) * np.hanning(len(x))
    f = np.fft.rfftfreq(nfft, 1 / fs) * 60
    p = np.abs(np.fft.rfft(x, nfft)) ** 2
    band = (f >= 45) & (f <= 150)
    near = (f >= center_bpm - width) & (f <= center_bpm + width)
    return p[near].max() / p[band].max()


print("subject   | record | PPG peak | CHROM | POS   | PPG power near record HR (1.00 = as strong as its peak)")
for n in CHECK:
    d = np.load(f"cache/{n}.npz")
    fs = float(d["fps"])
    rec = float(d["hr"])
    rgb = yuv_to_rgb(d["mst"][:, 62, :])
    ppg = bandpass(d["ppg"] - d["ppg"].mean(), fs)
    h_ppg, h_ch, h_po = [], [], []
    for s in range(0, len(rgb) - WIN + 1, STEP):
        w = rgb[s:s + WIN]
        h_ppg.append(hr_fft(ppg[s:s + WIN], fs))
        h_ch.append(hr_fft(chrom(w, fs), fs))
        h_po.append(hr_fft(pos(w, fs), fs))
    pn = power_near(ppg, fs, rec)
    print(f"{n:9s} | {rec:6.1f} | {np.median(h_ppg):8.1f} | {np.median(h_ch):5.1f} | "
          f"{np.median(h_po):5.1f} | {pn:.2f}")