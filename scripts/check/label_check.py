import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import json
import numpy as np
from dataset import bandpass, WIN, STEP

split = json.load(open("split.json"))
group = {n: g for g in ["train", "val", "test"] for n in split[g]}


def hr_fft(x, fs, lo=0.75, hi=2.5, nfft=4096):
    x = (x - x.mean()) * np.hanning(len(x))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    p = np.abs(np.fft.rfft(x, nfft)) ** 2
    m = (f >= lo) & (f <= hi)
    return 60 * f[m][np.argmax(p[m])]


rows = []
for n, g in group.items():
    d = np.load(f"cache/{n}.npz")
    fs = float(d["fps"])
    ppg = bandpass(d["ppg"] - d["ppg"].mean(), fs)
    hrs = [hr_fft(ppg[s:s + WIN], fs) for s in range(0, len(ppg) - WIN + 1, STEP)]
    rows.append((n, g, float(d["hr"]), float(np.mean(hrs)), float(np.std(hrs))))

rows.sort(key=lambda r: -abs(r[2] - r[3]))
print("subject    | group | HR from dataset record | HR from PPG signal | difference | spread across windows")
for n, g, h_file, h_ppg, sd in rows:
    flag = "  <-- differs by more than 10 bpm" if abs(h_file - h_ppg) > 10 else ""
    print(f"{n:10s} | {g:5s} | {h_file:22.1f} | {h_ppg:18.1f} | {h_ppg - h_file:+10.1f} | {sd:21.1f}{flag}")

big = [r for r in rows if abs(r[2] - r[3]) > 10]
print(f"\npeople with a difference above 10 bpm: {len(big)} of {len(rows)}")