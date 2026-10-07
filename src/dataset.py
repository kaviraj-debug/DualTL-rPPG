import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset

WIN = 300      # frames per window (about 10 s at 30 fps)
STEP = 15      # 0.5 s step, as in the paper


def minmax_255(x):
    """Paper-style: min-max each (N, C) row over time to [0, 255], then scale to [0, 1]."""
    mn = x.min(axis=-1, keepdims=True)
    mx = x.max(axis=-1, keepdims=True)
    return (x - mn) / (mx - mn + 1e-6)


def zscore(y):
    return (y - y.mean()) / (y.std() + 1e-6)


def zscore_rows(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)


def bandpass(x, fs, lo=0.6, hi=4.0, axis=0):
    b, a = butter(2, [lo, hi], btype="bandpass", fs=fs)
    return filtfilt(b, a, x, axis=axis)


def list_cached(cache_dir="cache"):
    return sorted(Path(cache_dir).glob("*.npz"))


class UBFCWindows(Dataset):
    """mode='paper'    : min-max input, raw PPG label (as in the paper)
       mode='filtered' : band-passed + z-scored input, band-passed PPG label
       stretch > 0     : training-time speed augmentation, factor in [1-stretch, 1+stretch]"""
    def __init__(self, subject_files, win=WIN, step=STEP, mode="filtered", stretch=0.0):
        self.win, self.mode, self.stretch = win, mode, stretch
        self.data, self.items = {}, []
        for f in subject_files:
            d = np.load(f)
            mst, ppg, fs = d["mst"], d["ppg"], float(d["fps"])
            if mode == "filtered":
                T = len(mst)
                rows = mst.reshape(T, -1)
                rows = bandpass(rows - rows.mean(0), fs)
                mst = rows.reshape(T, 63, 3)
                ppg = bandpass(ppg - ppg.mean(), fs)
            self.data[f.stem] = {"mst": mst.astype(np.float32), "ppg": ppg.astype(np.float32)}
            for s in range(0, len(mst) - win + 1, step):
                self.items.append((f.stem, s))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, s = self.items[i]
        d = self.data[name]
        if self.stretch > 0:
            f = random.uniform(1 - self.stretch, 1 + self.stretch)
            L = int(round(self.win * f))
            s = min(s, len(d["mst"]) - L)
            seg = torch.from_numpy(d["mst"][s:s + L].reshape(L, -1).T.copy())[None]   # (1, 189, L)
            lab = torch.from_numpy(d["ppg"][s:s + L].copy())[None, None]              # (1, 1, L)
            seg = F.interpolate(seg, size=self.win, mode="linear", align_corners=True)[0].numpy()
            y = F.interpolate(lab, size=self.win, mode="linear", align_corners=True)[0, 0].numpy()
            x = seg.reshape(63, 3, self.win)
        else:
            x = d["mst"][s:s + self.win].transpose(1, 2, 0)                           # (63, 3, T)
            y = d["ppg"][s:s + self.win]
        x = minmax_255(x) if self.mode == "paper" else zscore_rows(x)
        y = zscore(y)
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32)), name, s


def _files(names, cache_dir):
    return [Path(cache_dir) / f"{n}.npz" for n in names]


def make_datasets(split_path="split.json", cache_dir="cache", step_train=5, mode="filtered", stretch=0.0):
    """Fixed subject-wise split: returns (train, val, test) datasets. Augmentation only on train."""
    split = json.load(open(split_path))
    train = UBFCWindows(_files(split["train"], cache_dir), step=step_train, mode=mode, stretch=stretch)
    val = UBFCWindows(_files(split["val"], cache_dir), step=STEP, mode=mode)
    test = UBFCWindows(_files(split["test"], cache_dir), step=STEP, mode=mode)
    return train, val, test


def loso_split(test_name, cache_dir="cache", step_train=STEP, mode="filtered"):
    """Old leave-one-subject-out split (kept so older scripts still work)."""
    files = list_cached(cache_dir)
    train = [f for f in files if f.stem != test_name]
    test = [f for f in files if f.stem == test_name]
    return (UBFCWindows(train, step=step_train, mode=mode),
            UBFCWindows(test, step=STEP, mode=mode))