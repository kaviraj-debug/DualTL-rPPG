import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import argparse, csv, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from torch.utils.data import DataLoader
from dataset import make_datasets
from model import neg_pearson

ap = argparse.ArgumentParser()
ap.add_argument("--stretch", type=float, default=0.0, help="training speed augmentation, e.g. 0.25")
ap.add_argument("--name", default="", help="suffix for saved file names, e.g. stretch25")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--test", action="store_true", help="also score the TEST people (use once, at the end)")
args = ap.parse_args()
suffix = f"_{args.name}" if args.name else ""

dev = "cuda"
BS, LR, WD = 32, 1e-3, 1e-2
Path("outputs/compare").mkdir(parents=True, exist_ok=True)


class CNN1D(nn.Module):
    """1D convolutions over time; 63 regions x 3 channels = 189 input channels."""
    def __init__(self, c_in=189, dropout=0.2):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 7, padding=3), nn.BatchNorm1d(o),
                                 nn.ReLU(), nn.Dropout(dropout))
        self.net = nn.Sequential(block(c_in, 64), block(64, 64), block(64, 32), nn.Conv1d(32, 1, 1))

    def forward(self, m):
        return self.net(m.flatten(1, 2)).squeeze(1)


class GRUNet(nn.Module):
    """Two-layer bidirectional GRU over the 300 frames."""
    def __init__(self, c_in=189, h=64, dropout=0.2):
        super().__init__()
        self.inp = nn.Linear(c_in, h)
        self.gru = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.out = nn.Linear(2 * h, 1)

    def forward(self, m):
        x = m.flatten(1, 2).transpose(1, 2)
        y, _ = self.gru(torch.relu(self.inp(x)))
        return self.out(y).squeeze(-1)


def run_epoch(model, loader, opt=None):
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    for x, y, _, _ in loader:
        x, y = x.to(dev), y.to(dev)
        with torch.set_grad_enabled(train):
            loss = neg_pearson(model(x), y)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(x); n += len(x)
    return tot / n


def hr_fft(x, fs, lo=0.75, hi=2.5, nfft=4096):
    x = (x - x.mean()) * np.hanning(len(x))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    p = np.abs(np.fft.rfft(x, nfft)) ** 2
    m = (f >= lo) & (f <= hi)
    return 60 * f[m][np.argmax(p[m])]


_fps = {}
def get_fs(n):
    if n not in _fps:
        _fps[n] = float(np.load(f"cache/{n}.npz")["fps"])
    return _fps[n]


def hr_eval(model, ds):
    model.eval()
    names, starts, P, G = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(ds), 64):
            batch = [ds[j] for j in range(i, min(i + 64, len(ds)))]
            x = torch.stack([b[0] for b in batch]).to(dev)
            P.append(model(x).float().cpu().numpy())
            G.append(torch.stack([b[1] for b in batch]).numpy())
            names += [b[2] for b in batch]; starts += [b[3] for b in batch]
    P, G = np.concatenate(P), np.concatenate(G)
    rows = []
    for p, g, n, s in zip(P, G, names, starts):
        fs = get_fs(n)
        b, a = butter(1, [0.75, 2.5], btype="bandpass", fs=fs)
        rows.append((n, s, hr_fft(g, fs), hr_fft(filtfilt(b, a, p), fs)))
    return rows


def report(label, rows):
    hg = np.array([r[2] for r in rows]); hp = np.array([r[3] for r in rows])
    names = np.array([r[0] for r in rows])
    e = hp - hg
    print(f">>> {label}: MAE {np.abs(e).mean():.2f} | RMSE {np.sqrt((e ** 2).mean()):.2f} | "
          f"r {np.corrcoef(hp, hg)[0, 1]:.2f} | windows {len(rows)}")
    num = lambda s: int("".join(c for c in s if c.isdigit()))
    print("    per person MAE: " + ", ".join(
        f"{n} {np.abs(e[names == n]).mean():.1f}" for n in sorted(set(names), key=num)))


random.seed(0)
train_ds, val_ds, test_ds = make_datasets(mode="filtered", step_train=5, stretch=args.stretch)
train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

for mname, ctor in [("1D-CNN", CNN1D), ("GRU", GRUNet)]:
    torch.manual_seed(0)
    model = ctor().to(dev)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n=== {mname}{suffix} | parameters {n_par:.2f} M | stretch {args.stretch} ===")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best, best_ep, best_state = 1e9, 0, None
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, train_dl, opt)
        va = run_epoch(model, val_dl)
        if va < best:
            best, best_ep = va, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:2d}/{args.epochs} | train {tr:.4f} | val {va:.4f}" + ("  <- best" if best_ep == ep else ""))
    model.load_state_dict(best_state)
    torch.save(best_state, f"checkpoints/baseline_{mname}{suffix}.pt")
    print(f"best validation loss {best:.4f} at epoch {best_ep}")
    report(f"{mname}{suffix} VALIDATION", hr_eval(model, val_ds))
    if args.test:
        rows = hr_eval(model, test_ds)
        with open(f"outputs/compare/{mname}{suffix}_windows.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subject", "start_frame", "hr_true_bpm", "hr_pred_bpm"])
            for n, s, t, p in rows:
                w.writerow([n, s, round(float(t), 2), round(float(p), 2)])
        report(f"{mname}{suffix} TEST", rows)