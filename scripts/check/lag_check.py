import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import numpy as np
import torch
from dataset import loso_split, list_cached
from model import DualTL

dev = "cuda"
MAXLAG = 15      # half a second


def window_corr(p, g):
    p = p - p.mean(); g = g - g.mean()
    return float((p * g).sum() / (np.linalg.norm(p) * np.linalg.norm(g) + 1e-8))


for f in list_cached():
    name = f.stem
    ckpt = f"checkpoints/{name}_filtered_D128L2.pt"
    try:
        sd = torch.load(ckpt, map_location=dev)
    except FileNotFoundError:
        print(name, "- no checkpoint"); continue

    model = DualTL(D=128, L=2).to(dev)
    model.load_state_dict(sd)
    model.eval()
    _, test_ds = loso_split(name, mode="filtered")

    P, G = [], []
    with torch.no_grad():
        for i in range(0, len(test_ds), 64):
            batch = [test_ds[j] for j in range(i, min(i + 64, len(test_ds)))]
            x = torch.stack([b[0] for b in batch]).to(dev)
            P.append(model(x).float().cpu().numpy())
            G.append(torch.stack([b[1] for b in batch]).numpy())
    P, G = np.concatenate(P), np.concatenate(G)

    res = {}
    for lag in range(-MAXLAG, MAXLAG + 1):
        rs = []
        for p, g in zip(P, G):
            if lag >= 0:
                rs.append(window_corr(p[lag:], g[:len(g) - lag]))
            else:
                rs.append(window_corr(p[:lag], g[-lag:]))
        res[lag] = np.mean(rs)
    best = max(res, key=res.get)
    print(f"{name:10s} r at lag 0 = {res[0]:+.2f} | best lag {best:+3d} frames -> r = {res[best]:+.2f}")