import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import numpy as np
import torch
from dataset import make_datasets
from model import DualTL

dev = "cuda"
CKPT = "checkpoints/split_filtered_D128L2_stretch25_best.pt"
REGIONS = ["forehead", "upper cheek L", "lower cheek L", "upper cheek R", "lower cheek R", "chin"]
COMBO = np.array([[(k >> i) & 1 for i in range(6)] for k in range(1, 64)], np.float32)   # (63, 6)

model = DualTL(D=128, L=2).to(dev)
model.load_state_dict(torch.load(CKPT, map_location=dev))
model.eval()

store = {}
def keep(name):
    def hook(module, inp, out):
        out.retain_grad()
        store[name] = out
    return hook
model.spatial.embed.register_forward_hook(keep("s"))     # 63 patch embeddings
model.temporal.embed.register_forward_hook(keep("t"))    # 300 patch embeddings

_, val_ds, _ = make_datasets(mode="filtered", step_train=15)
print("explaining", len(val_ds), "VALIDATION windows with", CKPT)

S, T, R = [], [], []
for i in range(0, len(val_ds), 32):
    batch = [val_ds[j] for j in range(i, min(i + 32, len(val_ds)))]
    x = torch.stack([b[0] for b in batch]).to(dev)
    g = torch.stack([b[1] for b in batch]).to(dev)
    model.zero_grad()
    out = model(x)
    o = out - out.mean(1, keepdim=True)
    gg = g - g.mean(1, keepdim=True)
    r = (o * gg).sum(1) / (o.norm(dim=1) * gg.norm(dim=1) + 1e-8)   # correlation with the true pulse
    r.sum().backward()
    S.append(torch.relu(store["s"] * store["s"].grad).sum(-1).detach().cpu().numpy())   # (B, 63)
    T.append(torch.relu(store["t"] * store["t"].grad).sum(-1).detach().cpu().numpy())   # (B, 300)
    R.append(r.detach().cpu().numpy())

S, T, R = np.concatenate(S), np.concatenate(T), np.concatenate(R)
S = S / (S.sum(1, keepdims=True) + 1e-12)
T = T / (T.sum(1, keepdims=True) + 1e-12)
s_mean, t_mean = S.mean(0), T.mean(0)
region = s_mean @ (COMBO / COMBO.sum(1, keepdims=True))   # each combination's importance split among its regions

print(f"\nmean correlation of the output with the true pulse on these windows: {R.mean():.2f}")
print("\nimportance by face region (equal importance would be 16.7% each)")
for n, v in sorted(zip(REGIONS, region), key=lambda z: -z[1]):
    print(f"  {n:14s} {100 * v:5.1f}%")
print("\ntop 8 region combinations (equal importance would be 1.6% each)")
for k in np.argsort(-s_mean)[:8]:
    names = ", ".join(REGIONS[j] for j in range(6) if COMBO[k, j])
    print(f"  combination {k + 1:2d} ({100 * s_mean[k]:4.1f}%): {names}")
th = [t_mean[:100].sum(), t_mean[100:200].sum(), t_mean[200:].sum()]
print("\nimportance over time (equal would be 33.3% each)")
print(f"  first third {100 * th[0]:.1f}% | middle third {100 * th[1]:.1f}% | last third {100 * th[2]:.1f}%")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].bar(REGIONS, 100 * region)
    ax[0].axhline(100 / 6, ls="--", color="k")
    ax[0].set_title("importance by face region (%)")
    ax[0].tick_params(axis="x", labelrotation=45)
    ax[1].bar(range(1, 64), 100 * s_mean)
    ax[1].set_title("importance of the 63 combinations (%)")
    ax[1].set_xlabel("combination")
    ax[2].plot(100 * t_mean)
    ax[2].set_title("importance over the 300 frames (%)")
    ax[2].set_xlabel("frame")
    plt.tight_layout()
    plt.savefig("outputs/plots/fig_gradcam_style.png", dpi=150)
    print("\nSaved outputs/plots/fig_gradcam_style.png")
except Exception as e:
    print("Plot skipped:", e)