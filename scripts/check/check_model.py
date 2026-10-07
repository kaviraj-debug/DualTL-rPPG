import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import torch
from model import DualTL, neg_pearson

dev = "cuda"
model = DualTL().to(dev)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Parameters: {n_params:.1f} M")

opt = torch.optim.Adam(model.parameters(), lr=1e-4)

for bs in [8, 16, 32, 64]:
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = torch.rand(bs, 63, 3, 300, device=dev)
        y = torch.randn(bs, 300, device=dev)
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = neg_pearson(model(x).float(), y)
        loss.backward()
        opt.step()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"batch {bs:3d}: OK  loss={loss.item():.3f}  peak GPU memory={peak:.2f} GB")
    except torch.cuda.OutOfMemoryError:
        print(f"batch {bs:3d}: out of memory")
        break