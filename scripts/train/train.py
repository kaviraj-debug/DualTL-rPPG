import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import argparse, json, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import make_datasets
from model import DualTL, neg_pearson


def neg_pearson_shift(pred, gt, S=10):
    """1 - (best Pearson correlation over shifts in [-S, S] frames), averaged over the batch."""
    T = pred.size(1)
    g = gt[:, S:T - S]
    g = g - g.mean(dim=1, keepdim=True)
    best = None
    for s in range(-S, S + 1):
        p = pred[:, S + s:T - S + s]
        p = p - p.mean(dim=1, keepdim=True)
        r = (p * g).sum(1) / (p.norm(dim=1) * g.norm(dim=1) + 1e-8)
        best = r if best is None else torch.maximum(best, r)
    return (1 - best).mean()


def run_epoch(model, loader, dev, loss_fn, opt=None, scaler=None, accum=1):
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    if train:
        opt.zero_grad()
    for i, (x, y, _, _) in enumerate(loader):
        x, y = x.to(dev), y.to(dev)
        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(x)
            loss = loss_fn(pred.float(), y)
        if train:
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or i + 1 == len(loader):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
        total += loss.item() * len(x)
        n += len(x)
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--mode", default="filtered", choices=["filtered", "paper"])
    ap.add_argument("--shift", type=int, default=0, help="0 = plain neg. Pearson; S>0 = shift-tolerant (+-S frames)")
    ap.add_argument("--stretch", type=float, default=0.0, help="training speed augmentation, e.g. 0.25 = x0.75 to x1.25")
    ap.add_argument("--step_train", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="", help="suffix added to saved file names")
    ap.add_argument("--final_test", action="store_true", help="evaluate the best model on the TEST subjects (use once, at the end)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda"
    train_ds, val_ds, test_ds = make_datasets(step_train=args.step_train, mode=args.mode, stretch=args.stretch)
    print(f"windows: train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)} (test not used unless --final_test)")

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=0)

    model = DualTL(D=args.D, L=args.L, dropout=args.dropout).to(dev)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_par:.2f} M | mode={args.mode} | shift={args.shift} | stretch={args.stretch} | lr={args.lr}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = (lambda p, y: neg_pearson_shift(p, y, args.shift)) if args.shift > 0 else neg_pearson

    tag = f"split_{args.mode}_D{args.D}L{args.L}" + (f"_S{args.shift}" if args.shift else "")
    if args.name:
        tag += "_" + args.name
    ckpt = f"checkpoints/{tag}_best.pt"
    best, best_ep, history = 1e9, 0, []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_dl, dev, loss_fn, opt, scaler, args.accum)
        va = run_epoch(model, val_dl, dev, loss_fn)
        mark = ""
        if va < best:
            best, best_ep, mark = va, ep, "  <- best so far, saved"
            torch.save(model.state_dict(), ckpt)
        history.append({"epoch": ep, "train_loss": tr, "val_loss": va})
        print(f"epoch {ep:2d}/{args.epochs} | train {tr:.4f} | val {va:.4f} | {time.time()-t0:.0f}s{mark}")

    json.dump(history, open(f"outputs/history_{tag}.json", "w"), indent=2)
    print(f"Best validation loss {best:.4f} at epoch {best_ep}. Saved {ckpt}")

    if args.final_test:
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        te = run_epoch(model, test_dl, dev, loss_fn)
        print(f"FINAL TEST loss (best-val model): {te:.4f}")


if __name__ == "__main__":
    main()