import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from dataset import list_cached, loso_split

files = list_cached()
print("Cached subjects:", [f.stem for f in files])

test_name = files[0].stem
train_ds, test_ds = loso_split(test_name)
print(f"Test subject: {test_name}")
print("Train windows:", len(train_ds), "| Test windows:", len(test_ds))

x, y, name, s = train_ds[0]
print("x shape:", tuple(x.shape), "(expected (63, 3, 300))")
print("y shape:", tuple(y.shape), "(expected (300,))")
print("x min/max:", float(x.min()), float(x.max()), "| y mean/std:", float(y.mean()), float(y.std()))