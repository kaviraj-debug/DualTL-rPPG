import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import json, random
from pathlib import Path

names = sorted(p.stem for p in Path("cache").glob("*.npz"))
print("Subjects in cache:", len(names))

random.seed(42)
random.shuffle(names)

split = {
    "train": sorted(names[:28]),
    "val": sorted(names[28:34]),
    "test": sorted(names[34:]),
}
json.dump(split, open("split.json", "w"), indent=2)

for k, v in split.items():
    print(k, len(v), v)