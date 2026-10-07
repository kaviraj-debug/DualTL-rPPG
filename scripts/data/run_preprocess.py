import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import argparse, shutil, time
import numpy as np
from pathlib import Path
# pyrefly: ignore [missing-import]
from utils import load_ground_truth, video_info, sync_ppg_to_frames
from preprocess import make_face_mesh, build_mst_video

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/ubfc", help="folder holding the subject folders (searched recursively)")
ap.add_argument("--max_new", type=int, default=0, help="stop after this many NEW subjects (0 = all)")
args = ap.parse_args()


def find_subjects(root):
    subs = {}
    for gt in Path(root).rglob("ground_truth.txt"):
        if (gt.parent / "vid.avi").exists():
            subs[gt.parent.name] = gt.parent

    def num(name):
        digits = "".join(c for c in name if c.isdigit())
        return int(digits) if digits else 0

    return [subs[k] for k in sorted(subs, key=num)]


def safe_copy(src, dst, chunk=16 * 1024 * 1024, tries=3):
    """Copy only the file data (no timestamps/permissions), in chunks, with retries."""
    for attempt in range(1, tries + 1):
        try:
            with open(src, "rb") as fi, open(dst, "wb") as fo:
                while True:
                    buf = fi.read(chunk)
                    if not buf:
                        break
                    fo.write(buf)
            return
        except OSError as e:
            print(f"  copy attempt {attempt}/{tries} failed: {e}")
            if attempt == tries:
                raise
            time.sleep(10)


Path("cache").mkdir(exist_ok=True)
subjects = find_subjects(args.root)
print("Subjects found:", len(subjects))

done = 0
for s in subjects:
    out_path = Path("cache") / f"{s.name}.npz"
    if out_path.exists():
        print("skip (cached):", s.name)
        continue
    if args.max_new and done >= args.max_new:
        break
    tmp = Path("tmp_video") / s.name
    stage = "start"
    try:
        tmp.mkdir(parents=True, exist_ok=True)
        local_vid = tmp / "vid.avi"
        stage = "copy video from Drive"
        safe_copy(s / "vid.avi", local_vid)
        stage = "process video"
        info = video_info(local_vid)
        fm = make_face_mesh()
        mst, failed = build_mst_video(local_vid, fm)
        fm.close()
        stage = "read ground truth"
        ppg, hr, t = load_ground_truth(s / "ground_truth.txt")
        ppg_sync = sync_ppg_to_frames(ppg, t, len(mst), info["fps"])
        stage = "save cache"
        np.savez_compressed(out_path, mst=mst, ppg=ppg_sync.astype(np.float32),
                            hr=np.float32(np.mean(hr)), fps=np.float32(info["fps"]))
        print(f"{s.name}: mst={mst.shape} frames_without_face={failed}")
        done += 1
    except Exception as e:
        print(f"FAILED {s.name} at stage '{stage}' -> {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)