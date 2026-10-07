import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import numpy as np
import matplotlib.pyplot as plt
from utils import list_subjects, load_ground_truth, video_info, sync_ppg_to_frames

subjects = list_subjects("data/ubfc")
print("Subjects found:", len(subjects))

for s in subjects:
    ppg, hr, t = load_ground_truth(s / "ground_truth.txt")
    info = video_info(s / "vid.avi")
    video_dur = info["frames"] / info["fps"]
    gt_dur = t[-1] - t[0]
    print(f"{s.name:12s} frames={info['frames']:5d} fps={info['fps']:.1f} "
          f"size={info['width']}x{info['height']} video={video_dur:6.1f}s "
          f"gt={gt_dur:6.1f}s gt_samples={len(ppg)} meanHR={np.mean(hr):.1f}")

# Plot the first 10 seconds of the first subject to check the sync
s = subjects[0]
ppg, hr, t = load_ground_truth(s / "ground_truth.txt")
info = video_info(s / "vid.avi")
ppg_sync = sync_ppg_to_frames(ppg, t, info["frames"], info["fps"])
n = int(10 * info["fps"])
plt.figure(figsize=(10, 3))
plt.plot(np.arange(n) / info["fps"], ppg_sync[:n])
plt.xlabel("time (s)"); plt.title(f"{s.name}: synced PPG, first 10 s")
plt.tight_layout()
plt.savefig("outputs/plots/sync_check.png")
print("Saved outputs/plots/sync_check.png")