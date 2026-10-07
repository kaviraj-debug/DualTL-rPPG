from pathlib import Path
import numpy as np
import cv2


def list_subjects(root):
    """Return subject folders that contain vid.avi and ground_truth.txt, sorted by number."""
    root = Path(root)
    subs = [p for p in root.iterdir()
            if p.is_dir() and (p / "vid.avi").exists() and (p / "ground_truth.txt").exists()]
    def num(p):
        digits = "".join(ch for ch in p.name if ch.isdigit())
        return int(digits) if digits else 0
    return sorted(subs, key=num)


def load_ground_truth(path):
    """Return ppg, hr, t (each 1-D numpy array) from ground_truth.txt."""
    data = np.loadtxt(path)
    if data.shape[0] != 3 and data.shape[1] == 3:   # in case it is stored as columns
        data = data.T
    ppg, hr, t = data[0], data[1], data[2]
    return ppg, hr, t


def video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    info = dict(
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=cap.get(cv2.CAP_PROP_FPS),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def sync_ppg_to_frames(ppg, t, n_frames, fps):
    """Resample the PPG signal so there is exactly one value per video frame."""
    frame_times = np.arange(n_frames) / fps
    return np.interp(frame_times, t - t[0], ppg)