import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import cv2
import numpy as np
from tqdm import tqdm

try:
    import matplotlib.pyplot  # noqa: F401
except Exception:
    # Windows blocked a matplotlib DLL. MediaPipe only needs it for drawing helpers
    # we never use, so give it a harmless stand-in.
    import sys, types
    _mpl = types.ModuleType("matplotlib")
    _plt = types.ModuleType("matplotlib.pyplot")
    _mpl.pyplot = _plt
    sys.modules["matplotlib"] = _mpl
    sys.modules["matplotlib.pyplot"] = _plt

import mediapipe as mp

ROI_NAMES = ["forehead", "upper_cheek_L", "lower_cheek_L",
             "upper_cheek_R", "lower_cheek_R", "chin"]

# MediaPipe Face Mesh landmark indices (L/R = left/right side of the IMAGE).
# These are my design choice (the paper doesn't list exact points); check visually.
ROI_IDX = {
    "forehead":      [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109,
                      70, 63, 105, 66, 107, 336, 296, 334, 293, 300],
    "upper_cheek_L": [234, 145, 98, 61, 93],
    "lower_cheek_L": [93, 61, 172, 132],
    "upper_cheek_R": [454, 374, 327, 291, 323],
    "lower_cheek_R": [323, 291, 397, 361],
    "chin":          [172, 18, 397, 152],
}

# Combination matrix: row k-1 tells which of the 6 ROIs are in combination k (k = 1..63)
COMBO = np.array([[(k >> i) & 1 for i in range(6)] for k in range(1, 64)], np.float32)  # (63, 6)


def make_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)


def get_landmarks(frame_bgr, face_mesh):
    """Return (468, 2) pixel coordinates, or None if no face is found."""
    h, w = frame_bgr.shape[:2]
    res = face_mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    return np.array([[p.x * w, p.y * h] for p in lm], np.float32)


def roi_masks_from_landmarks(pts, shape):
    """Return a list of 6 boolean masks (H, W), one per ROI, with no overlap."""
    h, w = shape[:2]
    taken = np.zeros((h, w), np.uint8)
    masks = []
    for name in ROI_NAMES:
        hull = cv2.convexHull(pts[ROI_IDX[name]].astype(np.int32))
        m = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(m, hull, 1)
        m[taken == 1] = 0
        taken |= m
        masks.append(m.astype(bool))
    return masks


def build_mst_video(video_path, face_mesh):
    """Return array (T, 63, 3): mean Y, U, V of every ROI combination for every frame."""
    cap = cv2.VideoCapture(str(video_path))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = np.full((n_total, 63, 3), np.nan, np.float32)
    last_pts, failed = None, 0

    for t in tqdm(range(n_total), desc=video_path.parent.name, leave=False):
        ok, frame = cap.read()
        if not ok:
            break
        pts = get_landmarks(frame, face_mesh)
        if pts is None:
            failed += 1
            pts = last_pts            # reuse the previous frame's landmarks
        if pts is None:
            continue                  # no face yet; filled in below
        last_pts = pts

        masks = roi_masks_from_landmarks(pts, frame.shape)
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV).astype(np.float32)
        sums = np.stack([yuv[m].sum(0) for m in masks])               # (6, 3)
        cnts = np.array([m.sum() for m in masks], np.float32)         # (6,)
        out[t] = (COMBO @ sums) / np.maximum(COMBO @ cnts, 1)[:, None]
    cap.release()

    # fill frames with no face using the nearest valid frame
    valid = ~np.isnan(out[:, 0, 0])
    idx = np.where(valid, np.arange(len(out)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = out[idx]
    first = np.argmax(valid)
    out[:first] = out[first]
    return out, failed