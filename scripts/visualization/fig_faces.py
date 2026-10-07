import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import cv2
import numpy as np
from pathlib import Path
from preprocess import make_face_mesh, get_landmarks, roi_masks_from_landmarks

NAMES = ["forehead", "upper cheek (image left)", "lower cheek (image left)",
         "upper cheek (image right)", "lower cheek (image right)", "chin"]
GRAD = [14.7, 18.3, 16.8, 30.6, 12.3, 7.3]    # Grad-CAM-style importance in % (explain_gradcam.py)
OCC = [7.47, 3.82, 5.42, 8.18, 3.26, 0.0]     # MAE increase in bpm when the region is removed (explain_occlusion.py)
FONT = cv2.FONT_HERSHEY_SIMPLEX
S = 520                                        # size of each face picture in pixels

vids = sorted(Path("data/ubfc").rglob("vid.avi"))
if not vids:
    raise SystemExit("No video found under data/ubfc. Tell me and I will give you another way.")
cap = cv2.VideoCapture(str(vids[0]))
fm = make_face_mesh()
frame, pts = None, None
for i in range(300):
    ok, f = cap.read()
    if not ok:
        break
    if i < 60:
        continue
    p = get_landmarks(f, fm)
    if p is not None:
        frame, pts = f, p
        break
if frame is None:
    raise SystemExit("No face found in the first frames of " + str(vids[0]))
masks = roi_masks_from_landmarks(pts, frame.shape)

# crop a square around the face and enlarge it
allm = np.any(np.stack(masks), axis=0)
ys, xs = np.nonzero(allm)
cx, cy = int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2)
half = max(int(0.85 * max(xs.max() - xs.min(), ys.max() - ys.min())), 40)
fp = cv2.copyMakeBorder(frame, half, half, half, half, cv2.BORDER_REPLICATE)
img = cv2.resize(fp[cy:cy + 2 * half, cx:cx + 2 * half], (S, S), interpolation=cv2.INTER_CUBIC)
mk = []
for m in masks:
    mp = cv2.copyMakeBorder(m.astype(np.uint8), half, half, half, half, cv2.BORDER_CONSTANT, value=0)
    mk.append(cv2.resize(mp[cy:cy + 2 * half, cx:cx + 2 * half], (S, S), interpolation=cv2.INTER_NEAREST) > 0)


def jet(a):
    c = cv2.applyColorMap(np.uint8([[int(255 * a)]]), cv2.COLORMAP_JET)[0, 0]
    return tuple(int(x) for x in c)


def panel(vals, title, fmt):
    v = np.array(vals, float)
    v = v / (v.max() + 1e-9)
    over = img.copy()
    for m, a in zip(mk, v):
        over[m] = jet(a)
    out = cv2.addWeighted(img, 0.5, over, 0.5, 0)
    for i, m in enumerate(mk):
        yy, xx = np.nonzero(m)
        px, py = int(xx.mean()), int(yy.mean())
        cv2.circle(out, (px, py), 13, (255, 255, 255), -1)
        cv2.circle(out, (px, py), 13, (0, 0, 0), 1)
        cv2.putText(out, str(i + 1), (px - 6, py + 6), FONT, 0.55, (0, 0, 0), 2)
    head = np.full((46, S, 3), 255, np.uint8)
    cv2.putText(head, title, (8, 30), FONT, 0.6, (0, 0, 0), 2)
    leg = np.full((6 * 28 + 40, S, 3), 255, np.uint8)
    for i in range(6):
        y = 26 + 28 * i
        cv2.rectangle(leg, (8, y - 14), (26, y + 4), jet(v[i]), -1)
        cv2.putText(leg, f"{i + 1}  {NAMES[i]}", (36, y), FONT, 0.5, (0, 0, 0), 1)
        cv2.putText(leg, fmt.format(vals[i]), (S - 110, y), FONT, 0.5, (0, 0, 0), 1)
    cv2.putText(leg, "colour: blue = low, red = high (within this panel)", (8, 6 * 28 + 30), FONT, 0.45, (80, 80, 80), 1)
    return np.vstack([head, out, leg])


gap = np.full((46 + S + 6 * 28 + 40, 12, 3), 255, np.uint8)
fig = np.hstack([panel(GRAD, "Grad-CAM-style importance", "{:.1f} %"), gap,
                 panel(OCC, "Error increase when region removed", "+{:.1f} bpm")])
Path("outputs/plots").mkdir(parents=True, exist_ok=True)
cv2.imwrite("outputs/plots/fig_explain_faces_v2.png", fig)
print("Saved outputs/plots/fig_explain_faces_v2.png using a frame from", vids[0])