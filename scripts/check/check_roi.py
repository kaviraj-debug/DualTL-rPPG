import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import cv2
import numpy as np
from utils import list_subjects
from preprocess import make_face_mesh, get_landmarks, roi_masks_from_landmarks

subject = list_subjects("data/ubfc")[0]
cap = cv2.VideoCapture(str(subject / "vid.avi"))
fm = make_face_mesh()

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
          (0, 255, 255), (255, 0, 255), (255, 255, 0)]
show = [0, 150, 299]
tiles, idx = [], 0

while idx <= max(show):
    ok, frame = cap.read()
    if not ok:
        break
    if idx in show:
        pts = get_landmarks(frame, fm)
        if pts is None:
            print("No face found at frame", idx)
            tiles.append(frame)
        else:
            masks = roi_masks_from_landmarks(pts, frame.shape)
            overlay = frame.copy()
            for m, c in zip(masks, colors):
                overlay[m] = c
            tiles.append(cv2.addWeighted(frame, 0.5, overlay, 0.5, 0))
    idx += 1

cv2.imwrite("outputs/plots/roi_check.png", np.hstack(tiles))
print("Saved outputs/plots/roi_check.png")