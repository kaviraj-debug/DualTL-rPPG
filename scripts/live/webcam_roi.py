import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import cv2
from preprocess import make_face_mesh, get_landmarks, roi_masks_from_landmarks

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Could not open the webcam. Close other apps that use it and try again.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

fm = make_face_mesh()
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
          (0, 255, 255), (255, 0, 255), (255, 255, 0)]
n, lost = 0, 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    n += 1
    pts = get_landmarks(frame, fm)
    if pts is None:
        lost += 1
        shown, msg, col = frame, "NO FACE FOUND", (0, 0, 255)
    else:
        masks = roi_masks_from_landmarks(pts, frame.shape)
        overlay = frame.copy()
        for m, c in zip(masks, colors):
            overlay[m] = c
        shown = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        msg, col = "face OK", (0, 255, 0)
    cv2.putText(shown, f"{msg} - press q to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    cv2.imshow("face regions", shown)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print(f"frames: {n} | frames without a face: {lost} ({100 * lost / max(n, 1):.1f}%)")
cap.release()
cv2.destroyAllWindows()