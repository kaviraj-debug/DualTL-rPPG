import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import cv2, time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Could not open the webcam. Close other apps that use it (Zoom, Teams, Camera) and try again.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

n, fps, t0, size = 0, 0.0, time.time(), None
while True:
    ok, frame = cap.read()
    if not ok:
        break
    size = (frame.shape[1], frame.shape[0])
    n += 1
    if n % 30 == 0:
        fps = 30 / (time.time() - t0)
        t0 = time.time()
    cv2.putText(frame, f"fps {fps:.1f} - press q to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("webcam test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print(f"Measured fps: {fps:.1f} | frame size: {size}")
cap.release()
cv2.destroyAllWindows()