"""
Live heart-rate demo. Pipeline, with the function that does each stage:
  STAGE 1  MediaPipe Face Mesh  -> face landmarks -> 6 face regions       (get_landmarks, roi_masks_from_landmarks)
  STAGE 2  Colour signals       -> mean Y,U,V of 63 region combinations   (frame_column)
  STAGE 3  Pre-processing       -> band-pass + z-score, same as training  (estimate_hr)
  STAGE 4  Dual-TL model        -> predicts the pulse waveform            (model(...) inside estimate_hr)
  STAGE 5  Read the heart rate  -> frequency peak of the model's output   (hr_fft)
MediaPipe only finds the face. The pulse waveform is produced by the Dual-TL network.
CHROM (a classical method, no learning) runs on the same signal as a cross-check.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import csv, time
from collections import deque
from pathlib import Path
import numpy as np
import cv2
import torch
from scipy.signal import butter, filtfilt

from preprocess import make_face_mesh, get_landmarks, roi_masks_from_landmarks, COMBO
from dataset import bandpass, zscore_rows
from model import DualTL

CKPT = "checkpoints/split_filtered_D128L2_stretch40_best.pt"   # Dual-TL trained with time-stretch 0.4
FS = 30.0           # frame rate the model was trained on
WIN = 300           # model window (frames)
MARGIN = 90         # frames after the window end (keeps filter edge effects out)
MIN_SECONDS = 18    # wait this long before the first estimate
MAX_FRAMES = 900    # keep the last ~30 s
EVERY = 1.0         # new estimate every second
BRIEF_GAP = 5       # frames: reuse the last landmarks for a short flicker
RESET_GAP = 15      # frames (~0.5 s) without a face: discard collected data
SMOOTH_N = 15       # display = median of the last 15 accepted estimates
JUMP = 15           # bpm: ignore an estimate this far from the recent median...
MAX_REJECT = 8      # ...unless it happens this many times in a row (then it is a real change)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def frame_column(frame, masks):
    """STAGE 2: same computation as in training: mean Y, U, V of all 63 ROI combinations."""
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV).astype(np.float32)
    sums = np.stack([yuv[m].sum(0) for m in masks])
    cnts = np.array([m.sum() for m in masks], np.float32)
    return (COMBO @ sums) / np.maximum(COMBO @ cnts, 1)[:, None]


def hr_fft(x, fs, lo=0.75, hi=2.5, nfft=4096):
    """STAGE 5: return (bpm, quality). quality = share of band power within +-0.1 Hz of the peak."""
    x = (x - x.mean()) * np.hanning(len(x))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    p = np.abs(np.fft.rfft(x, nfft)) ** 2
    m = (f >= lo) & (f <= hi)
    fm, pm = f[m], p[m]
    k = int(np.argmax(pm))
    near = np.abs(fm - fm[k]) <= 0.1
    return 60 * float(fm[k]), float(pm[near].sum() / (pm.sum() + 1e-12))


def yuv_to_rgb(yuv):
    Y, U, V = yuv[:, 0], yuv[:, 1] - 128.0, yuv[:, 2] - 128.0
    B = Y + U / 0.492
    R = Y + V / 0.877
    G = (Y - 0.299 * R - 0.114 * B) / 0.587
    return np.stack([R, G, B], axis=1)


def chrom_signal(rgb):
    """CHROM cross-check (classical method, no learning)."""
    n = rgb / rgb.mean(0)
    xs = 3 * n[:, 0] - 2 * n[:, 1]
    ys = 1.5 * n[:, 0] + n[:, 1] - 1.5 * n[:, 2]
    b, a = butter(2, [0.75, 2.5], btype="bandpass", fs=FS)
    xf, yf = filtfilt(b, a, xs), filtfilt(b, a, ys)
    return xf - (xf.std() / (yf.std() + 1e-8)) * yf


def estimate_hr(times, cols, model, dev):
    t = np.array(times)
    X = np.array(cols).reshape(len(t), -1)                       # (n, 189)
    tu = np.arange(t[0], t[-1], 1.0 / FS)                        # steady 30 Hz time axis
    Xr = np.stack([np.interp(tu, t, X[:, k]) for k in range(X.shape[1])], axis=1)
    # STAGE 3: pre-processing (same filter and normalization as in training)
    Xu = bandpass(Xr - Xr.mean(0), FS)
    end = len(Xu) - MARGIN
    seg = Xu[end - WIN:end]                                      # (300, 189)
    x = seg.reshape(WIN, 63, 3).transpose(1, 2, 0)               # (63, 3, 300)
    x = zscore_rows(x).astype(np.float32)
    # STAGE 4: the Dual-TL model predicts the pulse waveform (300 points)
    with torch.no_grad():
        pulse = model(torch.from_numpy(x)[None].to(dev)).float().cpu().numpy()[0]
    # STAGE 5: filter the model's waveform and read its dominant frequency
    b, a = butter(1, [0.75, 2.5], btype="bandpass", fs=FS)
    pulse_f = filtfilt(b, a, pulse)
    bpm, q = hr_fft(pulse_f, FS)
    # cross-check: CHROM on the same frames (all-regions colour signal = combination 63)
    rgb = yuv_to_rgb(Xr[end - WIN:end, 186:189])
    bpm_c = hr_fft(chrom_signal(rgb), FS)[0]
    return bpm, q, pulse_f, bpm_c


def label(img, text, x, y, scale=0.45):
    cv2.putText(img, text, (x, y), FONT, scale, (0, 0, 0), 3)
    cv2.putText(img, text, (x, y), FONT, scale, (255, 255, 255), 1)


def draw_wave(img, wave, x0, y0, w=260, h=70):
    """Draw the waveform predicted by Dual-TL in a small panel."""
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (30, 30, 30), -1)
    cv2.putText(img, "Dual-TL output: predicted pulse", (x0 + 5, y0 + 14), FONT, 0.45, (0, 255, 255), 1)
    if wave is None:
        return
    v = np.clip((wave - wave.mean()) / (wave.std() + 1e-8), -2.5, 2.5)
    xs = np.linspace(x0 + 4, x0 + w - 4, len(v)).astype(np.int32)
    ys = (y0 + 22 + (h - 28) * (0.5 - v / 5.0)).astype(np.int32)
    cv2.polylines(img, [np.stack([xs, ys], 1).reshape(-1, 1, 2)], False, (0, 255, 0), 1)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = DualTL(D=128, L=2).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location=dev))
    model.eval()
    print("Loaded Dual-TL weights:", CKPT)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Close other apps that use it and try again.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    fm = make_face_mesh()                                        # STAGE 1: MediaPipe Face Mesh
    times, cols = deque(maxlen=MAX_FRAMES), deque(maxlen=MAX_FRAMES)
    lost, recent = deque(maxlen=90), deque(maxlen=SMOOTH_N)
    last_pts, bpm_shown, q_shown, wave_shown, last_est = None, None, None, None, 0.0
    miss, rej = 0, 0
    n, n_lost, fps, t_fps = 0, 0, 0.0, time.perf_counter()
    raw_all, chrom_all, shown_all = [], [], []

    Path("outputs").mkdir(exist_ok=True)
    log = open("outputs/live_log.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["seconds_since_start", "bpm_raw", "bpm_smoothed", "quality", "bpm_chrom"])
    t_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.perf_counter()
        n += 1
        if n % 30 == 0:
            fps = 30 / (now - t_fps)
            t_fps = now

        pts = get_landmarks(frame, fm)                           # STAGE 1: landmarks
        if pts is None:
            miss += 1
            n_lost += 1
            lost.append(1)
            if miss <= BRIEF_GAP and last_pts is not None:
                pts = last_pts                                   # brief flicker only
        else:
            miss = 0
            lost.append(0)
            last_pts = pts

        if miss >= RESET_GAP:                                    # the face is really gone: forget everything
            times.clear(); cols.clear(); recent.clear()
            bpm_shown, q_shown, wave_shown, last_pts, rej = None, None, None, None, 0

        shown = frame
        if pts is not None:
            masks = roi_masks_from_landmarks(pts, frame.shape)   # STAGE 1: 6 regions
            times.append(now)
            cols.append(frame_column(frame, masks))              # STAGE 2: colour signals
            overlay = frame.copy()
            for m, c in zip(masks, COLORS):
                overlay[m] = c
            shown = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)

        have = (times[-1] - times[0]) if len(times) > 1 else 0.0
        if pts is not None and have >= MIN_SECONDS and now - last_est >= EVERY:
            last_est = now
            raw, q, wave, raw_c = estimate_hr(times, cols, model, dev)   # STAGES 3-5, model called here
            raw_all.append(raw); chrom_all.append(raw_c)
            wave_shown = wave
            med = float(np.median(recent)) if len(recent) >= 8 else None
            if med is not None and abs(raw - med) > JUMP and rej < MAX_REJECT:
                rej += 1                                         # ignore a sudden jump
            else:
                if med is not None and abs(raw - med) > JUMP:    # jumps keep coming: a real change
                    recent.clear()
                rej = 0
                recent.append(raw)
            if len(recent) >= 5:
                bpm_shown, q_shown = float(np.median(recent)), q
                shown_all.append(bpm_shown)
                wr.writerow([round(now - t_start, 1), round(raw, 1), round(bpm_shown, 1),
                             round(q, 2), round(raw_c, 1)])
                log.flush()

        if miss >= RESET_GAP:
            cv2.putText(shown, "NO FACE - show your face to the camera", (10, 40), FONT, 0.8, (0, 0, 255), 2)
        elif bpm_shown is None and have < MIN_SECONDS:
            cv2.putText(shown, f"collecting data: {have:.0f}/{MIN_SECONDS} s - sit still",
                        (10, 35), FONT, 0.8, (0, 255, 255), 2)
        elif bpm_shown is None:
            cv2.putText(shown, "stabilizing...", (10, 35), FONT, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(shown, f"HR: {bpm_shown:.0f} bpm", (10, 50), FONT, 1.4, (0, 255, 0), 3)
            cv2.putText(shown, f"quality {q_shown:.2f}", (10, 80), FONT, 0.6, (0, 255, 0), 2)

        lx = shown.shape[1] - 310
        label(shown, "1 MediaPipe: face regions (coloured)", lx, 110)
        label(shown, "2-3 colour signals + filter", lx, 128)
        label(shown, "4 Dual-TL: pulse waveform (below)", lx, 146)
        label(shown, "5 frequency peak of that wave = HR", lx, 164)
        draw_wave(shown, wave_shown, shown.shape[1] - 270, shown.shape[0] - 95)

        lost_pct = 100 * float(np.mean(lost)) if lost else 0.0
        cv2.putText(shown, f"fps {fps:.0f} | lost {lost_pct:.0f}% | q = quit", (10, shown.shape[0] - 12),
                    FONT, 0.6, (255, 255, 255), 2)
        cv2.imshow("Dual-TL live heart rate", shown)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    log.close()

    if not raw_all:
        print("No estimate was produced (run it for at least 25 seconds).")
        return
    r, c = np.array(raw_all), np.array(chrom_all)
    print(f"\nestimates: {len(r)} | frames without a face: {100 * n_lost / max(n, 1):.1f}%")
    print(f"Dual-TL raw : median {np.median(r):.1f} | std {r.std():.1f} | range {r.min():.0f}-{r.max():.0f} bpm")
    print(f"CHROM raw   : median {np.median(c):.1f} | std {c.std():.1f} | range {c.min():.0f}-{c.max():.0f} bpm")
    print(f"|Dual-TL - CHROM|: mean {np.abs(r - c).mean():.1f} bpm | differ by more than 10 bpm in "
          f"{100 * np.mean(np.abs(r - c) > 10):.0f}% of estimates")
    if shown_all:
        s = np.array(shown_all)
        print(f"Displayed   : median {np.median(s):.1f} | std {s.std():.1f} | range {s.min():.0f}-{s.max():.0f} bpm")
    try:
        txt = input("\nReference heart rate you measured (press Enter to skip): ").strip()
        ref = float(txt) if txt else None
    except Exception:
        ref = None
    if ref is not None:
        print(f"vs reference {ref:.0f} bpm: Dual-TL raw MAE {np.abs(r - ref).mean():.1f} | "
              f"CHROM raw MAE {np.abs(c - ref).mean():.1f}"
              + (f" | displayed MAE {np.abs(np.array(shown_all) - ref).mean():.1f}" if shown_all else ""))


if __name__ == "__main__":
    main()