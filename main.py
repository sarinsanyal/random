"""
Eye-gaze + phone detection for assessment monitoring.

- MediaPipe Face Mesh (with iris refinement) -> gaze direction + head pose
- YOLOv8 (COCO) -> mobile phone detection (class 67 only, so bottles are ignored)

Keys:  q = quit   c = recalibrate
"""
import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# ----------------------------- CONFIG ---------------------------------------
CALIB_SECONDS = 3.0        # look straight at the screen during this time
LOOK_AWAY_SECONDS = 1.5    # gaze must stay off-screen this long before flagging
NO_FACE_SECONDS = 2.0
PHONE_CONF = 0.40          # YOLO confidence threshold
PHONE_MIN_FRAMES = 3       # phone must be seen in N detection passes to flag
DETECT_EVERY_N = 3         # run YOLO every N frames (speed)

# Deviation from calibrated "looking at screen" baseline
IRIS_H_THRESH = 0.12       # iris left/right inside the eye (0..1 scale)
IRIS_V_THRESH = 0.18       # iris up/down inside the eye
YAW_THRESH = 0.10          # head turn left/right
PITCH_THRESH = 0.09        # head up/down

COCO_CELL_PHONE = 67       # NOTE: bottle is 39 -> deliberately NOT requested
# ----------------------------------------------------------------------------

# MediaPipe landmark indices
R_EYE_OUTER, R_EYE_INNER, R_UP, R_LOW, R_IRIS = 33, 133, 159, 145, 468
L_EYE_INNER, L_EYE_OUTER, L_UP, L_LOW, L_IRIS = 362, 263, 386, 374, 473
NOSE, CHEEK_L, CHEEK_R, FOREHEAD, CHIN = 1, 234, 454, 10, 152


def project_ratio(p, a, b):
    """Where point p lies between a (0.0) and b (1.0)."""
    ab = b - a
    return float(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9))


def extract_features(pts):
    """Return (iris_h, iris_v, yaw, pitch) from landmark pixel coords."""
    h = (project_ratio(pts[R_IRIS], pts[R_EYE_OUTER], pts[R_EYE_INNER]) +
         project_ratio(pts[L_IRIS], pts[L_EYE_INNER], pts[L_EYE_OUTER])) / 2
    v = (project_ratio(pts[R_IRIS], pts[R_UP], pts[R_LOW]) +
         project_ratio(pts[L_IRIS], pts[L_UP], pts[L_LOW])) / 2
    yaw = project_ratio(pts[NOSE], pts[CHEEK_L], pts[CHEEK_R])
    pitch = project_ratio(pts[NOSE], pts[FOREHEAD], pts[CHIN])
    return np.array([h, v, yaw, pitch])


def gaze_label(delta):
    """Turn deviation from baseline into a direction label (or 'CENTER')."""
    dh, dv, dyaw, dpitch = delta
    labels = []
    if dh > IRIS_H_THRESH or dyaw > YAW_THRESH:
        labels.append("RIGHT")
    elif dh < -IRIS_H_THRESH or dyaw < -YAW_THRESH:
        labels.append("LEFT")
    if dv > IRIS_V_THRESH or dpitch > PITCH_THRESH:
        labels.append("DOWN")
    elif dv < -IRIS_V_THRESH or dpitch < -PITCH_THRESH:
        labels.append("UP")
    return "+".join(labels) if labels else "CENTER"


class EventLogger:
    def __init__(self, folder="logs"):
        os.makedirs(folder, exist_ok=True)
        self.folder = folder
        self.path = os.path.join(folder, f"session_{datetime.now():%Y%m%d_%H%M%S}.csv")
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "event", "detail", "snapshot"])

    def log(self, event, detail, frame):
        ts = datetime.now()
        snap = os.path.join(self.folder, f"{event}_{ts:%H%M%S_%f}.jpg")
        cv2.imwrite(snap, frame)
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([ts.isoformat(timespec="seconds"), event, detail, snap])
        print(f"[FLAG] {ts:%H:%M:%S} {event}: {detail}")


class Timer:
    """Tracks how long a condition has been continuously true."""
    def __init__(self):
        self.start = None

    def update(self, active):
        if active:
            if self.start is None:
                self.start = time.time()
            return time.time() - self.start
        self.start = None
        return 0.0


def draw_label(img, text, org, color):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO weights (auto-downloads)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera.")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=2, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    yolo = YOLO(args.model)
    logger = EventLogger()

    calib_samples, baseline = [], None
    calib_start = time.time()

    away_timer, noface_timer = Timer(), Timer()
    flagged = {"LOOKING_AWAY": False, "NO_FACE": False, "MULTIPLE_FACES": False, "PHONE": False}
    phone_hits, phone_boxes, frame_idx = 0, [], 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror so LEFT/RIGHT match the user's view
        h, w = frame.shape[:2]
        frame_idx += 1
        status_lines = []

        # ------------------------- face / gaze -------------------------------
        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faces = res.multi_face_landmarks or []

        face_missing = len(faces) == 0
        no_face_dur = noface_timer.update(face_missing)
        if no_face_dur > NO_FACE_SECONDS:
            status_lines.append(("NO FACE IN FRAME", (0, 0, 255)))
            if not flagged["NO_FACE"]:
                flagged["NO_FACE"] = True
                logger.log("NO_FACE", f"{no_face_dur:.1f}s", frame)
        else:
            flagged["NO_FACE"] = False

        multi = len(faces) > 1
        if multi:
            status_lines.append(("MULTIPLE FACES", (0, 0, 255)))
            if not flagged["MULTIPLE_FACES"]:
                flagged["MULTIPLE_FACES"] = True
                logger.log("MULTIPLE_FACES", f"{len(faces)} faces", frame)
        else:
            flagged["MULTIPLE_FACES"] = False

        if faces:
            lm = faces[0].landmark
            pts = np.array([[p.x * w, p.y * h] for p in lm])
            feats = extract_features(pts)

            for idx in (R_IRIS, L_IRIS):
                cv2.circle(frame, tuple(pts[idx].astype(int)), 3, (0, 255, 0), -1)

            if baseline is None:  # --- calibration phase
                elapsed = time.time() - calib_start
                calib_samples.append(feats)
                status_lines.append((f"CALIBRATING... look at screen ({elapsed:.1f}/{CALIB_SECONDS:.0f}s)",
                                     (0, 255, 255)))
                if elapsed >= CALIB_SECONDS:
                    baseline = np.median(np.array(calib_samples), axis=0)
                    print("Calibration done:", np.round(baseline, 3))
            else:  # --- monitoring phase
                label = gaze_label(feats - baseline)
                away = label != "CENTER"
                away_dur = away_timer.update(away and not multi)

                if away_dur >= LOOK_AWAY_SECONDS:
                    status_lines.append((f"LOOKING AWAY: {label} ({away_dur:.1f}s)", (0, 0, 255)))
                    if not flagged["LOOKING_AWAY"]:
                        flagged["LOOKING_AWAY"] = True
                        logger.log("LOOKING_AWAY", f"{label} for {away_dur:.1f}s", frame)
                else:
                    flagged["LOOKING_AWAY"] = False
                    colour = (0, 200, 255) if away else (0, 200, 0)
                    status_lines.append((f"Gaze: {label}", colour))
        else:
            away_timer.update(False)

        # ------------------------- phone detection ---------------------------
        if frame_idx % DETECT_EVERY_N == 0:
            det = yolo(frame, classes=[COCO_CELL_PHONE], conf=PHONE_CONF, verbose=False)[0]
            phone_boxes = [tuple(map(int, b.xyxy[0])) + (float(b.conf[0]),) for b in det.boxes]
            phone_hits = phone_hits + 1 if phone_boxes else 0

        if phone_hits >= PHONE_MIN_FRAMES:
            for x1, y1, x2, y2, conf in phone_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                draw_label(frame, f"PHONE {conf:.2f}", (x1, max(20, y1 - 8)), (0, 0, 255))
            status_lines.append(("MOBILE PHONE DETECTED", (0, 0, 255)))
            if not flagged["PHONE"]:
                flagged["PHONE"] = True
                logger.log("PHONE", f"conf={max(b[4] for b in phone_boxes):.2f}", frame)
        elif phone_hits == 0:
            flagged["PHONE"] = False

        # ------------------------- overlay -----------------------------------
        for i, (text, colour) in enumerate(status_lines):
            draw_label(frame, text, (10, 28 + 28 * i), colour)
        draw_label(frame, "q: quit | c: recalibrate", (10, h - 12), (255, 255, 255))

        cv2.imshow("Assessment Monitor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            baseline, calib_samples, calib_start = None, [], time.time()

    cap.release()
    cv2.destroyAllWindows()
    print(f"Session log saved to: {logger.path}")


if __name__ == "__main__":
    main()