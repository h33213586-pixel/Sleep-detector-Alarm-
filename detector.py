import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import pygame
import time
import sys
import os

# ─── SETTINGS ───────────────────────────────
EAR_THRESHOLD  = 0.22   # Aankh thori si bhi band ho to detect
ALARM_SECONDS  = 4      # 4 second baad alarm
SIREN_FREQ     = 880
MODEL_PATH     = "shape_predictor_68_face_landmarks.dat"

# ─── AUDIO ──────────────────────────────────
def init_audio():
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

def make_siren_sound(duration=0.5):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    freq = SIREN_FREQ + 250 * np.sin(2 * np.pi * 3 * t)
    wave = (32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    wave_stereo = np.column_stack([wave, wave])
    return pygame.sndarray.make_sound(wave_stereo)

# ─── EAR ────────────────────────────────────
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ─── MAIN ───────────────────────────────────
def main():
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] shape_predictor_68_face_landmarks.dat nahi mili!")
        sys.exit(1)

    init_audio()
    siren_sound = make_siren_sound()

    print("[INFO] Models load ho rahe hain...")
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL_PATH)

    # Multiple cascades — kisi bhi angle mein face pakad lega
    frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    print("[INFO] Camera chal rahi hai... (Q = band karo)")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Camera nahi mili!")
        sys.exit(1)

    eyes_closed_since = None
    alarm_active      = False
    blink_ignore_time = 0.3  # 0.3 sec se kam = blink, ignore karo

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = imutils.resize(frame, width=720)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Gray
        if len(frame.shape) == 2:
            gray = frame.copy()
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.array(gray, dtype=np.uint8)

        # ── Face detect: dlib first ──
        dlib_faces = detector(gray, 0)

        # Agar dlib ne nahi pakda — cascade use karo
        extra_rects = []
        if len(dlib_faces) == 0:
            # Front
            cv_f = frontal_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
            for (x,y,w,h) in cv_f:
                extra_rects.append(dlib.rectangle(int(x), int(y), int(x+w), int(y+h)))
            # Profile (left)
            cv_p = profile_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
            for (x,y,w,h) in cv_p:
                extra_rects.append(dlib.rectangle(int(x), int(y), int(x+w), int(y+h)))
            # Profile (right — flipped)
            flipped = cv2.flip(gray, 1)
            fw = gray.shape[1]
            cv_pf = profile_cascade.detectMultiScale(flipped, 1.1, 5, minSize=(80,80))
            for (x,y,w,h) in cv_pf:
                extra_rects.append(dlib.rectangle(int(fw-x-w), int(y), int(fw-x), int(y+h)))

        all_faces = list(dlib_faces) + extra_rects

        h_frame, w_frame = frame.shape[:2]
        status_text  = "JAGTA HUA  ✓"
        status_color = (0, 220, 0)
        face_found   = False
        ear_value    = 0.0
        eyes_closed_now = False

        for face in all_faces:
            try:
                shape    = predictor(gray, face)
                shape_np = face_utils.shape_to_np(shape)

                left_eye  = shape_np[lStart:lEnd]
                right_eye = shape_np[rStart:rEnd]

                left_ear  = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                ear_value = ear
                face_found = True

                # Aankh outline
                cv2.drawContours(frame, [cv2.convexHull(left_eye)],  -1, (0, 255, 255), 1)
                cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 255), 1)

                x1 = max(face.left(), 0)
                y1 = max(face.top(), 0)
                x2 = min(face.right(), w_frame)
                y2 = min(face.bottom(), h_frame)

                if ear < EAR_THRESHOLD:
                    eyes_closed_now = True

                    if eyes_closed_since is None:
                        eyes_closed_since = time.time()

                    closed_sec = time.time() - eyes_closed_since

                    # Blink ignore karo (0.3 sec se kam)
                    if closed_sec < blink_ignore_time:
                        status_text  = "JAGTA HUA  ✓"
                        status_color = (0, 220, 0)
                        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,220,0),2)

                    elif closed_sec < ALARM_SECONDS:
                        # Band hai — countdown chal raha hai
                        remaining    = ALARM_SECONDS - closed_sec
                        status_text  = f"AANKH BAND!  {remaining:.1f}s mein alarm"
                        status_color = (0, 140, 255)
                        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,140,255),2)

                        # Progress bar
                        progress = int((closed_sec / ALARM_SECONDS) * (w_frame - 20))
                        cv2.rectangle(frame, (10, h_frame-60), (10+progress, h_frame-50), (0,140,255), -1)
                        cv2.rectangle(frame, (10, h_frame-60), (w_frame-10, h_frame-50), (100,100,100), 1)

                    else:
                        # 4 second ho gaye — ALARM!
                        status_text  = "⚠  NEEND AA RAHI HAI!  ⚠"
                        status_color = (0, 0, 255)
                        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255),3)

                        # Progress bar full red
                        cv2.rectangle(frame, (10, h_frame-60), (w_frame-10, h_frame-50), (0,0,255), -1)

                        alarm_active = True
                        # Lagatar bajta rahe
                        if not pygame.mixer.get_busy():
                            siren_sound.play()

                else:
                    cv2.rectangle(frame, (x1,y1),(x2,y2),(0,220,0),2)

            except Exception:
                pass

        # Aankh khul gayi — sab reset
        if not eyes_closed_now:
            eyes_closed_since = None
            if alarm_active:
                pygame.mixer.stop()
                alarm_active = False

        # ── HUD ──────────────────────────────
        # Top bar
        ov = frame.copy()
        cv2.rectangle(ov, (0,0),(w_frame,55),(15,15,15),-1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

        # Bottom bar
        ov2 = frame.copy()
        cv2.rectangle(ov2,(0,h_frame-70),(w_frame,h_frame),(15,15,15),-1)
        cv2.addWeighted(ov2, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, "DROWSINESS DETECTOR",
                    (12,35), cv2.FONT_HERSHEY_DUPLEX, 0.75, (220,220,220), 1)

        if face_found:
            ear_color = (0,220,0) if ear_value >= EAR_THRESHOLD else (0,0,255)
            cv2.putText(frame, f"EAR: {ear_value:.2f}",
                        (w_frame-130,35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 1)

        if not face_found:
            cv2.putText(frame, "Chehra nahi mila...",
                        (int(w_frame*0.25), int(h_frame*0.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,80,255), 2)

        cv2.putText(frame, status_text,
                    (12, h_frame-75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

        cv2.putText(frame, "Q = Quit",
                    (w_frame-100, h_frame-75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1)

        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    pygame.mixer.stop()
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("[INFO] Program band ho gaya.")

if __name__ == "__main__":
    main()