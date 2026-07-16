import cv2
import mediapipe as mp
import time
import sys

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    sys.exit(1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE_TOP = [159, 160, 161]
LEFT_EYE_BOTTOM = [145, 144, 153]
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

RIGHT_EYE_TOP = [386, 387, 388]
RIGHT_EYE_BOTTOM = [374, 373, 380]
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

def get_ear(landmarks, top_ids, bottom_ids, left_id, right_id):
    vertical = 0
    for t, b in zip(top_ids, bottom_ids):
        vertical += abs(landmarks[t].y - landmarks[b].y)
    vertical /= len(top_ids)
    horizontal = abs(landmarks[left_id].x - landmarks[right_id].x)
    if horizontal == 0:
        return 0.0
    ear = vertical / horizontal
    return ear

EAR_THRESHOLD = 0.2



BLINK_TIME_WINDOW = 2.0
BLINKS_TO_LOCK = 3
MIN_BLINK_DURATION = 2

STATE_IDLE = "IDLE"
STATE_COUNTING = "COUNTING"
STATE_LOCKED = "LOCKED"

state = STATE_IDLE
blink_count = 0
counting_start_time = 0
eye_closed_frames = 0
eye_was_closed = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if state == STATE_LOCKED:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        cv2.putText(frame, "LOCKED", (w // 2 - 120, h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
        cv2.putText(frame, "Press 'u' to unlock", (w // 2 - 130, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.imshow("BlinkLock - Day 04", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('u'):
            state = STATE_IDLE
            blink_count = 0
            print("UNLOCKED")
        elif key == ord('q'):
            break
        continue

    current_ear = 0.0

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = get_ear(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
        right_ear = get_ear(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
        current_ear = (left_ear + right_ear) / 2

        eye_is_closed = current_ear < EAR_THRESHOLD

        if eye_is_closed:
            eye_closed_frames += 1
        else:
            if eye_closed_frames >= MIN_BLINK_DURATION:
                if state == STATE_IDLE:
                    state = STATE_COUNTING
                    blink_count = 1
                    counting_start_time = time.time()
                    print(f"Blink {blink_count}/{BLINKS_TO_LOCK}")
                elif state == STATE_COUNTING:
                    blink_count += 1
                    print(f"Blink {blink_count}/{BLINKS_TO_LOCK}")
                    if blink_count >= BLINKS_TO_LOCK:
                        state = STATE_LOCKED
                        blink_count = 0
                        print("LOCKED! Press 'u' to unlock.")
            eye_closed_frames = 0

        if state == STATE_COUNTING:
            elapsed = time.time() - counting_start_time
            if elapsed > BLINK_TIME_WINDOW:
                print(f"Timeout. Only got {blink_count}/{BLINKS_TO_LOCK} blinks. Resetting.")
                state = STATE_IDLE
                blink_count = 0

        for idx in LEFT_EYE_TOP + LEFT_EYE_BOTTOM + RIGHT_EYE_TOP + RIGHT_EYE_BOTTOM:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    ear_color = (0, 0, 255) if current_ear < EAR_THRESHOLD else (0, 255, 0)
    cv2.putText(frame, f"EAR: {current_ear:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
    cv2.putText(frame, f"Threshold: {EAR_THRESHOLD}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    state_colors = {STATE_IDLE: (200, 200, 200), STATE_COUNTING: (0, 200, 255), STATE_LOCKED: (0, 0, 255)}
    cv2.putText(frame, f"State: {state}", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_colors[state], 2)

    if state == STATE_COUNTING:
        elapsed = time.time() - counting_start_time
        remaining = max(0, BLINK_TIME_WINDOW - elapsed)
        cv2.putText(frame, f"Blinks: {blink_count}/{BLINKS_TO_LOCK}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, f"Time left: {remaining:.1f}s", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        bar_width = int((remaining / BLINK_TIME_WINDOW) * 200)
        cv2.rectangle(frame, (10, 170), (10 + bar_width, 180), (0, 200, 255), -1)
        cv2.rectangle(frame, (10, 170), (210, 180), (100, 100, 100), 1)

    cv2.putText(frame, f"Blink {BLINKS_TO_LOCK}x rapidly to LOCK | 'u' = unlock | 'q' = quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("BlinkLock - Day 04", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('u') and state == STATE_LOCKED:
        state = STATE_IDLE
        blink_count = 0
        print("UNLOCKED")

cap.release()
cv2.destroyAllWindows()