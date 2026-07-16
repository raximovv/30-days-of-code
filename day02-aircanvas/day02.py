import cv2
import mediapipe as mp
import numpy as np
import math

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    exit(1)

ret, test_frame = cap.read()
FRAME_H, FRAME_W = test_frame.shape[:2]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

THUMB_TIP = 4
INDEX_TIP = 8

prev_x, prev_y = None, None
was_drawing = False

PINCH_THRESHOLD = 40

COLORS = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
]

COLOR_NAMES = ["Green", "Red", "Blue", "Yellow"]

current_color_index = 0

def get_distance(lm1, lm2, w, h):
    x1, y1 = int(lm1.x * w), int(lm1.y * h)
    x2, y2 = int(lm2.x * w), int(lm2.y * h)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    drawing_now = False

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = hand_landmarks.landmark
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]

        distance = get_distance(thumb, index, FRAME_W, FRAME_H)
        ix, iy = int(index.x * FRAME_W), int(index.y * FRAME_H)

        cv2.circle(frame, (ix, iy), 8, COLORS[current_color_index], -1)

        if distance < PINCH_THRESHOLD:
            drawing_now = True
            if was_drawing and prev_x is not None:
                cv2.line(canvas, (prev_x, prev_y), (ix, iy),
                         COLORS[current_color_index], thickness=4)
            prev_x, prev_y = ix, iy
        else:
            prev_x, prev_y = None, None

        cv2.putText(frame, f"Distance: {distance:.0f}px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {PINCH_THRESHOLD}px", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        prev_x, prev_y = None, None

    was_drawing = drawing_now

    status = "DRAWING" if drawing_now else "NOT DRAWING"
    color = (0, 255, 0) if drawing_now else (100, 100, 100)
    cv2.putText(frame, status, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.putText(frame, f"Color: {COLOR_NAMES[current_color_index]}",
                (10, FRAME_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                COLORS[current_color_index], 2)

    mask = canvas > 0
    frame[mask] = canvas[mask]

    cv2.imshow("AirCanvas - Day 02", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    elif ord('1') <= key <= ord('9'):
        idx = key - ord('1')
        if idx < len(COLORS):
            current_color_index = idx

cap.release()
cv2.destroyAllWindows()