import cv2
import mediapipe as mp
import pygame
import sys
import os

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

pygame.mixer.init()
MUSIC_FILE = "music.mp3"
if os.path.exists(MUSIC_FILE):
    pygame.mixer.music.load(MUSIC_FILE)
    print("Music loaded!")
else:
    print("WARNING: music.mp3 not found!")

LEFT_IRIS = 468
RIGHT_IRIS = 473
NOSE_TIP = 1
is_playing = False
GAZE_THRESHOLD = -0.08

print("RockLook is running!")
print("Look DOWN to play music. Look UP to pause.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_iris_y = landmarks[LEFT_IRIS].y
        right_iris_y = landmarks[RIGHT_IRIS].y
        nose_y = landmarks[NOSE_TIP].y

        iris_y = (left_iris_y + right_iris_y) / 2
        gaze_offset = iris_y - nose_y

        looking_down = gaze_offset > GAZE_THRESHOLD
        status = "LOOKING DOWN" if looking_down else "LOOKING UP"
        color = (0, 0, 255) if looking_down else (0, 255, 0)

        cv2.putText(frame, f"Gaze offset: {gaze_offset:.4f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {GAZE_THRESHOLD}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, status, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        if looking_down and not is_playing:
            if os.path.exists(MUSIC_FILE):
                if pygame.mixer.music.get_pos() == -1:
                    pygame.mixer.music.play()
                else:
                    pygame.mixer.music.unpause()
                is_playing = True
                print("Playing music!")
        elif not looking_down and is_playing:
            pygame.mixer.music.pause()
            is_playing = False
            print("Music paused!")

    else:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("RockLook - Day 01", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("Done!")