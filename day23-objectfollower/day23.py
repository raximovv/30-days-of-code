import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

lower_hsv = np.array([35, 80, 80])
upper_hsv = np.array([85, 255, 255])

kp = 0.08
ki = 0.0005
kd = 0.04

integral_x = 0
integral_y = 0
prev_error_x = 0
prev_error_y = 0
prev_time = time.time()

min_area = 800

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]
    center_x = width // 2
    center_y = height // 2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_time = time.time()
    dt = current_time - prev_time
    if dt <= 0:
        dt = 1e-6

    object_found = False

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area > min_area:
            M = cv2.moments(largest_contour)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                error_x = cx - center_x
                error_y = center_y - cy

                integral_x += error_x * dt
                integral_y += error_y * dt

                derivative_x = (error_x - prev_error_x) / dt
                derivative_y = (error_y - prev_error_y) / dt

                output_x = kp * error_x + ki * integral_x + kd * derivative_x
                output_y = kp * error_y + ki * integral_y + kd * derivative_y

                prev_error_x = error_x
                prev_error_y = error_y
                prev_time = current_time

                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.line(frame, (center_x, center_y), (cx, cy), (255, 0, 0), 2)

                cv2.putText(frame, f"Centroid: ({cx}, {cy})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Error X: {error_x}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Error Y: {error_y}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"PID X: {output_x:.2f}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"PID Y: {output_y:.2f}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                object_found = True

    if not object_found:
        integral_x = 0
        integral_y = 0
        prev_error_x = 0
        prev_error_y = 0
        prev_time = current_time
        cv2.putText(frame, "No object found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.circle(frame, (center_x, center_y), 6, (255, 255, 0), -1)
    cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 0), 2)
    cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 0), 2)

    cv2.imshow("Object Follower", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()