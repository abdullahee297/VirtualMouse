import cv2
import time
import autopy
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ------------------ MediaPipe Setup ------------------
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

# ------------------ Hand Connections ------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),                  # Thumb
    (0, 5), (5, 9), (9, 13), (13, 17), (17, 0),      # Wrist
    (5, 6), (6, 7), (7, 8),                          # Index
    (9, 10), (10, 11), (11, 12),                     # Middle
    (13, 14), (14, 15), (15, 16),                    # Ring
    (17, 18), (18, 19), (19, 20)                     # Pinky
]

# ------------------ Variables ------------------
cap = cv2.VideoCapture(0)
p_time = 0


# ------------------ Main Loop ------------------
while True:
    success, img = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)

    timestamp = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, timestamp)

    finger_count = 0
    finger = []

    if result.hand_landmarks:
        for hands in result.hand_landmarks:
            h, w, _ = img.shape
            lm_list = []

            for lm in hands:
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # Draw connections
            for start, end in HAND_CONNECTIONS:
                cv2.line(img, lm_list[start], lm_list[end], (0, 255, 0), 2)

            # Draw landmarks
            for x, y in lm_list:
                cv2.circle(img, (x, y), 4, (0, 0, 255), cv2.FILLED)

            x1, y1 = lm_list[8][0], lm_list[8][1]
            x2, y2 = lm_list[12][0], lm_list[12][1]
            x3, y3 = (x2+x1)//2, (y2+y1)//2

            x4 = np.interp(x1, (0, wcam))

            if lm_list[4][0] < lm_list[3][0]:
                finger_count += 1
                finger.append(1)
            else:
                finger.append(0)

            tips = [8, 12, 16, 20]
            pips = [6, 10, 14, 18]

            for tip, pip in zip(tips, pips):
                if lm_list[tip][1] < lm_list[pip][1]:
                    finger_count += 1
                    finger.append(1)
                else:
                    finger.append(0)
            
            # print(finger_count)
            # print(finger)

            if finger[1] and finger[2]:
                print("Left Click")
                cv2.circle(img, (x3, y3), 15, (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "Left Click", (20, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)

            if finger[1] and not finger[2]:
                print("Move")
                autopy.mouse.move(x1, y1)
                cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "Move the curser", (20, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)

    # ------------------ FPS ------------------
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(img,
                f'FPS: {int(fps)}',
                (20, 40),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 0),
                3)

    cv2.imshow("Hand Control", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ------------------ Cleanup ------------------
cap.release()
cv2.destroyAllWindows()
