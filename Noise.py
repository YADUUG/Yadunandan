import cv2
import mediapipe as mp
import time
import streamlit as st
import numpy as np

mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands

# Initialize Face Detection and Hands modules
face_detection = mp_face.FaceDetection(min_detection_confidence=0.2)
hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.2)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

lap_speeds = []

# Initialize variables for smoothing
smooth_factor = 0.5
prev_ix, prev_iy, prev_nx, prev_ny = 0, 0, 0, 0
prev_time = time.time()

st.title("Hand and Face Detection with Lap Speed")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_detection.process(rgb_frame)
    hands_results = hands.process(rgb_frame)

    if hands_results.multi_hand_landmarks:
        for landmarks in hands_results.multi_hand_landmarks:
            index_fingertip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            height, width, _ = frame.shape
            ix, iy = int(index_fingertip.x * width), int(index_fingertip.y * height)

            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    nx = int((bboxC.xmin + bboxC.width / 2) * iw)
                    ny = int((bboxC.ymin + bboxC.height / 2) * ih)

                    # Smooth out the movement
                    ix = int((1 - smooth_factor) * prev_ix + smooth_factor * ix)
                    iy = int((1 - smooth_factor) * prev_iy + smooth_factor * iy)
                    nx = int((1 - smooth_factor) * prev_nx + smooth_factor * nx)
                    ny = int((1 - smooth_factor) * prev_ny + smooth_factor * ny)

                    distance = ((ix - nx) ** 2 + (iy - ny) ** 2) ** 0.5

                    current_time = time.time()
                    lap_speed = distance / (current_time - prev_time) * 1000 if current_time != prev_time else 0
                    lap_speeds.append(lap_speed)
                    prev_time = current_time

                    # Save current coordinates for smoothing
                    prev_ix, prev_iy, prev_nx, prev_ny = ix, iy, nx, ny

                    # Draw line connecting nose and index fingertip
                    cv2.line(frame, (nx, ny), (ix, iy), (0, 255, 0), 2)

    # Draw circle on index fingertip
    if hands_results.multi_hand_landmarks:
        for landmarks in hands_results.multi_hand_landmarks:
            index_fingertip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            height, width, _ = frame.shape
            ix, iy = int(index_fingertip.x * width), int(index_fingertip.y * height)
            cv2.circle(frame, (ix, iy), 10, (255, 0, 0), cv2.FILLED)

    # Draw circle on nose
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            nx = int((bboxC.xmin + bboxC.width / 2) * iw)
            ny = int((bboxC.ymin + bboxC.height / 2) * ih)
            cv2.circle(frame, (nx, ny), 10, (0, 255, 0), cv2.FILLED)

    # Display lap speed in milliseconds
    if lap_speeds:
        st.write(f"Lap Speed: {lap_speeds[-1]:.2f} ms")

    # Display the image in Streamlit
    st.image(frame, channels="BGR")

    if st.button("Stop"):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()

average_speed = np.mean(lap_speeds) if len(lap_speeds) > 0 else 0
st.write(f"Average Speed: {average_speed:.2f} ms per lap")
