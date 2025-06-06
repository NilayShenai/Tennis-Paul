import streamlit as st
import cv2
import time
import numpy as np
from ultralytics import YOLO
from PIL import Image
import math
import tempfile
from collections import deque
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tennis Paul Beta-1", layout="wide")
st.title("Tennis Paul Beta-1")
st.caption("YOLO and OpenCV based Tennis Ball Analysis Interface")

model = YOLO("paul.pt")
mode = st.sidebar.radio("Choose Mode", ("Image", "Upload Video", "Webcam"))

def draw_trajectory(points, frame, color=(0, 255, 255)):
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], color, 2)
    return frame

def estimate_speed(p1, p2, dt, ppm=80):
    dx = (p2[0] - p1[0]) / ppm
    dy = (p2[1] - p1[1]) / ppm
    dist = math.sqrt(dx**2 + dy**2)
    return dist / dt if dt > 0 else 0

def predict_landing(points):
    if len(points) >= 2:
        (x1, y1), (x2, y2) = points[-2], points[-1]
        dx = x2 - x1
        dy = y2 - y1
        landing_point = (int(x2 + dx * 3), int(y2 + dy * 3))
        return landing_point
    return None

if mode == "Image":
    uploaded_file = st.file_uploader("Upload a tennis image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        frame = np.array(img)
        results = model(frame)[0]
        annotated = results.plot()
        st.image(annotated, caption="Detected Image", use_column_width=True)

elif mode == "Video":
    uploaded_video = st.file_uploader("Upload a video clip", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        analyze = st.checkbox("Run Analysis")
        if analyze:
            cap = cv2.VideoCapture(tfile.name)
            prev_time = time.time()
            prev_pos = None
            trajectory = deque(maxlen=20)
            landing_point = None
            landing_timer = 10
            speeds = []

            col1, col2 = st.columns([2, 1])
            with col1:
                stframe = st.empty()
            with col2:
                graph_speed = st.empty()
                graph_traj = st.empty()

            while cap.isOpened():
                start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=0.4)[0]
                boxes = results.boxes

                if boxes and len(boxes.xyxy) > 0:
                    x1, y1, x2, y2 = map(int, boxes.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

                    if prev_pos is not None:
                        dt = time.time() - prev_time
                        speed = estimate_speed(prev_pos, (cx, cy), dt)
                        speeds.append(speed)
                        cv2.putText(frame, f"Speed: {speed:.2f} m/s", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)

                    prev_pos = (cx, cy)
                    prev_time = time.time()
                    trajectory.append((cx, cy))

                    landing_point = predict_landing(trajectory)
                    landing_timer = 10

                if landing_point and landing_timer > 0:
                    cv2.circle(frame, landing_point, 8, (255, 0, 255), -1)
                    cv2.putText(frame, "Predicted Landing", (landing_point[0] + 10, landing_point[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    landing_timer -= 1

                frame = draw_trajectory(trajectory, frame)

                fps = 1.0 / (time.time() - start)
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

                if speeds:
                    graph_speed.line_chart(speeds)

                if trajectory:
                    xs, ys = zip(*trajectory)
                    fig, ax = plt.subplots()
                    ax.plot(xs, ys, marker='x', color='orange')
                    ax.set_title("Ball Trajectory")
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    graph_traj.pyplot(fig)

            cap.release()

elif mode == "Webcam":
    st.caption("This may or may not work as I don't have a webcam with me, 99 percent doesn't")
    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        prev_time = time.time()
        prev_pos = None
        trajectory = deque(maxlen=10)
        landing_point = None
        landing_timer = 15
        speeds = []

        col1, col2 = st.columns([2, 1])
        with col1:
            stframe = st.empty()
        with col2:
            graph_speed = st.empty()
            graph_traj = st.empty()

        while cap.isOpened():
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam disconnected.")
                break

            results = model(frame, conf=0.4)[0]
            boxes = results.boxes

            if boxes and len(boxes.xyxy) > 0:
                x1, y1, x2, y2 = map(int, boxes.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

                if prev_pos is not None:
                    dt = time.time() - prev_time
                    speed = estimate_speed(prev_pos, (cx, cy), dt)
                    speeds.append(speed)
                    cv2.putText(frame, f"Speed: {speed:.2f} m/s", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)

                prev_pos = (cx, cy)
                prev_time = time.time()
                trajectory.append((cx, cy))

                landing_point = predict_landing(trajectory)
                landing_timer = 10

            if landing_point and landing_timer > 0:
                cv2.circle(frame, landing_point, 8, (255, 0, 255), -1)
                cv2.putText(frame, "Predicted Landing", (landing_point[0] + 10, landing_point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                landing_timer -= 1

            frame = draw_trajectory(trajectory, frame)

            fps = 1.0 / (time.time() - start)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            if speeds:
                graph_speed.line_chart(speeds)

            if trajectory:
                xs, ys = zip(*trajectory)
                fig, ax = plt.subplots()
                ax.plot(xs, ys, marker='x', color='orange')
                ax.set_title("Live Ball Trajectory")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                graph_traj.pyplot(fig)

        cap.release()
