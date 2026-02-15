#!/usr/bin/env python3
import argparse
import socket
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return v
    return v / norm


def rotation_matrix_to_quaternion(r):
    # r is 3x3
    m00, m01, m02 = r[0, 0], r[0, 1], r[0, 2]
    m10, m11, m12 = r[1, 0], r[1, 1], r[1, 2]
    m20, m21, m22 = r[2, 0], r[2, 1], r[2, 2]

    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m21 - m12) * s
        qy = (m02 - m20) * s
        qz = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    return qw, qx, qy, qz


def hand_pose_from_world_landmarks(world_landmarks):
    # landmarks: list of 21 points with x,y,z in meters
    # Use wrist (0), index_mcp (5), pinky_mcp (17)
    wrist = np.array([world_landmarks[0].x, world_landmarks[0].y, world_landmarks[0].z])
    index_mcp = np.array([world_landmarks[5].x, world_landmarks[5].y, world_landmarks[5].z])
    pinky_mcp = np.array([world_landmarks[17].x, world_landmarks[17].y, world_landmarks[17].z])

    x_axis = normalize(index_mcp - wrist)
    y_axis = normalize(pinky_mcp - wrist)
    z_axis = normalize(np.cross(x_axis, y_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))

    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    qw, qx, qy, qz = rotation_matrix_to_quaternion(rot)

    return wrist, (qw, qx, qy, qz)


def send_pose(sock, addr, hand_tag, position, quat, valid, timestamp, trigger_value):
    px, py, pz = position
    qw, qx, qy, qz = quat
    msg = f"{hand_tag} {px:.6f} {py:.6f} {pz:.6f} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {1 if valid else 0} {timestamp:.6f} {trigger_value:.3f}"
    sock.sendto(msg.encode("utf-8"), addr)


def compute_pinch_trigger(world_landmarks, was_pressed, press_threshold, release_threshold):
    thumb_tip = np.array([world_landmarks[4].x, world_landmarks[4].y, world_landmarks[4].z])
    index_tip = np.array([world_landmarks[8].x, world_landmarks[8].y, world_landmarks[8].z])
    distance = np.linalg.norm(thumb_tip - index_tip)

    if was_pressed:
        return distance < release_threshold
    return distance < press_threshold


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Hand Tracker -> UDP (Tasks API)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--udp-ip", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=7000)
    parser.add_argument("--model", default="hand_landmarker.task")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--pinch-press-threshold", type=float, default=0.03)
    parser.add_argument("--pinch-release-threshold", type=float, default=0.04)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.udp_ip, args.udp_port)

    base_options = mp_python.BaseOptions(model_asset_path=args.model)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Erro ao abrir camera")
        return

    last_time = time.time()
    fps = 0.0
    trigger_state = {"L": False, "R": False}

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.time() * 1000)
        results = detector.detect_for_video(mp_image, timestamp_ms)

        left_sent = False
        right_sent = False
        timestamp = time.time()

        left_conf = None
        right_conf = None

        if results.hand_world_landmarks and results.handedness:
            for hand_landmarks, handedness in zip(results.hand_world_landmarks,
                                                  results.handedness):
                label = handedness[0].category_name.lower()
                score = handedness[0].score
                hand_tag = "L" if label == "left" else "R"

                pos, quat = hand_pose_from_world_landmarks(hand_landmarks)
                trigger_state[hand_tag] = compute_pinch_trigger(
                    hand_landmarks,
                    trigger_state[hand_tag],
                    args.pinch_press_threshold,
                    args.pinch_release_threshold,
                )
                trigger_value = 1.0 if trigger_state[hand_tag] else 0.0
                send_pose(sock, addr, hand_tag, pos, quat, True, timestamp, trigger_value)

                if hand_tag == "L":
                    left_sent = True
                    left_conf = score
                else:
                    right_sent = True
                    right_conf = score

        if not args.no_preview and results.hand_landmarks:
            height, width = frame.shape[:2]
            for landmarks in results.hand_landmarks:
                for i, lm in enumerate(landmarks):
                    cx = int(lm.x * width)
                    cy = int(lm.y * height)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
                    cv2.putText(frame, str(i), (cx + 4, cy - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                for a, b in HAND_CONNECTIONS:
                    x1 = int(landmarks[a].x * width)
                    y1 = int(landmarks[a].y * height)
                    x2 = int(landmarks[b].x * width)
                    y2 = int(landmarks[b].y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

        if not left_sent:
            trigger_state["L"] = False
            send_pose(sock, addr, "L", (0, 0, 0), (1, 0, 0, 0), False, timestamp, 0.0)
        if not right_sent:
            trigger_state["R"] = False
            send_pose(sock, addr, "R", (0, 0, 0), (1, 0, 0, 0), False, timestamp, 0.0)

        if not args.no_preview:
            cv2.putText(frame, "MediaPipe Hands", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            left_text = f"L: {left_conf:.2f}" if left_conf is not None else "L: --"
            right_text = f"R: {right_conf:.2f}" if right_conf is not None else "R: --"
            conf_text = f"{left_text}  {right_text}"
            text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            x = max(10, frame.shape[1] - text_size[0] - 10)
            cv2.putText(frame, conf_text, (x, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Hand Tracker", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
