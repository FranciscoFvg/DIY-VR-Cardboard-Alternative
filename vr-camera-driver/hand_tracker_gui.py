#!/usr/bin/env python3
import argparse
import socket
import time
import threading
import json
import os
import tkinter as tk
from tkinter import ttk

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

CONFIG_FILE = "hand_tracker_config.json"


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


def normalize_quaternion(q):
    q = np.array(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def slerp_quaternion(q0, q1, t):
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = float(np.clip(dot, -1.0, 1.0))

    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return normalize_quaternion(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    s0 = np.sin(theta_0 - theta) / np.sin(theta_0)
    s1 = np.sin(theta) / np.sin(theta_0)
    result = (s0 * q0) + (s1 * q1)
    return normalize_quaternion(result)


def hand_pose_from_world_landmarks(world_landmarks):
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
        return distance < release_threshold, distance
    return distance < press_threshold, distance


def apply_limb_offsets(position, hand_tag, shoulder_width, upper_arm_length, forearm_length, hand_movement_offset):
    """Aplica offsets de membro à posição da mão
    
    - shoulder_width: separação horizontal entre controles esquerdo e direito
    - upper_arm_length + forearm_length: distância para frente (profundidade em relação ao olhar)
    - hand_movement_offset: movimento dinâmico da mão
    """
    pos = np.array(position, dtype=float)
    
    # Shoulder width: separação horizontal entre mãos
    if hand_tag == "L":
        pos[0] -= shoulder_width / 2.0
    else:  # Right
        pos[0] += shoulder_width / 2.0
    
    # Upper arm + forearm: distância total para frente (eixo Z)
    total_arm_offset = upper_arm_length + forearm_length
    pos[2] += total_arm_offset
    
    # Adicionar movimento dinâmico da mão
    pos += hand_movement_offset
    
    return tuple(pos)


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def parse_float_or_default(value_text, default_value):
    try:
        return float(value_text)
    except (TypeError, ValueError):
        return default_value


class ConfigTabsPanel:
    def __init__(self, config, on_calibrate_request):
        self.config = config
        self.on_calibrate_request = on_calibrate_request
        self.enabled = False

        try:
            self.root = tk.Tk()
            self.root.title("Hand Tracker - Config")
            self.root.geometry("420x320")
            self.root.resizable(False, False)

            tabs = ttk.Notebook(self.root)
            tabs.pack(fill="both", expand=True, padx=8, pady=8)

            with self.config.lock:
                press_threshold = self.config.pinch_press_threshold
                release_threshold = self.config.pinch_release_threshold
                position_smoothing = self.config.position_smoothing
                rotation_smoothing = self.config.rotation_smoothing
                shoulder_width = self.config.shoulder_width
                upper_arm_length = self.config.upper_arm_length
                forearm_length = self.config.forearm_length
                hand_movement_scale = self.config.hand_movement_scale

            self.press_var = tk.StringVar(value=f"{press_threshold:.3f}")
            self.release_var = tk.StringVar(value=f"{release_threshold:.3f}")
            self.pos_smooth_var = tk.StringVar(value=f"{position_smoothing:.2f}")
            self.rot_smooth_var = tk.StringVar(value=f"{rotation_smoothing:.2f}")
            self.shoulder_var = tk.StringVar(value=f"{shoulder_width:.2f}")
            self.upper_arm_var = tk.StringVar(value=f"{upper_arm_length:.2f}")
            self.forearm_var = tk.StringVar(value=f"{forearm_length:.2f}")
            self.movement_scale_var = tk.StringVar(value=f"{hand_movement_scale:.2f}")

            pinch_tab = ttk.Frame(tabs)
            smoothing_tab = ttk.Frame(tabs)
            offsets_tab = ttk.Frame(tabs)

            tabs.add(pinch_tab, text="Pinch")
            tabs.add(smoothing_tab, text="Smoothing")
            tabs.add(offsets_tab, text="Offsets")

            self._add_number_input(pinch_tab, "Press Threshold", self.press_var, 0)
            self._add_number_input(pinch_tab, "Release Threshold", self.release_var, 1)

            self._add_number_input(smoothing_tab, "Position Smoothing (0-0.95)", self.pos_smooth_var, 0)
            self._add_number_input(smoothing_tab, "Rotation Smoothing (0-0.95)", self.rot_smooth_var, 1)

            self._add_number_input(offsets_tab, "Shoulder Width (-0.5..0.5)", self.shoulder_var, 0)
            self._add_number_input(offsets_tab, "Upper Arm Length (-0.5..0.5)", self.upper_arm_var, 1)
            self._add_number_input(offsets_tab, "Forearm Length (-0.5..0.5)", self.forearm_var, 2)
            self._add_number_input(offsets_tab, "Hand Movement Scale (0..2)", self.movement_scale_var, 3)

            actions = ttk.Frame(self.root)
            actions.pack(fill="x", padx=8, pady=(0, 8))

            ttk.Button(actions, text="Aplicar", command=self.apply_values).pack(side="left")
            ttk.Button(actions, text="Calibrar Mãos", command=self.on_calibrate_request).pack(side="left", padx=(8, 0))

            self.status_var = tk.StringVar(value="Edite os números e clique em Aplicar")
            ttk.Label(actions, textvariable=self.status_var).pack(side="left", padx=(12, 0))

            self.enabled = True
        except Exception as exc:
            print(f"Falha ao iniciar painel de configuração: {exc}")
            self.enabled = False

    def _add_number_input(self, parent, label_text, variable, row):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", padx=10, pady=8)
        entry = ttk.Entry(parent, textvariable=variable, width=12)
        entry.grid(row=row, column=1, sticky="e", padx=10, pady=8)
        entry.bind("<Return>", lambda _event: self.apply_values())

    def apply_values(self):
        with self.config.lock:
            self.config.pinch_press_threshold = clamp(
                parse_float_or_default(self.press_var.get(), self.config.pinch_press_threshold), 0.0, 0.2
            )
            self.config.pinch_release_threshold = clamp(
                parse_float_or_default(self.release_var.get(), self.config.pinch_release_threshold), 0.0, 0.2
            )
            self.config.position_smoothing = clamp(
                parse_float_or_default(self.pos_smooth_var.get(), self.config.position_smoothing), 0.0, 0.95
            )
            self.config.rotation_smoothing = clamp(
                parse_float_or_default(self.rot_smooth_var.get(), self.config.rotation_smoothing), 0.0, 0.95
            )
            self.config.shoulder_width = clamp(
                parse_float_or_default(self.shoulder_var.get(), self.config.shoulder_width), -0.5, 0.5
            )
            self.config.upper_arm_length = clamp(
                parse_float_or_default(self.upper_arm_var.get(), self.config.upper_arm_length), -0.5, 0.5
            )
            self.config.forearm_length = clamp(
                parse_float_or_default(self.forearm_var.get(), self.config.forearm_length), -0.5, 0.5
            )
            self.config.hand_movement_scale = clamp(
                parse_float_or_default(self.movement_scale_var.get(), self.config.hand_movement_scale), 0.0, 2.0
            )

            self.press_var.set(f"{self.config.pinch_press_threshold:.3f}")
            self.release_var.set(f"{self.config.pinch_release_threshold:.3f}")
            self.pos_smooth_var.set(f"{self.config.position_smoothing:.2f}")
            self.rot_smooth_var.set(f"{self.config.rotation_smoothing:.2f}")
            self.shoulder_var.set(f"{self.config.shoulder_width:.2f}")
            self.upper_arm_var.set(f"{self.config.upper_arm_length:.2f}")
            self.forearm_var.set(f"{self.config.forearm_length:.2f}")
            self.movement_scale_var.set(f"{self.config.hand_movement_scale:.2f}")

        self.config.save_to_file()
        self.status_var.set("Config aplicada e salva")

    def update_events(self):
        if not self.enabled:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.enabled = False

    def close(self):
        if not self.enabled:
            return
        try:
            self.root.destroy()
        except tk.TclError:
            pass


class ConfigState:
    def __init__(self, press_threshold, release_threshold):
        self.lock = threading.Lock()
        self.pinch_press_threshold = press_threshold
        self.pinch_release_threshold = release_threshold
        # Offsets de posição (em metros)
        self.shoulder_width = 0.0
        self.upper_arm_length = 0.0
        self.forearm_length = 0.0
        # Posição neutra (calibração)
        self.neutral_hand_pos_x = 0.0
        self.neutral_hand_pos_y = 0.0
        self.neutral_hand_pos_z = 0.0
        # Escala de movimento (sensibilidade)
        self.hand_movement_scale = 1.0  # 1x = movimento direto
        # Suavização para reduzir tremor
        self.position_smoothing = 0.70
        self.rotation_smoothing = 0.70

    def save_to_file(self):
        with self.lock:
            config_data = {
                "pinch_press_threshold": self.pinch_press_threshold,
                "pinch_release_threshold": self.pinch_release_threshold,
                "shoulder_width": self.shoulder_width,
                "upper_arm_length": self.upper_arm_length,
                "forearm_length": self.forearm_length,
                "neutral_hand_pos_x": self.neutral_hand_pos_x,
                "neutral_hand_pos_y": self.neutral_hand_pos_y,
                "neutral_hand_pos_z": self.neutral_hand_pos_z,
                "hand_movement_scale": self.hand_movement_scale,
                "position_smoothing": self.position_smoothing,
                "rotation_smoothing": self.rotation_smoothing,
            }
            try:
                with open(CONFIG_FILE, "w") as f:
                    json.dump(config_data, f, indent=2)
            except Exception as e:
                print(f"Erro ao salvar config: {e}")

    def load_from_file(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    config_data = json.load(f)
                    with self.lock:
                        self.pinch_press_threshold = config_data.get("pinch_press_threshold", self.pinch_press_threshold)
                        self.pinch_release_threshold = config_data.get("pinch_release_threshold", self.pinch_release_threshold)
                        self.shoulder_width = config_data.get("shoulder_width", self.shoulder_width)
                        self.upper_arm_length = config_data.get("upper_arm_length", self.upper_arm_length)
                        self.forearm_length = config_data.get("forearm_length", self.forearm_length)
                        self.neutral_hand_pos_x = config_data.get("neutral_hand_pos_x", self.neutral_hand_pos_x)
                        self.neutral_hand_pos_y = config_data.get("neutral_hand_pos_y", self.neutral_hand_pos_y)
                        self.neutral_hand_pos_z = config_data.get("neutral_hand_pos_z", self.neutral_hand_pos_z)
                        self.hand_movement_scale = config_data.get("hand_movement_scale", self.hand_movement_scale)
                        self.position_smoothing = config_data.get("position_smoothing", self.position_smoothing)
                        self.rotation_smoothing = config_data.get("rotation_smoothing", self.rotation_smoothing)
                    print(f"Config carregada de {CONFIG_FILE}")
            except Exception as e:
                print(f"Erro ao carregar config: {e}")


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Hand Tracker -> UDP (Tasks API) com GUI")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--udp-ip", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=7000)
    parser.add_argument("--model", default="hand_landmarker.task")
    parser.add_argument("--pinch-press-threshold", type=float, default=0.03)
    parser.add_argument("--pinch-release-threshold", type=float, default=0.04)
    args = parser.parse_args()

    config = ConfigState(args.pinch_press_threshold, args.pinch_release_threshold)
    config.load_from_file()  # Carregar config anterior se existir

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
    trigger_distance = {"L": 0.0, "R": 0.0}
    
    # Rastreamento de movimento de mão para calibração
    hand_neutral_pos = {"L": None, "R": None}  # Posição de calibração (wrist)
    hand_current_pos = {"L": None, "R": None}  # Posição atual (wrist)
    filtered_pos = {"L": None, "R": None}
    filtered_rot = {"L": None, "R": None}

    calibrate_requested = False

    def request_calibration():
        nonlocal calibrate_requested
        calibrate_requested = True

    config_panel = ConfigTabsPanel(config, request_calibration)
    if config_panel.enabled:
        print("Painel de configuração aberto com abas e inputs numéricos")
    else:
        print("Painel de configuração indisponível; use o arquivo hand_tracker_config.json")

    cv2.namedWindow("Hand Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Tracker", 1280, 720)

    while True:
        config_panel.update_events()

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

        with config.lock:
            press_th = config.pinch_press_threshold
            release_th = config.pinch_release_threshold
            shoulder_w = config.shoulder_width
            upper_arm = config.upper_arm_length
            forearm = config.forearm_length
            movement_scale = config.hand_movement_scale
            pos_smoothing = config.position_smoothing
            rot_smoothing = config.rotation_smoothing

        if results.hand_world_landmarks and results.handedness:
            for hand_landmarks, handedness in zip(results.hand_world_landmarks,
                                                  results.handedness):
                label = handedness[0].category_name.lower()
                score = handedness[0].score
                hand_tag = "L" if label == "left" else "R"

                pos, quat = hand_pose_from_world_landmarks(hand_landmarks)
                
                # Armazenar posição atual da mão (wrist)
                hand_current_pos[hand_tag] = np.array(pos)
                
                # Inicializar posição neutra se não existir
                if hand_neutral_pos[hand_tag] is None:
                    hand_neutral_pos[hand_tag] = np.array(pos)
                
                # Calcular movimento dinâmico (diferença em relação à posição neutra)
                hand_movement = (hand_current_pos[hand_tag] - hand_neutral_pos[hand_tag]) * movement_scale
                
                # Aplicar offsets de membro
                pos = apply_limb_offsets(pos, hand_tag, shoulder_w, upper_arm, forearm, hand_movement)

                raw_pos = np.array(pos, dtype=float)
                raw_quat = np.array(quat, dtype=float)

                if filtered_pos[hand_tag] is None:
                    filtered_pos[hand_tag] = raw_pos.copy()
                    filtered_rot[hand_tag] = normalize_quaternion(raw_quat)
                else:
                    filtered_pos[hand_tag] = (
                        pos_smoothing * filtered_pos[hand_tag]
                        + (1.0 - pos_smoothing) * raw_pos
                    )
                    filtered_rot[hand_tag] = slerp_quaternion(
                        filtered_rot[hand_tag],
                        raw_quat,
                        1.0 - rot_smoothing,
                    )

                pos = tuple(filtered_pos[hand_tag])
                quat = tuple(filtered_rot[hand_tag])
                
                pressed, distance = compute_pinch_trigger(
                    hand_landmarks,
                    trigger_state[hand_tag],
                    press_th,
                    release_th,
                )
                trigger_state[hand_tag] = pressed
                trigger_distance[hand_tag] = distance
                trigger_value = 1.0 if pressed else 0.0
                send_pose(sock, addr, hand_tag, pos, quat, True, timestamp, trigger_value)

                if hand_tag == "L":
                    left_sent = True
                    left_conf = score
                else:
                    right_sent = True
                    right_conf = score

        if results.hand_landmarks:
            height, width = frame.shape[:2]
            for landmarks in results.hand_landmarks:
                for i, lm in enumerate(landmarks):
                    cx = int(lm.x * width)
                    cy = int(lm.y * height)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
                    if i in [4, 8]:
                        cv2.putText(frame, str(i), (cx + 4, cy - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                for a, b in HAND_CONNECTIONS:
                    x1 = int(landmarks[a].x * width)
                    y1 = int(landmarks[a].y * height)
                    x2 = int(landmarks[b].x * width)
                    y2 = int(landmarks[b].y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

        if not left_sent:
            trigger_state["L"] = False
            trigger_distance["L"] = 0.0
            send_pose(sock, addr, "L", (0, 0, 0), (1, 0, 0, 0), False, timestamp, 0.0)
        if not right_sent:
            trigger_state["R"] = False
            trigger_distance["R"] = 0.0
            send_pose(sock, addr, "R", (0, 0, 0), (1, 0, 0, 0), False, timestamp, 0.0)

        cv2.putText(frame, "MediaPipe Hands - Config & Smoothing", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        left_text = f"L: {left_conf:.2f}" if left_conf is not None else "L: --"
        right_text = f"R: {right_conf:.2f}" if right_conf is not None else "R: --"
        conf_text = f"{left_text}  {right_text}"
        text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        x = max(10, frame.shape[1] - text_size[0] - 10)
        cv2.putText(frame, conf_text, (x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        trigger_info = f"L_Trig: {int(trigger_state['L'])} (d={trigger_distance['L']:.3f})  R_Trig: {int(trigger_state['R'])} (d={trigger_distance['R']:.3f})"
        cv2.putText(frame, trigger_info, (10, frame.shape[0] - 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        with config.lock:
            threshold_info = f"Press: {config.pinch_press_threshold:.3f}  Release: {config.pinch_release_threshold:.3f}"
            offset_info = f"Shoulder Sep: {config.shoulder_width:+.2f}m | Forward: {config.upper_arm_length + config.forearm_length:+.2f}m"
            movement_info = f"Move Scale: {config.hand_movement_scale:.2f}x | Pos Smooth: {config.position_smoothing:.2f} | Rot Smooth: {config.rotation_smoothing:.2f}"
        cv2.putText(frame, threshold_info, (10, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, offset_info, (10, frame.shape[0] - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 2)
        cv2.putText(frame, movement_info, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 0), 2)
        cv2.putText(frame, "Ajustes no painel com abas (inputs numericos)", (10, frame.shape[0] - 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(frame, "Pressione C para calibrar | ESC para sair", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        cv2.imshow("Hand Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('c') or key == ord('C'):  # Calibração
            calibrate_requested = True

        if calibrate_requested:
            hand_neutral_pos["L"] = hand_current_pos["L"].copy() if hand_current_pos["L"] is not None else None
            hand_neutral_pos["R"] = hand_current_pos["R"].copy() if hand_current_pos["R"] is not None else None
            print("Calibração de posição de mão realizada!")
            print(f"  L: {hand_neutral_pos['L']}")
            print(f"  R: {hand_neutral_pos['R']}")
            calibrate_requested = False

    config_panel.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
