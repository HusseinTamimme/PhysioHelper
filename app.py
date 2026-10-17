import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="AI PT Coach", layout="wide")

# ==========================================
# 1. BIOMECHANICAL THRESHOLDS & COLORS
# ==========================================
KNEE_MIN, KNEE_MAX = 70, 120
ELBOW_MIN, SPINE_MIN = 145, 150     
SPAN_SETUP, SPAN_HOLD = 45, 150     
ARM_HEIGHT_MIN, ARM_HEIGHT_MAX = 75, 110 
HIP_TWIST_MAX = 0.15 

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_ORANGE = (255, 165, 0)
COLOR_YELLOW = (255, 255, 0)

# ==========================================
# 2. MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource
def load_model():
    MODEL_PATH = "pose_landmarker_heavy.task"
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
            MODEL_PATH
        )
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, # Changed to IMAGE mode for WebRTC frames
        num_poses=1
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)

landmarker = load_model()

# ==========================================
# 3. MATH ENGINE & LOGIC
# ==========================================
def calculate_3d_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_landmark(landmarks, index):
    lm = landmarks[index]
    return [lm.x, lm.y, lm.z]

def evaluate_full_body(world_landmarks):
    l_sh, r_sh = get_landmark(world_landmarks, 11), get_landmark(world_landmarks, 12)
    l_elb, r_elb = get_landmark(world_landmarks, 13), get_landmark(world_landmarks, 14)
    l_wr, r_wr = get_landmark(world_landmarks, 15), get_landmark(world_landmarks, 16)
    l_hip, r_hip = get_landmark(world_landmarks, 23), get_landmark(world_landmarks, 24)
    l_kn, r_kn = get_landmark(world_landmarks, 25), get_landmark(world_landmarks, 26)
    l_ank, r_ank = get_landmark(world_landmarks, 27), get_landmark(world_landmarks, 28)
    
    mid_sh = [(l_sh[0]+r_sh[0])/2, (l_sh[1]+r_sh[1])/2, (l_sh[2]+r_sh[2])/2]

    ang_l_elbow = calculate_3d_angle(l_sh, l_elb, l_wr)
    ang_r_elbow = calculate_3d_angle(r_sh, r_elb, r_wr)
    ang_l_knee = calculate_3d_angle(l_hip, l_kn, l_ank)
    ang_r_knee = calculate_3d_angle(r_hip, r_kn, r_ank)
    ang_span = calculate_3d_angle(l_wr, mid_sh, r_wr)
    
    ang_spine = (calculate_3d_angle(l_sh, l_hip, l_kn) + calculate_3d_angle(r_sh, r_hip, r_kn)) / 2.0
    ang_l_elev = calculate_3d_angle(l_hip, l_sh, l_wr)
    ang_r_elev = calculate_3d_angle(r_hip, r_sh, r_wr)
    hip_z_diff = abs(l_hip[2] - r_hip[2])

    colors, mistakes = {}, []
    
    colors['l_arm'] = COLOR_GREEN if (ang_l_elbow >= ELBOW_MIN and ARM_HEIGHT_MIN <= ang_l_elev <= ARM_HEIGHT_MAX) else COLOR_RED
    if colors['l_arm'] == COLOR_RED: mistakes.append("Fix L Arm")

    colors['r_arm'] = COLOR_GREEN if (ang_r_elbow >= ELBOW_MIN and ARM_HEIGHT_MIN <= ang_r_elev <= ARM_HEIGHT_MAX) else COLOR_RED
    if colors['r_arm'] == COLOR_RED: mistakes.append("Fix R Arm")

    colors['l_leg'] = COLOR_GREEN if (KNEE_MIN <= ang_l_knee <= KNEE_MAX) else COLOR_RED
    if colors['l_leg'] == COLOR_RED: mistakes.append("Fix L Knee")

    colors['r_leg'] = COLOR_GREEN if (KNEE_MIN <= ang_r_knee <= KNEE_MAX) else COLOR_RED
    if colors['r_leg'] == COLOR_RED: mistakes.append("Fix R Knee")

    torso_ok = (ang_spine >= SPINE_MIN and hip_z_diff <= HIP_TWIST_MAX)
    colors['torso'] = COLOR_GREEN if torso_ok else (COLOR_ORANGE if ang_spine >= SPINE_MIN else COLOR_RED)
    if not torso_ok: mistakes.append("Fix Torso/Hips")

    return colors, mistakes

def draw_dynamic_skeleton(frame, pixel_landmarks, colors, width, height):
    limb_groups = {
        'l_arm': [(11, 13), (13, 15)], 'r_arm': [(12, 14), (14, 16)],
        'l_leg': [(23, 25), (25, 27)], 'r_leg': [(24, 26), (26, 28)],
        'torso': [(11, 12), (11, 23), (12, 24), (23, 24)]
    }
    for group_name, connections in limb_groups.items():
        color = colors.get(group_name, COLOR_WHITE)
        for p1, p2 in connections:
            x1, y1 = int(pixel_landmarks[p1].x * width), int(pixel_landmarks[p1].y * height)
            x2, y2 = int(pixel_landmarks[p2].x * width), int(pixel_landmarks[p2].y * height)
            cv2.line(frame, (x1, y1), (x2, y2), color, 6)
            cv2.circle(frame, (x1, y1), 8, color, -1)
            cv2.circle(frame, (x2, y2), 8, color, -1)

# ==========================================
# 4. WEBRTC FRAME PROCESSOR
# ==========================================
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1) # Mirror image
    height, width, _ = img.shape
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_image)
    
    if result.pose_world_landmarks and result.pose_landmarks:
        world = result.pose_world_landmarks[0]
        pixels = result.pose_landmarks[0]
        
        colors, mistakes = evaluate_full_body(world)
        draw_dynamic_skeleton(img, pixels, colors, width, height)
        
        if len(mistakes) > 0:
            cv2.rectangle(img, (10, height - (len(mistakes)*30 + 50)), (300, height - 10), (0, 0, 0), -1)
            cv2.putText(img, "CORRECTIONS:", (20, height - (len(mistakes)*30 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
            for i, mistake in enumerate(mistakes):
                y_pos = height - ((len(mistakes) - i - 1) * 30 + 20)
                cv2.putText(img, f"- {mistake}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
        else:
            cv2.rectangle(img, (10, height - 50), (250, height - 10), (0, 0, 0), -1)
            cv2.putText(img, "FORM PERFECT", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)
            
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 5. STREAMLIT UI
# ==========================================
st.title("🏃 AI Physical Therapy Coach")
st.markdown("### Half-Kneeling Thoracic Rotation (Open Book)")
st.markdown("Ensure your full body is visible in the camera frame.")

# WebRTC configuration for cloud deployment (forces secure connection)
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="pt-coach",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
