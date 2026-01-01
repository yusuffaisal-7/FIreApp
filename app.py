
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase
import av

# Set page configuration with a custom title and icon
st.set_page_config(
    page_title="FireGuard AI | Foxmen Studio",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS & AESTHETICS ---
st.markdown("""
<style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Syncopate:wght@400;700&display=swap');

    /* GLOBAL VARIABLES */
    :root {
        --primary-color: #FF3D00; /* Deep Orange Fire */
        --secondary-color: #00E5FF; /* Cyan Cyber */
        --bg-color: #050505; /* Deep Black */
        --card-bg: rgba(20, 20, 20, 0.6);
        --text-color: #E0E0E0;
        --sidebar-bg: #0a0a0a;
        --glass-border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* GLOBAL STYLES */
    .stApp {
        background-color: var(--bg-color);
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(255, 61, 0, 0.1) 0%, transparent 50%),
            linear-gradient(rgba(0, 0, 0, 0.7) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 0, 0, 0.7) 1px, transparent 1px);
        background-size: 100% 100%, 40px 40px, 40px 40px;
        font-family: 'Rajdhani', sans-serif;
        color: var(--text-color);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Syncopate', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: white !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
    }
    
    h1 {
        background: linear-gradient(90deg, #FF3D00, #FFEA00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    p, li, label, .stMarkdown {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.1rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    /* SIDEBAR STYLING */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: var(--glass-border);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] h1 {
        font-size: 1.5rem;
        background: none;
        -webkit-text-fill-color: white;
    }

    /* BUTTONS */
    .stButton > button {
        background: transparent;
        border: 1px solid var(--primary-color);
        color: var(--primary-color);
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        border-radius: 0px;
        box-shadow: 0 0 10px rgba(255, 61, 0, 0.1);
    }

    .stButton > button:hover {
        background: var(--primary-color);
        color: #000;
        box-shadow: 0 0 20px rgba(255, 61, 0, 0.6);
        transform: translateY(-2px);
    }

    /* CARDS & CONTAINERS */
    .css-1r6slb0, .stFileUploader {
        background: var(--card-bg);
        border: var(--glass-border);
        border-radius: 8px;
        padding: 20px;
        backdrop-filter: blur(5px);
    }

    /* IMAGE BORDER */
    img {
        border: 1px solid var(--secondary-color);
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.2);
    }

    /* FOOTER */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(10px);
        color: #888;
        padding: 10px 0;
        text-align: center;
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.9rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        z-index: 999;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
    }
    
    .footer a {
        color: var(--secondary-color);
        text-decoration: none;
        font-weight: 600;
        transition: color 0.3s;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .footer a:hover {
        color: white;
        text-shadow: 0 0 8px var(--secondary-color);
    }
    
    .spark {
        color: var(--primary-color);
        margin: 0 5px;
    }

    /* ANIMATIONS */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 61, 0, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 61, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 61, 0, 0); }
    }
    
    .status-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 4px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-safe {
        background: rgba(0, 229, 255, 0.1);
        color: var(--secondary-color);
        border: 1px solid var(--secondary-color);
    }
    
    .status-danger {
        background: rgba(255, 61, 0, 0.2);
        color: var(--primary-color);
        border: 1px solid var(--primary-color);
        animation: pulse 2s infinite;
    }

    /* --- MOBILE RESPONSIVENESS --- */
    @media only screen and (max-width: 768px) {
        /* Typography Scale */
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.4rem !important; }
        h3 { font-size: 1.2rem !important; }
        p, .stMarkdown { font-size: 0.9rem !important; }
        
        /* Layout Adjustments */
        [data-testid="stSidebar"] {
            width: 100% !important; /* Full width sidebar on mobile open */
        }
        
        /* Footer Stack */
        .footer {
            flex-direction: column;
            gap: 5px;
            font-size: 0.7rem;
            padding: 15px 0;
        }
        
        /* Buttons touch-friendly */
        .stButton > button {
            width: 100%;
            padding: 15px 10px;
            font-size: 1rem;
        }
        
        /* Hide complex backgrounds to save battery/perf */
        .stApp {
            background-image: none;
            background-color: var(--bg-color);
        }
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# --- ALGORITHMS ---

def detect_fire_heuristic_conservative(frame):
    """(LESS SENSITIVE) Very conservative fire detection"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 150, 150])
    upper_red1 = np.array([8, 255, 255])
    lower_red2 = np.array([172, 150, 150])
    upper_red2 = np.array([180, 255, 255])
    lower_orange = np.array([8, 150, 150])
    upper_orange = np.array([15, 255, 255])
    fire_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2) + cv2.inRange(hsv, lower_orange, upper_orange)
    fire_pixels = cv2.countNonZero(fire_mask)
    fire_percentage = (fire_pixels / (frame.shape[0] * frame.shape[1])) * 100
    is_fire = fire_percentage > 5 
    return is_fire, fire_percentage

def detect_fire_heuristic_balanced(frame):
    """(BALANCED) Moderate detection"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 80, 80]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 80, 80]), np.array([180, 255, 255])
    lower_orange, upper_orange = np.array([10, 80, 80]), np.array([25, 255, 255])
    lower_yellow, upper_yellow = np.array([25, 80, 80]), np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2) + cv2.inRange(hsv, lower_orange, upper_orange) + cv2.inRange(hsv, lower_yellow, upper_yellow)
    pct = (cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])) * 100
    return pct > 8, pct

def detect_screen_pattern(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / h if h > 0 else 0
            if 1.2 <= aspect <= 2.5:
                region = gray[y:y+h, x:x+w]
                if region.size > 0 and np.std(region) < 30:
                    return True
    return False

def detect_fire_heuristic_smart(frame):
    """(SMART) Distinguishes screens"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower, upper = np.array([0, 80, 80]), np.array([10, 255, 255]) 
    mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([35, 255, 255])) 
    mask += cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
    
    pct = (cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])) * 100
    is_screen = detect_screen_pattern(frame)
    
    threshold = 15 if is_screen else 8
    is_fire = pct > threshold
    return is_fire, pct, is_screen

# --- DEEP LEARNING ---

@st.cache_resource
def load_model_from_path(path):
    if os.path.exists(path):
        try:
            return load_model(path)
        except Exception:
            return None
    return None

def predict_dl_model(model, frame):
    try:
        resized = cv2.resize(frame, (150, 150))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm = rgb.astype(np.float32) / 255.0
        batch = np.expand_dims(norm, axis=0)
        prob = model.predict(batch, verbose=0)[0][0]
        return prob > 0.5, prob
    except:
        return False, 0.0

# --- GLOBAL STATE FOR WEBRTC ---
# We use a global class to hold params because VideoProcessor is instantiated by webrtc_streamer
class ProcessorSettings:
    model_option = "Heuristic (Smart + Anti-Screen)"
    active_model = None

# --- UI VISUALS ---

def draw_hud(frame, is_fire, confidence, label_text=""):
    height, width = frame.shape[:2]
    overlay = frame.copy()
    
    color_safe = (255, 255, 0) # Cyan-ish
    color_danger = (0, 0, 255) # Red
    active_color = color_danger if is_fire else color_safe
    status_text = "WARNING: FIRE DETECTED" if is_fire else "SYSTEM NOMINAL"
    
    # Vignette
    Y, X = np.ogrid[:height, :width]
    center_y, center_x = height / 2, width / 2
    mask = np.sqrt((X - center_x)**2 + (Y - center_y)**2) / np.sqrt(center_x**2 + center_y**2)
    frame = (frame * (1 - 0.3 * mask[:,:,np.newaxis])).astype(np.uint8)

    # Brackets
    bracket_len = 40
    t = 2
    cv2.line(frame, (20, 20), (20+bracket_len, 20), active_color, t)
    cv2.line(frame, (20, 20), (20, 20+bracket_len), active_color, t)
    cv2.line(frame, (width-20, 20), (width-20-bracket_len, 20), active_color, t)
    cv2.line(frame, (width-20, 20), (width-20, 20+bracket_len), active_color, t)
    cv2.line(frame, (20, height-20), (20+bracket_len, height-20), active_color, t)
    cv2.line(frame, (20, height-20), (20, height-20-bracket_len), active_color, t)
    cv2.line(frame, (width-20, height-20), (width-20-bracket_len, height-20), active_color, t)
    cv2.line(frame, (width-20, height-20), (width-20, height-20-bracket_len), active_color, t)

    # Alert Overlay
    if is_fire:
        cv2.rectangle(overlay, (0, 0), (width, height), active_color, -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    # Text Badge
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(status_text, font, 0.6, 1)
    cx = width // 2
    cv2.rectangle(frame, (cx - tw//2 - 10, 10), (cx + tw//2 + 10, 10 + th + 20), (0,0,0), -1)
    cv2.rectangle(frame, (cx - tw//2 - 10, 10), (cx + tw//2 + 10, 10 + th + 20), active_color, 1)
    cv2.putText(frame, status_text, (cx - tw//2, 10 + th + 10), font, 0.6, active_color, 1, cv2.LINE_AA)
    
    # Info
    info = f"CONF: {confidence:.1f}% | {label_text}"
    cv2.putText(frame, info, (30, height - 30), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Bar
    bw = 150
    filled = int(bw * (confidence / 100))
    cv2.rectangle(frame, (30, height - 25), (30 + bw, height - 15), (50, 50, 50), -1)
    cv2.rectangle(frame, (30, height - 25), (30 + filled, height - 15), active_color, -1)

    return frame

# --- WEBRTC PROCESSOR ---
class FireDetectionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror for UX
        img = cv2.flip(img, 1)
        
        # Access global settings (simple way for Streamlit script model)
        # Note: In a robust multi-user prod app we might pass args via factory
        model_opt = ProcessorSettings.model_option
        act_model = ProcessorSettings.active_model
        
        is_fire, conf = False, 0.0
        label = ""
        
        if "Deep Learning" in model_opt and act_model:
            is_fire, raw = predict_dl_model(act_model, img)
            conf = raw * 100
            label = "CNN"
        elif "Smart" in model_opt:
            is_fire, conf, is_screen = detect_fire_heuristic_smart(img)
            label = "SMART" + (" (SCREEN)" if is_screen else "")
        elif "Balanced" in model_opt:
            is_fire, conf = detect_fire_heuristic_balanced(img)
            label = "BALANCED"
        else:
            is_fire, conf = detect_fire_heuristic_conservative(img)
            label = "FAST"
            
        img = draw_hud(img, is_fire, conf, label)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- MAIN ---

def main():
    st.sidebar.markdown("""
        <div style='text-align: center; padding-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px;'>
            <h1 style='font-size: 2rem; margin:0;'>FIRE<span style='color: #FF3D00;'>APP</span></h1>
            <p style='font-size: 0.8rem; letter-spacing: 3px; color: #888;'>ADVANCED DETECTION SYSTEM</p>
        </div>
    """, unsafe_allow_html=True)
    
    app_mode = st.sidebar.radio("MODE", ["Dashboard", "Video Analysis", "Live Surveillance"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 AI MODEL SELECTOR")
    
    model_option = st.sidebar.selectbox("Choose Detection Engine:", 
        [
            "Deep Learning (Recent - Fixed)",
            "Deep Learning (Original - Standard)",
            "Heuristic (Smart + Anti-Screen)",
            "Heuristic (Balanced)",
            "Heuristic (Fast/Conservative)"
        ]
    )
    
    # Model Loading Logic
    active_model = None
    if "Deep Learning" in model_option:
        filename = "fire_detection_model_fixed.h5" if "Fixed" in model_option else "fire_detection_model.h5"
        with st.spinner(f"LOADING NEURAL NETWORK: {filename}..."):
            active_model = load_model_from_path(filename)
            if active_model is None:
                st.sidebar.error("❌ Model not found! Switching to Heuristic.")
                model_option = "Heuristic (Smart + Anti-Screen)"

    # Update Global Settings for WebRTC
    ProcessorSettings.model_option = model_option
    ProcessorSettings.active_model = active_model

    if app_mode == "Dashboard":
        st.markdown(f"""
            <div style="padding: 40px; border: 1px solid rgba(255, 61, 0, 0.2); background: linear-gradient(135deg, rgba(20,20,20,0.8), rgba(10,10,10,0.9)); border-radius: 10px; text-align: center; margin-bottom: 30px;">
                <h1 style="font-size: 3rem; margin-bottom: 10px;">FIRE GUARD <span style="color: #FF3D00;">AI</span></h1>
                <p style="font-size: 1.2rem; color: #888; max-width: 600px; margin: 0 auto;">NEXT-GENERATION SURVEILLANCE SUITE</p>
            </div>
        """, unsafe_allow_html=True)
        st.image("https://images.unsplash.com/photo-1542317854-f9596afbd6cb", width="stretch", caption="SYSTEM ONLINE") 

    elif app_mode == "Video Analysis":
        st.markdown("<h3>VIDEO INGEST</h3>", unsafe_allow_html=True)
        uploaded = st.file_uploader("UPLOAD FOOTAGE", type=["mp4", "mov", "avi"])
        
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded.read())
            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (640, 480))
                
                # Inference
                is_fire, conf = False, 0.0
                label = ""
                
                if "Deep Learning" in model_option and active_model:
                    is_fire, raw = predict_dl_model(active_model, frame)
                    conf = raw * 100
                    label = "CNN"
                elif "Smart" in model_option:
                    is_fire, conf, is_screen = detect_fire_heuristic_smart(frame)
                    label = "SMART" + (" (SCREEN)" if is_screen else "")
                elif "Balanced" in model_option:
                    is_fire, conf = detect_fire_heuristic_balanced(frame)
                    label = "BALANCED"
                else:
                    is_fire, conf = detect_fire_heuristic_conservative(frame)
                    label = "FAST"
                    
                frame = draw_hud(frame, is_fire, conf, label)
                st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")
            cap.release()

    elif app_mode == "Live Surveillance":
        st.markdown("<h3>LIVE FEED <span style='color:#00E5FF'>// ONLINE</span></h3>", unsafe_allow_html=True)
        st.info("Initiating WebRTC Stream... Please allow camera access.")
        
        # WebRTC Streamer
        ctx = webrtc_streamer(
            key="fire-detection",
            video_processor_factory=FireDetectionProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    st.markdown("""
        <div class="footer">
            <span>POWERED BY <a href="https://foxmen.studio" target="_blank">FOXMEN STUDIO</a></span>
            <span class="spark">⚡</span>
            <span>CREATED BY <a href="https://github.com/yusuffaisal-7" target="_blank">YOUSUF H FAYSAL</a></span>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
