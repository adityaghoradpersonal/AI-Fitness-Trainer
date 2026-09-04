import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

# MUST be first Streamlit command
st.set_page_config(page_title='Fitness AI Coach', layout='centered', page_icon="💪")

import cv2
import tempfile
import time

# Lazy imports - don't load TF/mediapipe until needed
_exercise_module = None
def _get_exercise():
    global _exercise_module
    if _exercise_module is None:
        import ExerciseAiTrainer as ex
        _exercise_module = ex
    return _exercise_module

from chatbot import chat_ui

# WebRTC
_WEBRTC_AVAILABLE = False
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    _WEBRTC_AVAILABLE = True
except Exception:
    pass

def _browser_speak(text):
    """Speak text aloud using browser's speechSynthesis via st.components.v1.html.
    Each call uses a unique HTML body so Streamlit always re-renders the iframe."""
    if not text:
        return
    last = st.session_state.get("_last_spoken", "")
    last_time = st.session_state.get("_last_spoken_time", 0)
    now = time.time()
    if text == last and (now - last_time) < 5:
        return
    st.session_state["_last_spoken"] = text
    st.session_state["_last_spoken_time"] = now
    safe = text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
    # Unique timestamp in the HTML body forces Streamlit to create a new iframe
    ts = int(now * 1000)
    html = f"""
    <html><body>
    <div id="t{ts}" style="display:none">{ts}</div>
    <script>
    const synth = window.parent.speechSynthesis || window.speechSynthesis;
    if (synth) {{
        synth.cancel();
        const u = new SpeechSynthesisUtterance('{safe}');
        u.rate = 1.0; u.volume = 1.0; u.lang = 'en-US';
        synth.speak(u);
    }}
    </script>
    </body></html>
    """
    st.components.v1.html(html, height=0)


ASSETS_VIDEOS = ROOT / "assets" / "videos"
DEMO_VIDEO = ASSETS_VIDEOS / "demo_2.mp4"
MODELS_DIR = ROOT / "models"

def _fetch_metered_ice_servers(api_key):
    """Fetch fresh TURN credentials from Metered REST API."""
    import requests as req
    try:
        resp = req.get(
            f"https://fitness.metered.live/api/v1/turn/credentials?apiKey={api_key}",
            timeout=5
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    # Fallback: try the generic endpoint
    try:
        resp = req.get(
            f"https://global.relay.metered.ca/api/v1/turn/credentials?apiKey={api_key}",
            timeout=5
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _get_rtc_configuration():
    """Build RTC config. Uses METERED_API_KEY to fetch fresh TURN credentials."""
    if not _WEBRTC_AVAILABLE:
        return None, False

    api_key = os.environ.get("METERED_API_KEY", "")
    ice_servers = None

    # Method 1: Fetch from Metered API (best - gives fresh rotating credentials)
    if api_key:
        ice_servers = _fetch_metered_ice_servers(api_key)

    if ice_servers:
        return RTCConfiguration({"iceServers": ice_servers}), True

    # Method 2: Manual TURN config from env vars
    turn_url = os.environ.get("TURN_URL", "")
    turn_user = os.environ.get("TURN_USERNAME", "")
    turn_cred = os.environ.get("TURN_CREDENTIAL", "")

    if turn_url and turn_user and turn_cred:
        servers = [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": [turn_url], "username": turn_user, "credential": turn_cred},
            {"urls": [turn_url + "?transport=tcp"], "username": turn_user, "credential": turn_cred},
        ]
        return RTCConfiguration({"iceServers": servers}), True

    # Method 3: STUN only (works locally, NOT on cloud behind NAT)
    return RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}), False


RTC_CONFIGURATION, _TURN_CONFIGURED = _get_rtc_configuration()

def _show_turn_help():
    st.warning("**WebCam needs a TURN server to work on cloud.**")
    st.markdown("""
### Setup (free, 2 minutes):

**Step 1:** Go to [metered.ca/stun-turn](https://www.metered.ca/stun-turn) → Click **"Sign Up Free"**

**Step 2:** After signup, go to Dashboard → Click **"TURN Server"** → Copy your **API Key**

**Step 3:** In **Render Dashboard** → your service → **Environment** tab → add **ONE** variable:

| Key | Value |
|-----|-------|
| `METERED_API_KEY` | *(paste your API key from Step 2)* |

**Step 4:** Click **"Save Changes"** → Render auto-redeploys → WebCam works!

---
**Use Video mode now** - upload any exercise video and the AI will analyze it!
""")

EXERCISE_NAME_MAP = {
    'Bicep Curl': 'barbell biceps curl',
    'Push Up': 'push-up',
    'Squat': 'squat',
    'Shoulder Press': 'shoulder press',
}


def _read_accuracy_from_metrics_file(path):
    if not path.exists():
        return None
    try:
        text = path.read_text()
        m = re.search(r"Test accuracy:\s*([\d.]+)", text)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def _show_model_metrics_in_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Status")
    model_h5 = MODELS_DIR / "final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5"
    model_keras = MODELS_DIR / "final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.keras"
    has_model = model_h5.exists() or model_keras.exists()
    if not has_model:
        st.sidebar.warning("No model found. Run training first.")
        return
    st.sidebar.success("Model ready")
    info_path = MODELS_DIR / "train_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            acc = info.get("test_accuracy")
            if acc:
                st.sidebar.metric("Accuracy", f"{acc * 100:.2f}%")
        except Exception:
            pass
    st.sidebar.markdown("---")


def _webrtc_exercise_mode(exercise_display_name, voice_on=True):
    exercise_canonical = EXERCISE_NAME_MAP.get(exercise_display_name, "push-up")
    from webrtc_processor import WebRTCExerciseProcessor

    ctx = webrtc_streamer(
        key=f"exercise-{exercise_canonical}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=WebRTCExerciseProcessor,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.set_exercise(exercise_canonical)

    if ctx.state.playing and ctx.video_processor:
        # Auto-refresh every 2 seconds so stats and voice update while streaming
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=2000, limit=None, key="exercise_refresh")
        except ImportError:
            pass  # works without it, just won't auto-update stats

        exercise = _get_exercise()
        state = ctx.video_processor.get_state()
        cols = st.columns(4)
        cols[0].metric("Reps", state["counter"])
        cols[1].metric("Calories", f"{state['calories']:.1f}")
        cols[2].metric("Duration", f"{state['duration'] / 60:.1f} min")
        cols[3].metric("Exercise", exercise.canonical_to_display_name(state["exercise"]))
        if state["injury"]:
            st.error(f"**{state['injury']}**")
            if voice_on:
                _browser_speak(state["injury"])
        if state["tip"]:
            st.info(f"**Tip:** {state['tip']}")
            if voice_on:
                _browser_speak(state["tip"])


def _webrtc_auto_classify_mode(voice_on=True):
    from webrtc_processor import WebRTCAutoClassifyProcessor

    ctx = webrtc_streamer(
        key="auto-classify",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=WebRTCAutoClassifyProcessor,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )

    if ctx.state.playing and ctx.video_processor:
        if not ctx.video_processor.model_ready:
            st.error("Model not loaded. Check **models/** directory.")
            return

        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=2000, limit=None, key="autoclassify_refresh")
        except ImportError:
            pass

        state = ctx.video_processor.get_state()
        cols = st.columns(4)
        cols[0].metric("Detected", state["prediction"])
        cols[1].metric("Total Reps", state["total_reps"])
        cols[2].metric("Calories", f"{state['total_calories']}")
        cols[3].metric("Duration", f"{state['duration'] / 60:.1f} min")
        if state["injury"]:
            st.error(f"**{state['injury']}**")
            if voice_on:
                _browser_speak(state["injury"])
        if state["tip"]:
            st.info(f"**Tip:** {state['tip']}")
            if voice_on:
                _browser_speak(state["tip"])
        if state["breakdown"]:
            for name, data in state["breakdown"].items():
                st.write(f"- **{name}**: {data['reps']} reps ({data['calories']} cal)")


def main():
    # Load custom styling if available
    css_path = ROOT / "static" / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # ============================================================
    # HEADER
    # ============================================================
    st.title("AI Fitness Trainer")
    st.caption("Computer Vision • Pose Estimation • BiLSTM Exercise Analysis")

    st.markdown(
        """
        <div style="
            padding: 15px;
            border-radius: 10px;
            background-color: rgba(128,128,128,0.08);
            margin-bottom: 20px;
        ">
            <b>AI Model Demonstration</b><br>
            The system analyzes an exercise video using pose estimation,
            movement features, exercise-specific logic, and repetition counting.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # SIDEBAR — MODEL INFORMATION
    # ============================================================
    _show_model_metrics_in_sidebar()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Demo Settings")

    exercise_name = st.sidebar.selectbox(
        "Exercise in demo video",
        (
            "Bicep Curl",
            "Push Up",
            "Squat",
            "Shoulder Press",
        ),
        index=0,
    )

    voice_enabled = st.sidebar.checkbox(
        "Voice Feedback",
        value=False,
        help="Enable spoken form and injury feedback."
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        """
        **Model Pipeline**

        Video  
        ↓  
        MediaPipe Pose  
        ↓  
        Body Landmarks  
        ↓  
        Feature Extraction  
        ↓  
        Exercise Analysis  
        ↓  
        Rep Counting & Form Feedback
        """
    )

    # ============================================================
    # MODEL INFORMATION
    # ============================================================
    with st.expander("Model Information", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Model", "BiLSTM")

        with col2:
            st.metric("Pose Estimation", "MediaPipe")

        with col3:
            st.metric("Input Sequence", "30 Frames")

        st.markdown(
            """
            **What the AI does**

            - Extracts human body landmarks from each video frame.
            - Converts the pose into movement-related features.
            - Processes a sequence of frames using the trained BiLSTM model.
            - Identifies the exercise being demonstrated.
            - Applies the existing exercise-specific repetition logic.
            - Provides form and safety feedback.
            """
        )

    # ============================================================
    # DEMO VIDEO
    # ============================================================
    st.subheader("Exercise Analysis")

    if not DEMO_VIDEO.exists():
        st.error(
            "Demo video not found. Expected file: "
            "`assets/videos/demo_2.mp4`"
        )
        return

    video_path = str(DEMO_VIDEO)

    st.video(video_path)

    st.caption(
        f"Demo video: `{DEMO_VIDEO.name}`"
    )

    # ============================================================
    # START ANALYSIS
    # ============================================================
    st.markdown("### Run AI Analysis")

    st.write(
        f"The selected exercise is **{exercise_name}**. "
        "Start the analysis to run the existing AI exercise-processing pipeline "
        "on the demo video."
    )

    if st.button(
        "Start AI Analysis",
        type="primary",
        use_container_width=True
    ):
        exercise = _get_exercise()
        exer = exercise.Exercise()

        # Disable voice by default for a clean professor demonstration
        exer.voice.enabled = voice_enabled

        # Open the existing demo video
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            st.error("Unable to open the demo video.")
            return

        # --------------------------------------------------------
        # Use the EXISTING exercise-processing functions.
        # No changes are made to the model or AI logic.
        # --------------------------------------------------------
        if exercise_name == "Bicep Curl":
            exer.bicept_curl(
                cap,
                is_video=True,
                counter=0,
                stage_right=None,
                stage_left=None
            )

        elif exercise_name == "Push Up":
            exer.push_up(
                cap,
                is_video=True,
                counter=0,
                stage=None
            )

        elif exercise_name == "Squat":
            exer.squat(
                cap,
                is_video=True,
                counter=0,
                stage=None
            )

        elif exercise_name == "Shoulder Press":
            exer.shoulder_press(
                cap,
                is_video=True,
                counter=0,
                stage=None
            )

        cap.release()

        st.success("Video analysis completed.")


# ================================================================
# APPLICATION ENTRY POINT
# ================================================================
if __name__ == '__main__':
    main()
