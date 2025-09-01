import json
import streamlit as st
from pathlib import Path
from langgraph_interview_flow import run_stage
from main import ALL_STAGES, create_new_session, load_existing_session, save_checkpoint
from utils.audio_utils import AudioHandler

# Animated waveform HTML for recording indicator
WAVE_HTML = """
<div class='wave'>
  <span></span><span></span><span></span><span></span><span></span>
</div>
<style>
.wave{display:flex;align-items:flex-end;height:20px}
.wave span{display:block;width:4px;height:20px;margin:0 2px;background:#ff4b4b;animation:wave 1s infinite}
.wave span:nth-child(2){animation-delay:0.1s}
.wave span:nth-child(3){animation-delay:0.2s}
.wave span:nth-child(4){animation-delay:0.3s}
.wave span:nth-child(5){animation-delay:0.4s}
@keyframes wave{0%,100%{transform:scaleY(.3)}50%{transform:scaleY(1)}}
</style>
"""


config = json.load(open("config.json"))
audio = AudioHandler(config)


def main():
    st.title("Voice Interview")
    st.write("Select which stages to run and click Start Interview.")

    if "transcripts" not in st.session_state:
        st.session_state["transcripts"] = []
    if "recording" not in st.session_state:
        st.session_state["recording"] = False

    stage_names = [stage for _, stage in ALL_STAGES]
    selected = st.multiselect("Stages", stage_names, default=stage_names)

    for t in st.session_state["transcripts"]:
        st.markdown(t)

    if st.button("Start Interview"):
        checkpoint_dir = Path("interview_outputs")
        checkpoint_path = checkpoint_dir / "checkpoint_session.json"

        if checkpoint_path.exists():
            st.info("Resuming from checkpoint...")
            session = load_existing_session(checkpoint_path)
        else:
            session = create_new_session()

        stages_to_run = [s for s in ALL_STAGES if not selected or s[1] in selected]

        for role, stage in stages_to_run:
            if session.get(stage):
                st.write(f"Skipping completed stage: {stage}")
                continue

            st.header(f"Running stage: {stage}")
            session = run_stage(role=role, stage=stage, session=session)
            save_checkpoint(session, checkpoint_path)

            result = session.get(stage, {})
            transcript = result.get("transcript", [])
            summary = result.get("summary")

            if transcript:
                for qa in transcript:
                    st.markdown(f"**Q:** {qa['q']}")
                    st.markdown(f"**A:** {qa['a']}")
            if summary:
                st.subheader("Summary")
                st.write(summary)

        st.success("Interview complete.")
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    # --- Bottom input bar ---
    bottom = st.container()
    with bottom:
        st.markdown(
            """
            <style>
            #bottom-bar{position:fixed;bottom:0;width:100%;background:#f9f9f9;padding:10px 5px;z-index:1000;}
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div id='bottom-bar'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1,6])

        if col1.button("🎤", key="mic"):
            st.session_state["recording"] = True
            wave = col1.empty()
            wave.markdown(WAVE_HTML, unsafe_allow_html=True)
            path = audio.record_until_silence()
            wave.empty()
            text = audio.transcribe(path)
            st.session_state["recording"] = False
            if text:
                st.session_state["transcripts"].append(text)

        user_text = col2.chat_input("Type here")
        if user_text:
            st.session_state["transcripts"].append(user_text)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()