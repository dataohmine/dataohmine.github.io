import json
import streamlit as st
from pathlib import Path
from datetime import datetime
from langgraph_interview_flow import run_stage
from main import ALL_STAGES, create_new_session, load_existing_session, save_checkpoint
from utils.audio_utils import AudioHandler
import time

# Enhanced CSS for modern UI
MODERN_CSS = """
<style>
/* Main container styling */
.main-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    padding: 0;
}

/* Header styling */
.header {
    background: white;
    padding: 1.5rem 2rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
    text-align: center;
}

.header h1 {
    color: #2d3748;
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    font-weight: 700;
}

.header p {
    color: #718096;
    font-size: 1.1rem;
    margin: 0;
}

/* Progress bar */
.progress-container {
    background: white;
    padding: 1rem 2rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 1.5rem;
}

.progress-bar {
    width: 100%;
    height: 10px;
    background-color: #e2e8f0;
    border-radius: 5px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #48bb78, #38a169);
    transition: width 0.3s ease;
}

/* Chat container */
.chat-container {
    background: white;
    border-radius: 15px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    max-height: 500px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

/* Message bubbles */
.ai-message {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 20px 20px 20px 5px;
    margin: 1rem 0;
    max-width: 80%;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.user-message {
    background: #f7fafc;
    color: #2d3748;
    padding: 1rem 1.5rem;
    border-radius: 20px 20px 5px 20px;
    margin: 1rem 0;
    max-width: 80%;
    margin-left: auto;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

/* Recording indicator */
.recording-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem;
    background: #fff5f5;
    border: 2px dashed #fc8181;
    border-radius: 10px;
    margin: 1rem 0;
}

.wave {
    display: flex;
    align-items: flex-end;
    height: 30px;
    margin-right: 1rem;
}

.wave span {
    display: block;
    width: 4px;
    height: 30px;
    margin: 0 2px;
    background: #e53e3e;
    border-radius: 2px;
    animation: wave 1.2s infinite;
}

.wave span:nth-child(2) { animation-delay: 0.1s; }
.wave span:nth-child(3) { animation-delay: 0.2s; }
.wave span:nth-child(4) { animation-delay: 0.3s; }
.wave span:nth-child(5) { animation-delay: 0.4s; }

@keyframes wave {
    0%, 100% { transform: scaleY(0.3); }
    50% { transform: scaleY(1); }
}

/* Control panel */
.control-panel {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    position: sticky;
    bottom: 0;
    z-index: 100;
}

/* Buttons */
.custom-button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 25px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.custom-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* Status indicators */
.status-success {
    color: #38a169;
    font-weight: 600;
}

.status-warning {
    color: #d69e2e;
    font-weight: 600;
}

.status-error {
    color: #e53e3e;
    font-weight: 600;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #555;
}
</style>
"""


# Load configuration
config = json.load(open("config.json"))
audio = AudioHandler(config)

# Page configuration
st.set_page_config(
    page_title="AI Interview Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Apply modern CSS
    st.markdown(MODERN_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="header">
        <h1>🤖 AI Interview Assistant</h1>
        <p>Advanced LangGraph-powered interview preparation platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "transcripts" not in st.session_state:
        st.session_state["transcripts"] = []
    if "recording" not in st.session_state:
        st.session_state["recording"] = False
    if "interview_started" not in st.session_state:
        st.session_state["interview_started"] = False
    if "current_question_idx" not in st.session_state:
        st.session_state["current_question_idx"] = 0
    if "interview_complete" not in st.session_state:
        st.session_state["interview_complete"] = False
    if "session_data" not in st.session_state:
        st.session_state["session_data"] = None
    if "candidate_name" not in st.session_state:
        st.session_state["candidate_name"] = ""

    # Sidebar configuration
    with st.sidebar:
        st.markdown("### 🎯 Interview Configuration")
        
        # Candidate information
        candidate_name = st.text_input(
            "Candidate Name", 
            value=st.session_state.get("candidate_name", ""),
            placeholder="Enter your full name"
        )
        if candidate_name:
            st.session_state["candidate_name"] = candidate_name
        
        # Job position
        job_position = st.text_input(
            "Target Position", 
            placeholder="e.g., Software Engineer, Product Manager"
        )
        
        # Interview stages
        stage_names = [stage for _, stage in ALL_STAGES]
        selected = st.multiselect(
            "Interview Stages", 
            stage_names, 
            default=stage_names,
            help="Select which interview stages to include"
        )
        
        # Interview statistics
        if st.session_state["interview_started"]:
            st.markdown("### 📊 Session Stats")
            st.metric("Questions Answered", len(st.session_state["transcripts"]))
            st.metric("Current Stage", "Interview" if not st.session_state["interview_complete"] else "Completed")
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["🎤 Interview", "📝 Transcript", "📈 Results"])
    
    with tab1:
        # Progress indicator
        if st.session_state["interview_started"]:
            progress = min(st.session_state["current_question_idx"] / max(len(stage_names), 1), 1.0)
            st.markdown(f"""
            <div class="progress-container">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-weight: 600; color: #2d3748;">Interview Progress</span>
                    <span style="color: #718096;">{int(progress * 100)}% Complete</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress * 100}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Chat interface
        chat_container = st.container()
        with chat_container:
            if st.session_state["transcripts"]:
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                for i, transcript in enumerate(st.session_state["transcripts"]):
                    if i % 2 == 0:  # AI message
                        st.markdown(f'<div class="ai-message">🤖 {transcript}</div>', unsafe_allow_html=True)
                    else:  # User message
                        st.markdown(f'<div class="user-message">👤 {transcript}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📝 Interview Transcript")
        if st.session_state["transcripts"]:
            for i in range(0, len(st.session_state["transcripts"]), 2):
                if i < len(st.session_state["transcripts"]):
                    st.markdown(f"**Q:** {st.session_state['transcripts'][i]}")
                if i + 1 < len(st.session_state["transcripts"]):
                    st.markdown(f"**A:** {st.session_state['transcripts'][i + 1]}")
                st.markdown("---")
        else:
            st.info("No transcript available yet. Start the interview to see questions and answers.")
    
    with tab3:
        st.markdown("### 📈 Interview Results")
        if st.session_state["interview_complete"]:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Questions", len(st.session_state["transcripts"]) // 2)
            with col2:
                st.metric("Duration", "N/A")
            with col3:
                st.metric("Completion", "100%")
            
            st.success("✅ Interview completed successfully!")
            if st.button("📄 Generate Report"):
                st.info("Report generation feature coming soon!")
        else:
            st.info("Results will be available after completing the interview.")

    # Interview controls
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if not st.session_state["interview_started"]:
            if st.button("🚀 Start Interview", key="start_btn"):
                if not candidate_name:
                    st.error("Please enter your name before starting the interview.")
                elif not selected:
                    st.error("Please select at least one interview stage.")
                else:
                    st.session_state["interview_started"] = True
                    st.session_state["transcripts"].append(f"Welcome {candidate_name}! Let's begin your interview for the {job_position or 'specified'} position.")
                    st.rerun()
        else:
            if not st.session_state["interview_complete"]:
                if st.button("⏸️ Pause Interview", key="pause_btn"):
                    st.warning("Interview paused. Click 'Resume Interview' to continue.")
                if st.button("🔄 Resume Interview", key="resume_btn"):
                    st.success("Interview resumed!")
    
    with col2:
        if st.session_state["interview_started"] and not st.session_state["interview_complete"]:
            if st.button("🏁 End Interview", key="end_btn"):
                st.session_state["interview_complete"] = True
                st.session_state["transcripts"].append("Thank you for completing the interview!")
                st.rerun()
    
    with col3:
        if st.button("🔄 Reset", key="reset_btn"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main interview logic
    if st.session_state["interview_started"] and not st.session_state["interview_complete"]:
        with st.spinner("🔄 Processing interview..."):
            checkpoint_dir = Path("interview_outputs")
            checkpoint_path = checkpoint_dir / "checkpoint_session.json"

            if checkpoint_path.exists():
                st.info("📂 Resuming from checkpoint...")
                session = load_existing_session(checkpoint_path)
            else:
                session = create_new_session()
                session["candidate"] = candidate_name
                session["position"] = job_position

            stages_to_run = [s for s in ALL_STAGES if not selected or s[1] in selected]

            for role, stage in stages_to_run:
                if session.get(stage):
                    st.success(f"✅ Skipping completed stage: {stage}")
                    continue

                st.markdown(f"### 🎯 Running stage: {stage}")
                
                try:
                    session = run_stage(role=role, stage=stage, session=session)
                    save_checkpoint(session, checkpoint_path)
                    
                    result = session.get(stage, {})
                    transcript = result.get("transcript", [])
                    summary = result.get("summary")

                    if transcript:
                        for qa in transcript:
                            st.session_state["transcripts"].extend([qa['q'], qa['a']])
                            
                    if summary:
                        st.markdown("#### 📋 Summary")
                        st.success(summary)
                        
                except Exception as e:
                    st.error(f"❌ Error during interview: {str(e)}")
                    st.info("Progress has been saved. You can resume later.")
                    break

            st.session_state["interview_complete"] = True
            st.balloons()
            st.success("🎉 Interview completed successfully!")
            
            if checkpoint_path.exists():
                checkpoint_path.unlink()

    # Audio input section (only show during active interview)
    if st.session_state["interview_started"] and not st.session_state["interview_complete"]:
        st.markdown("---")
        st.markdown("### 🎤 Audio Input")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("🎙️ Record Answer", key="record_btn", help="Click to record your answer"):
                st.session_state["recording"] = True
                
                # Show recording indicator
                recording_placeholder = st.empty()
                recording_placeholder.markdown("""
                <div class="recording-indicator">
                    <div class="wave">
                        <span></span><span></span><span></span><span></span><span></span>
                    </div>
                    <span style="color: #e53e3e; font-weight: 600;">🔴 Recording... Speak now!</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Record audio
                try:
                    path = audio.record_until_silence()
                    recording_placeholder.empty()
                    
                    if path:
                        with st.spinner("🔄 Transcribing your response..."):
                            text = audio.transcribe(path)
                        
                        if text:
                            st.session_state["transcripts"].append(text)
                            st.success(f"✅ Recorded: {text[:50]}{'...' if len(text) > 50 else ''}")
                        else:
                            st.warning("⚠️ No speech detected. Please try again.")
                    else:
                        st.error("❌ Recording failed. Please check your microphone.")
                        
                except Exception as e:
                    st.error(f"❌ Recording error: {str(e)}")
                finally:
                    st.session_state["recording"] = False
                    recording_placeholder.empty()
        
        with col2:
            # Text input as alternative
            user_text = st.text_area(
                "Or type your answer:", 
                placeholder="Type your response here...",
                height=100,
                key="text_input"
            )
            
            if st.button("📝 Submit Text Answer", key="submit_text"):
                if user_text.strip():
                    st.session_state["transcripts"].append(user_text.strip())
                    st.success("✅ Text answer submitted!")
                    st.session_state["text_input"] = ""  # Clear the text area
                else:
                    st.warning("⚠️ Please enter some text before submitting.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #718096; padding: 1rem;">
        <p>🤖 Powered by LangGraph & OpenAI • Built with ❤️ for better interviews</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()