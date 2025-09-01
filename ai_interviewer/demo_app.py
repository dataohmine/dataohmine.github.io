#!/usr/bin/env python3
"""
AI Interview Assistant - Demo Version (Text-Only)
Simplified version for cloud deployment without audio dependencies
"""

import json
import streamlit as st
from datetime import datetime
import time
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AI Interview Assistant - Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for modern UI
MODERN_CSS = """
<style>
/* Main styling */
.header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}

.header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    font-weight: 700;
}

.chat-container {
    background: white;
    border-radius: 15px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    max-height: 500px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

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

.feature-card {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
    border-left: 4px solid #667eea;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""

# Sample interview questions
SAMPLE_QUESTIONS = [
    {
        "id": "intro",
        "text": "Welcome to the AI Interview Assistant! Can you please introduce yourself and tell me about your background?",
        "category": "Introduction"
    },
    {
        "id": "experience",
        "text": "Tell me about your most relevant work experience for this position.",
        "category": "Experience"
    },
    {
        "id": "strengths",
        "text": "What are your key strengths and how do they relate to this role?",
        "category": "Skills"
    },
    {
        "id": "challenges",
        "text": "Describe a challenging project you worked on and how you overcame obstacles.",
        "category": "Problem Solving"
    },
    {
        "id": "goals",
        "text": "What are your career goals and why are you interested in this position?",
        "category": "Motivation"
    }
]

def generate_ai_feedback(answer):
    """Generate simple AI feedback for demo purposes"""
    if len(answer.split()) < 5:
        return "⚠️ Consider providing more detailed responses to showcase your experience and skills."
    elif len(answer.split()) > 100:
        return "✅ Great detailed response! Remember to be concise while covering key points."
    else:
        return "✅ Good response length. Consider adding specific examples to strengthen your answer."

def main():
    # Apply modern CSS
    st.markdown(MODERN_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="header">
        <h1>🤖 AI Interview Assistant</h1>
        <p>Advanced interview preparation platform - Demo Version</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "current_question" not in st.session_state:
        st.session_state.current_question = 0
    if "responses" not in st.session_state:
        st.session_state.responses = {}
    if "interview_started" not in st.session_state:
        st.session_state.interview_started = False
    if "candidate_name" not in st.session_state:
        st.session_state.candidate_name = ""
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Interview Setup")
        
        # Candidate info
        candidate_name = st.text_input(
            "Your Name",
            value=st.session_state.candidate_name,
            placeholder="Enter your full name"
        )
        if candidate_name:
            st.session_state.candidate_name = candidate_name
            
        job_position = st.text_input(
            "Target Position",
            placeholder="e.g., Software Engineer"
        )
        
        company_name = st.text_input(
            "Company Name",
            placeholder="e.g., Google, Microsoft"
        )
        
        st.markdown("---")
        
        # Interview progress
        if st.session_state.interview_started:
            st.markdown("### 📊 Progress")
            progress = (st.session_state.current_question / len(SAMPLE_QUESTIONS)) * 100
            st.progress(progress / 100)
            st.write(f"Question {st.session_state.current_question + 1} of {len(SAMPLE_QUESTIONS)}")
            st.write(f"Responses: {len(st.session_state.responses)}")
        
        st.markdown("---")
        
        # Demo features
        st.markdown("### ✨ Demo Features")
        st.success("✅ Text-based interview")
        st.success("✅ AI feedback")
        st.success("✅ Progress tracking")
        st.info("🎤 Audio recording (Full version)")
        st.info("🎯 Advanced AI analysis (Full version)")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎤 Interview", "📝 Responses", "📊 Results"])
    
    with tab1:
        if not st.session_state.interview_started:
            # Start interview section
            st.markdown("### 🚀 Ready to Start Your Interview?")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                <div class="feature-card">
                    <h4>🎯 What to Expect:</h4>
                    <ul>
                        <li>5 comprehensive interview questions</li>
                        <li>Real-time AI feedback on your responses</li>
                        <li>Progress tracking throughout the session</li>
                        <li>Detailed analysis at the end</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if candidate_name:
                    if st.button("🚀 Start Interview", key="start_btn", use_container_width=True):
                        st.session_state.interview_started = True
                        st.balloons()
                        st.rerun()
                else:
                    st.warning("⚠️ Please enter your name first")
        
        else:
            # Interview in progress
            current_q = st.session_state.current_question
            
            if current_q < len(SAMPLE_QUESTIONS):
                question = SAMPLE_QUESTIONS[current_q]
                
                # Progress indicator
                progress = (current_q / len(SAMPLE_QUESTIONS))
                st.markdown(f"""
                <div class="progress-container">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="font-weight: 600;">Question {current_q + 1} of {len(SAMPLE_QUESTIONS)}</span>
                        <span style="color: #718096;">{question['category']}</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {progress * 100}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Current question
                st.markdown(f"""
                <div class="ai-message">
                    🤖 <strong>AI Interviewer:</strong><br>
                    {question['text']}
                </div>
                """, unsafe_allow_html=True)
                
                # Response input
                st.markdown("### 💬 Your Response")
                response = st.text_area(
                    f"Response to Question {current_q + 1}:",
                    height=150,
                    placeholder="Type your detailed response here...",
                    key=f"response_{current_q}"
                )
                
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    if st.button("➡️ Submit & Next", key="next_btn"):
                        if response.strip():
                            st.session_state.responses[current_q] = {
                                "question": question,
                                "answer": response.strip(),
                                "timestamp": datetime.now().isoformat(),
                                "feedback": generate_ai_feedback(response.strip())
                            }
                            
                            # Show feedback
                            feedback = st.session_state.responses[current_q]["feedback"]
                            st.success(f"Response saved! {feedback}")
                            
                            st.session_state.current_question += 1
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.warning("⚠️ Please provide a response before continuing.")
                
                with col2:
                    if current_q > 0:
                        if st.button("⬅️ Previous", key="prev_btn"):
                            st.session_state.current_question -= 1
                            st.rerun()
                
                with col3:
                    if st.button("🏁 End Interview", key="end_btn"):
                        st.session_state.current_question = len(SAMPLE_QUESTIONS)
                        st.rerun()
                
            else:
                # Interview complete
                st.markdown("### 🎉 Interview Complete!")
                st.success(f"Congratulations {candidate_name}! You've completed the interview.")
                st.balloons()
                
                if st.button("🔄 Start New Interview", key="restart_btn"):
                    # Reset session
                    st.session_state.current_question = 0
                    st.session_state.responses = {}
                    st.session_state.interview_started = False
                    st.rerun()
    
    with tab2:
        st.markdown("### 📝 Your Interview Responses")
        
        if st.session_state.responses:
            for idx, (q_num, data) in enumerate(st.session_state.responses.items()):
                question = data["question"]
                answer = data["answer"]
                feedback = data["feedback"]
                
                with st.expander(f"Question {q_num + 1}: {question['category']}", expanded=False):
                    st.markdown(f"**Question:** {question['text']}")
                    st.markdown(f"**Your Answer:** {answer}")
                    st.markdown(f"**AI Feedback:** {feedback}")
                    st.markdown(f"**Word Count:** {len(answer.split())} words")
        else:
            st.info("No responses yet. Complete some interview questions to see them here.")
    
    with tab3:
        st.markdown("### 📊 Interview Analysis")
        
        if st.session_state.responses:
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            total_questions = len(st.session_state.responses)
            total_words = sum(len(data["answer"].split()) for data in st.session_state.responses.values())
            avg_words = total_words // total_questions if total_questions > 0 else 0
            
            with col1:
                st.metric("Questions Answered", total_questions)
            with col2:
                st.metric("Total Words", total_words)
            with col3:
                st.metric("Avg Words/Answer", avg_words)
            with col4:
                completion = (total_questions / len(SAMPLE_QUESTIONS)) * 100
                st.metric("Completion", f"{completion:.0f}%")
            
            # Response analysis
            st.markdown("### 📈 Response Quality Analysis")
            
            categories = {}
            for data in st.session_state.responses.values():
                category = data["question"]["category"]
                word_count = len(data["answer"].split())
                if category not in categories:
                    categories[category] = []
                categories[category].append(word_count)
            
            for category, word_counts in categories.items():
                avg_words = sum(word_counts) / len(word_counts)
                st.write(f"**{category}:** {avg_words:.1f} words average")
            
            # Overall feedback
            st.markdown("### 🎯 Overall Performance")
            if avg_words < 30:
                st.warning("💡 Consider providing more detailed responses to better showcase your experience.")
            elif avg_words > 80:
                st.info("📝 Great detail in your responses! Consider being more concise for better impact.")
            else:
                st.success("✅ Good balance of detail and conciseness in your responses!")
            
            # Download results
            if st.button("📥 Download Interview Report"):
                report = {
                    "candidate": st.session_state.candidate_name,
                    "timestamp": datetime.now().isoformat(),
                    "responses": st.session_state.responses,
                    "statistics": {
                        "total_questions": total_questions,
                        "total_words": total_words,
                        "average_words": avg_words
                    }
                }
                
                st.download_button(
                    label="📄 Download JSON Report",
                    data=json.dumps(report, indent=2),
                    file_name=f"interview_report_{candidate_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.info("Complete the interview to see detailed analysis and statistics.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #718096; padding: 1rem;">
        <p>🤖 <strong>AI Interview Assistant Demo</strong> • Built with Streamlit & OpenAI</p>
        <p>⚡ <em>This is a simplified demo version. The full version includes advanced AI analysis and audio recording.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()