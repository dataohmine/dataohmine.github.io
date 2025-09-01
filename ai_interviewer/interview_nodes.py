import json
import os
import warnings
from pathlib import Path
from utils.audio_utils import AudioHandler
from utils.text_utils import extract_first_last
from openai import OpenAI
from datetime import datetime
from pydub import AudioSegment
from io import BytesIO
import logging
import re

# Suppress all warnings including pkg_resources deprecation
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress all logging completely
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
logging.disable(logging.CRITICAL)

# Configuration
config = json.load(open("config.json"))

# Additional logging suppression for third-party libraries
for logger_name in ['pydub', 'openai', 'urllib3', 'requests']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(logger_name).disabled = True

audio = AudioHandler(config)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_questions_node(state):
    # Load questions from the single interview file
    question_file = "questions/interview.json"
    
    # Load questions from the JSON file
    path = Path(question_file)
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {question_file}")
    
    with open(path, 'r', encoding='utf-8') as file:
        state['questions'] = json.load(file)
    
    return state

def ask_question_node(state):
    while state['current_q_idx'] < len(state['questions']):
        idx = state['current_q_idx']
        question = state['questions'][idx]
        q = question['text']

        print(f"Asking question {idx + 1}/{len(state['questions'])}: {question.get('id', 'unknown')}")

        # Check if this is the outro question - just play it and move on
        if question.get('id') == 'outro':
            print("Playing outro message...")
            audio_bytes = audio.speak(q)
            audio.play_audio(audio_bytes)
            
            # Store the audio segment
            state['audio_segments'].append(AudioSegment.from_file(BytesIO(audio_bytes), format="mp3"))
            
            # Store a completion message in transcript
            store_transcription(state, "[Interview completed]", question)
            
            # Move to next question index and break
            state['current_q_idx'] += 1
            break

        # For all other questions, follow normal flow
        # Initialize retry counter for this question
        retry_count = 0
        max_retries = 2  # Limit retries to prevent infinite loops
        
        while retry_count <= max_retries:
            # Ask the question and handle audio
            if not ask_and_record_audio(state, q, question):
                retry_count += 1
                if retry_count > max_retries:
                    print(f"  No audio detected after {max_retries} attempts, skipping question")
                    # Store empty response if max retries reached
                    store_transcription(state, "[No audio detected]", question)
                    break
                continue

            # Wait for a valid transcription response
            transcription = wait_for_transcription(state, question)
            print(f"¤ Transcribed: '{transcription}'")

            # Handle responses and clarify if needed
            if not validate_response(state, transcription, question):
                retry_count += 1
                if retry_count > max_retries:
                    print(f"  Max retries reached, accepting response: '{transcription}'")
                    # Store whatever transcription we got if max retries reached
                    store_transcription(state, transcription, question)
                    break
                print(f"„ Retry {retry_count}/{max_retries}")
                continue
            
            # Store valid transcriptions
            store_transcription(state, transcription, question)
            
            # Handle any special logic for specific questions (like name extraction)
            # BUT DO NOT INCREMENT QUESTION INDEX HERE
            handle_special_logic(state, transcription, question)
            break
        
        # Move to the next question ONLY ONCE per question
        state['current_q_idx'] += 1
        print(f"  Moving to question {state['current_q_idx'] + 1}")
    
    # Mark questions as complete when we've asked all questions
    state['questions_complete'] = True
    print(" All questions completed!")
    
    # Once all questions are asked, end the interview
    state = end_interview_node(state)
    
    return state

def handle_special_logic(state, transcription, question):
    # Handle name question
    if question["id"] == "name":
        fn, ln = extract_first_last(transcription)
        state["session"]["candidate"] = f"{fn}_{ln}"
        # Also store first_name and last_name for session saving
        state["session"]["first_name"] = fn
        state["session"]["last_name"] = ln
        audio.play_audio(audio.speak(f"Thanks, {fn}. Let's begin."))




def validate_response(state, transcription, question):
    # If the transcription is empty or contains system prompts (like "Use complete sentences")
    if not transcription or "use complete sentences" in transcription.lower():
        print(f" No valid response detected. Can you please clarify or repeat your answer?")
        audio_bytes = audio.speak("Can you please clarify or repeat your answer?")
        audio.play_audio(audio_bytes)
        return False
    
    # Check for common misheard phrases like "thank you", "see you", etc.
    misheard_responses = ["thank you", "thanks", "see you", "see ya", "see y'all", "goodbye"]
    transcription_clean = transcription.strip().lower()

    if any(misheard in transcription_clean for misheard in misheard_responses):
        print(f" Misunderstanding detected. Please provide a proper response.")
        audio_bytes = audio.speak("I believe you said 'thank you' or something similar. Could you please provide a response?")
        audio.play_audio(audio_bytes)
        return False
    
    # For role questions, be very lenient since CEO/CFO often get transcribed as short words
    question_id = question.get("id", "").lower()
    
    # Special handling for role questions like CEO, CFO, etc.
    if "role" in question_id:
        if is_likely_executive_role(transcription_clean):
            print(f" Executive role detected: {transcription}")
            return True
        elif len(transcription_clean) >= 1:
            print(f" Role response accepted: {transcription}")
            return True
        else:
            print(" No role detected")
            audio_bytes = audio.speak("Could you please state your role or title?")
            audio.play_audio(audio_bytes)
            return False
    
    # Handle name question to ensure it's a valid name (not too short)
    if question_id == "name":
        if len(transcription_clean) < 2:
            print(" Name too short")
            audio_bytes = audio.speak("Could you please state your name clearly?")
            audio.play_audio(audio_bytes)
            return False
        print(f" Name accepted: {transcription}")
        return True

    # For general questions, if the response is too short or seems incomplete, ask for clarification
    if len(transcription_clean) <= 2:
        print(f" Transcription too short (possible audio issue): '{transcription}'")
        audio_bytes = audio.speak("I only caught a few words. Could you please repeat your full answer?")
        audio.play_audio(audio_bytes)
        return False
    
    return True

def is_likely_executive_role(text):
    """Check if the transcription likely represents an executive role"""
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Direct matches for common executive roles
    executive_roles = [
        'ceo', 'cfo', 'coo', 'cto', 'cmo', 'president', 'founder', 
        'director', 'manager', 'vp', 'vice president', 'chief',
        'owner', 'partner', 'head', 'lead'
    ]
    
    # Check for direct matches
    for role in executive_roles:
        if role in text_lower:
            return True
    
    # Check for common transcription errors of CEO/CFO/COO
    ceo_variants = ['io', 'co', 'seo', 'see o', 'c e o', 'c.e.o', 'seeo', 'cio']
    cfo_variants = ['fo', 'sefo', 'c f o', 'c.f.o', 'seafo', 'seaphone']
    coo_variants = ['ku', 'coo', 'c o o', 'c.o.o', 'koo']
    
    all_variants = ceo_variants + cfo_variants + coo_variants
    
    # Check if the entire transcription matches a variant (for short responses)
    if text_lower in all_variants:
        return True
    
    # Check if any variant is contained in the text
    for variant in all_variants:
        if variant in text_lower:
            return True
    
    return False

def ask_and_record_audio(state, question_text, question):
    audio_bytes = audio.speak(question_text)
    audio.play_audio(audio_bytes)
    
    # Store the audio segment
    state['audio_segments'].append(AudioSegment.from_file(BytesIO(audio_bytes), format="mp3"))
    
    # Give a moment for the user to start speaking
    print("¤ Recording... (speak now)
    
    # Record the response with extended timeout
    path = audio.record_until_silence()
    if not path:  # If no audio detected, return False but don't repeat immediately
        print(" No audio recorded")
        return False
    
    # Check if audio file exists and has reasonable size
    try:
        import os
        file_size = os.path.getsize(path)
        print(f" Audio file size: {file_size} bytes")
        if file_size < 1000:  # Less than 1KB suggests very short recording
            print("  Audio file seems very short - this might cause transcription issues")
    except:
        pass
    
    state['last_audio_path'] = path
    print(" Audio recorded successfully")
    return True

def wait_for_transcription(state, question):
    transcription = ""
    try:
        with open(state['last_audio_path'], 'rb') as f:
            # Use context-specific prompts if available
            context_prompt = get_context_specific_prompt(question, config)
            
            result = client.audio.transcriptions.create(
                file=f,
                model=config["transcription_model"],
                prompt=context_prompt,
                temperature=config.get("default_transcription_temperature", 0),
                language=config.get("default_transcription_language", "en")
            )
        raw_transcription = result.text.strip()
        
        # Apply post-processing to fix common business acronym errors
        transcription = fix_business_acronyms(raw_transcription, question)
        
        print(f" Raw transcription: '{raw_transcription}'")
        if raw_transcription != transcription:
            print(f"§ Fixed transcription: '{transcription}'")
        
        # Check if transcription seems unusually short
        if len(transcription) < 5:
            print(f"  Warning: Transcription seems very short: '{transcription}'")
        
        # Clean up the audio file after transcription
        try:
            os.unlink(state['last_audio_path'])
        except:
            pass
            
    except Exception as e:
        print(f" Transcription error: {e}")
        # Don't fail silently - return what we can
        transcription = ""
    return transcription

def fix_business_acronyms(text, question):
    """Fix common business acronym transcription errors with improved CEO detection"""
    if not text:
        return text
    
    # Get question type for context
    question_id = question.get("id", "").lower()
    
    # Enhanced CEO/CFO/COO detection
    ceo_fixes = {
        # Common CEO transcription errors
        r'\bio\b': 'CEO',
        r'\bco\b': 'CEO',
        r'\bseo\b': 'CEO',
        r'\bseeo\b': 'CEO',
        r'\bcio\b': 'CEO',
        r'\bc e o\b': 'CEO',
        r'\bc\.e\.o\b': 'CEO',
        r'\bsee o\b': 'CEO',
        r'\bsee oh\b': 'CEO',
        r'\bc o\b': 'CEO',
        r'\beeo\b': 'CEO',
        r'\beo\b': 'CEO',
        # Handle cases where only part is transcribed
        r'^io$': 'CEO',
        r'^co$': 'CEO',
        r'^seo$': 'CEO',
        r'^o$': 'CEO',  # Sometimes just the "O" is caught
        # Misunderstood variations for CEO
        r'\bsee y\'all\b': 'CEO',
        r'\bsee e o\b': 'CEO',
        r'\bsee ya\b': 'CEO',
        r'\bsee you\b': 'CEO',
        r'\bsea oh\b': 'CEO',
        r'\bseeo\b': 'CEO',
    }
    
    cfo_fixes = {
        r'\bseafo\b': 'CFO',
        r'\bseafoo\b': 'CFO', 
        r'\bseaphone\b': 'CFO',
        r'\bseefo\b': 'CFO',
        r'\bc f o\b': 'CFO',
        r'\bc\.f\.o\b': 'CFO',
        r'\bsefo\b': 'CFO',
        r'^fo$': 'CFO',
        r'^sefo$': 'CFO',
    }
    
    coo_fixes = {
        r'\bku\b': 'COO',
        r'\bkoo\b': 'COO',
        r'\bc o o\b': 'COO',
        r'\bc\.o\.o\b': 'COO',
        r'^ku$': 'COO',
        r'^koo$': 'COO',
    }
    
    # Other business terms
    other_fixes = {
        r'\bipl\b': 'IPL',
        r'\bipo\b': 'IPO',
        r'\bi p o\b': 'IPO',
        r'\bi\.p\.o\b': 'IPO',
        
        # Functional areas
        r'\bsails\b': 'sales',
        r'\bmarketting\b': 'marketing',
        r'\bopps\b': 'ops',
        
        # Common business words
        r'\bfinanse\b': 'finance',
        r'\bengineering\b': 'engineering',
        r'\btechnology\b': 'technology',
    }
    
    fixed_text = text
    
    # Apply all fixes with case-insensitive matching
    all_fixes = {**ceo_fixes, **cfo_fixes, **coo_fixes, **other_fixes}
    
    for pattern, replacement in all_fixes.items():
        fixed_text = re.sub(pattern, replacement, fixed_text, flags=re.IGNORECASE)
    
    # For role questions, be extra aggressive with CEO detection
    if "role" in question_id:
        # If the entire response is just a few characters, check if it's likely CEO
        if len(fixed_text.strip()) <= 3:
            text_lower = fixed_text.lower().strip()
            if text_lower in ['io', 'co', 'seo', 'o', 'eo', 'cio', 'seeo']:
                fixed_text = 'CEO'
                print(f"§ Aggressive CEO fix applied: '{text}' -> 'CEO'")
    
    return fixed_text

def get_context_specific_prompt(question, config):
    """Get the appropriate transcription prompt based on question type"""
    question_id = question.get("id", "").lower()
    
    # Use specific prompts for different question types
    if "role" in question_id:
        return config.get("role_transcription_prompt", config.get("default_transcription_prompt", ""))
    elif "exit" in question_id:
        return config.get("exit_transcription_prompt", config.get("default_transcription_prompt", ""))
    elif "functional" in question_id:
        return config.get("functional_transcription_prompt", config.get("default_transcription_prompt", ""))
    else:
        return config.get("default_transcription_prompt", "")

def store_transcription(state, transcription, question):
    print(f"Q: {question['text']}")
    print(f"A: {transcription}")
    print("-" * 50)
    state["transcript"].append({"id": question["id"], "q": question["text"], "a": transcription})


def detect_unclear_responses(transcript):
    """Identify questions that received no or unclear answers."""
    unclear = []
    for entry in transcript:
        if entry.get("id") == "outro":
            continue

        answer = entry.get("a", "").strip().lower()
        if not answer or answer in {"[no audio detected]", "thank you", "thanks", "ok", "okay"}:
            unclear.append(entry["q"])
            continue
        if len(answer.split()) <= 2:
            unclear.append(entry["q"])
    return unclear



def save_session_node(state):
    # Get the first and last name from the session
    first_name = state["session"].get("first_name", "Unknown")
    last_name = state["session"].get("last_name", "Unknown")
    
    # Create timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    out_dir = Path("interview_outputs") / f"{first_name}_{last_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save complete session data as JSON
    session_data = {
        "participant_info": {
            "first_name": first_name,
            "last_name": last_name,
            "timestamp": ts,
            "stage": state["stage"],
            "role": state["role"]
        },
        "transcript": state["transcript"],
        "session_data": state["session"]
    }
    
    json_path = out_dir / f"{first_name}_{last_name}_{ts}_interview.json"
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)
    
    # Save audio file if available
    if state["audio_segments"]:
        try:
            # Combine all audio segments. sum() does not work without a start
            # value, so build the final audio manually.
            full_audio = AudioSegment.silent(duration=0)
            for segment in state["audio_segments"]:
                full_audio += segment
            audio_path = out_dir / f"{first_name}_{last_name}_{ts}_audio.mp3"
            full_audio.export(str(audio_path), format="mp3")
            print(f" Audio saved: {audio_path}")
        except Exception as e:
            print(f" Audio export error: {e}")
    
    # Print completion message
    print(f" Interview completed for {first_name} {last_name}")
    print(f" Files saved in: {out_dir}")
    print(f"„ JSON file: {json_path}")
    
    # Mark session as complete
    state['session_complete'] = True
    
    return state

def summary_node(state):
    print("‹ Generating summary...")
    
    # If there are no transcriptions, create a default message
    if not state["transcript"]:
        state["session"][state["stage"]] = {
            "transcript": [],
            "summary": "No responses recorded for this stage."
        }
        return state

    # Create a string to hold the question/answer pairs
    qa = "\n".join(f"Q: {t['q']}\nA: {t['a']}" for t in state["transcript"])
    
    # Create a prompt for summarizing the responses
    prompt = f"{config['system_prompt']}\n\nPlease summarize:\n\n{qa}"
    
    # Attempt to generate a summary using the OpenAI client
    try:
        response = client.chat.completions.create(
            model=config["summary_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        summary = response.choices[0].message.content.strip()
        unclear = detect_unclear_responses(state["transcript"])
        if unclear:
            summary += "\n\n**Clarifications Needed:**\n" + "\n".join(f"- {q}" for q in unclear)
        print(" Summary generated successfully")
    except Exception as e:
        summary = f"Summary generation failed: {e}"
        print(f" Summary generation failed: {e}")

    # Save the summary to the session
    state["session"][state["stage"]] = {
        "transcript": state["transcript"],
        "summary": summary
    }
    return state

def route_after_question(state):
    """Determine whether to ask the next question or summarize."""
    current_idx = state.get('current_q_idx', 0)
    total_questions = len(state.get('questions', []))
    questions_complete = state.get('questions_complete', False)
    
    print(f"€ Routing: questions_complete={questions_complete}, current_idx={current_idx}, total={total_questions}")
    
    # Check if all questions have been asked
    if questions_complete or current_idx >= total_questions:
        print("  Routing to summary")
        return "summarize"
    else:
        print("  Routing to ask_question")
        return "ask_question"

def route_after_summary(state):
    """Route to save session after summary is complete."""
    print("  Routing to save_session")
    return "save_session"

def end_interview_node(state):
    """Final node to end the interview gracefully."""
    # Play goodbye message
    goodbye_message = "Thank you for completing the interview. Have a great day!"
    audio_bytes = audio.speak(goodbye_message)
    audio.play_audio(audio_bytes)
    
    # Store the goodbye message in audio segments
    state['audio_segments'].append(AudioSegment.from_file(BytesIO(audio_bytes), format="mp3"))
    
    # Mark the interview as ended
    state['interview_ended'] = True
    print("¯ Interview session ended successfully!")
    return state