import os
import json
from datetime import datetime
import types
import sys

# Provide dummy modules so interview_nodes can import without real deps
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

requests_mod = types.ModuleType("requests")
def dummy_post(*a, **k):
    class R:
        content = b""
        def raise_for_status(self):
            pass
    return R()
requests_mod.post = dummy_post
sys.modules['requests'] = requests_mod

pyaudio_mod = types.ModuleType("pyaudio")
class DummyStream:
    def read(self, *a, **k):
        return b""
    def stop_stream(self):
        pass
    def close(self):
        pass
class DummyPyAudio:
    def open(self, *a, **k):
        return DummyStream()
    def terminate(self):
        pass
pyaudio_mod.PyAudio = DummyPyAudio
pyaudio_mod.paInt16 = 0
sys.modules['pyaudio'] = pyaudio_mod

webrtcvad_mod = types.ModuleType("webrtcvad")
class DummyVAD:
    def __init__(self, *a, **k):
        pass
    def is_speech(self, frame, rate):
        return False
webrtcvad_mod.Vad = DummyVAD
sys.modules['webrtcvad'] = webrtcvad_mod

dotenv_mod = types.ModuleType("dotenv")
dotenv_mod.load_dotenv = lambda: None
sys.modules['dotenv'] = dotenv_mod

audio_utils = types.ModuleType("utils.audio_utils")
class DummyAudioHandler:
    def __init__(self, config=None):
        pass
    def speak(self, text):
        return b""
    def play_audio(self, audio_bytes):
        pass
    def record_until_silence(self):
        return ""
    def transcribe(self, path):
        return ""
audio_utils.AudioHandler = DummyAudioHandler
sys.modules['utils.audio_utils'] = audio_utils

openai = types.ModuleType("openai")
class DummyOpenAI:
    def __init__(self, api_key=None):
        pass
    class audio:
        class transcriptions:
            @staticmethod
            def create(*a, **k):
                class R: text = ""
                return R()
    class chat:
        class completions:
            @staticmethod
            def create(*a, **k):
                class Msg:
                    content = ""
                class Choice:
                    message = Msg()
                class R:
                    choices = [Choice()]
                return R()
openai.OpenAI = DummyOpenAI
sys.modules['openai'] = openai

pydub_mod = types.ModuleType("pydub")
class DummyAudioSegment:
    @classmethod
    def from_file(cls, *a, **k):
        return cls()
    def __add__(self, other):
        return self
    def export(self, *a, **k):
        pass
pydub_mod.AudioSegment = DummyAudioSegment
playback_mod = types.ModuleType("pydub.playback")
playback_mod.play = lambda *a, **k: None
sys.modules['pydub'] = pydub_mod
sys.modules['pydub.playback'] = playback_mod

# Import after stubbing dependencies
import interview_nodes
from interview_nodes import load_questions_node, save_session_node

class DummyAudio(DummyAudioHandler):
    pass


def test_load_questions_node_operator():
    state = {"role": "operator", "stage": "post_operator"}
    interview_nodes.audio = DummyAudio()
    result = load_questions_node(state)
    assert isinstance(result["questions"], list)
    assert result["questions"]


def test_load_questions_node_sponsor():
    state = {"role": "sponsor", "stage": "pre_sponsor"}
    interview_nodes.audio = DummyAudio()
    result = load_questions_node(state)
    assert isinstance(result["questions"], list)
    assert result["questions"]


def test_save_session_node(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    interview_nodes.audio = DummyAudio()
    fixed_time = datetime(2024, 1, 1, 12, 0, 0)
    class MockDateTime(datetime):
        @classmethod
        def now(cls):
            return fixed_time
    monkeypatch.setattr(interview_nodes, "datetime", MockDateTime)
    state = {
        "session": {"first_name": "Test", "last_name": "User"},
        "stage": "pre_operator",
        "role": "operator",
        "transcript": [],
        "audio_segments": []
    }
    result = save_session_node(state)
    folder = tmp_path / "interview_outputs" / f"Test_User_{fixed_time.strftime('%Y%m%d_%H%M%S')}"
    json_file = folder / f"Test_User_{fixed_time.strftime('%Y%m%d_%H%M%S')}_interview.json"
    assert json_file.exists()
    data = json.load(open(json_file))
    assert data["participant_info"]["first_name"] == "Test"
    assert result["session_complete"] is True
