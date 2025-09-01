import os
import wave
import tempfile
import requests
import pyaudio
import warnings
import audioop
import math

# Suppress pkg_resources deprecation warning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import webrtcvad
from pydub import AudioSegment
from pydub.playback import play
from openai import OpenAI
from dotenv import load_dotenv
import json

class AudioHandler:
    def __init__(self, config):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.voice_model = config['voice_model']
        self.voice_name = config['voice_name']
        self.transcription_model = config['transcription_model']

        audio_settings = config.get('audio_settings', {})
        self.silence_threshold = audio_settings.get('silence_threshold', -50)
        self.silence_duration = audio_settings.get(
            'silence_duration', config.get('silence_seconds', 2.0)
        )

        self.frame_dur = 30
        self.rate = audio_settings.get('sample_rate', 16000)
        self.frame_size = int(self.rate * self.frame_dur / 1000)
        self.channels = 1
        self.format = pyaudio.paInt16
        max_dur = audio_settings.get('max_recording_duration', 60)
        self.max_frames = int(max_dur * 1000 / self.frame_dur)
        self.silence_frames = int(self.silence_duration * 1000 / self.frame_dur)
        # Use a more aggressive VAD for quicker silence detection
        self.vad = webrtcvad.Vad(3)

    def speak(self, text):
        headers = {
            'Authorization': f"Bearer {os.getenv('OPENAI_API_KEY')}",
            'Content-Type': 'application/json'
        }
        payload = {
            'model': self.voice_model,
            'input': text,
            'voice': self.voice_name
        }
        response = requests.post(
            'https://api.openai.com/v1/audio/speech',
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.content

    def play_audio(self, audio_bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(audio_bytes)
            path = f.name
        play(AudioSegment.from_file(path))
        os.remove(path)

    def record_until_silence(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.frame_size)
        frames = []
        heard, silent = False, 0

        for _ in range(self.max_frames):
            frame = stream.read(self.frame_size, exception_on_overflow=False)
            frames.append(frame)

            # Determine if this frame contains speech using VAD and volume level
            is_speech = self.vad.is_speech(frame, self.rate)
            rms = audioop.rms(frame, 2)
            db = 20 * math.log10(rms / 32768) if rms > 0 else -math.inf

            if is_speech or db > self.silence_threshold:
                heard = True
                silent = 0
            elif heard:
                silent += 1

            if heard and silent >= self.silence_frames:
                break
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        if not heard:
            return ''  # Return empty if no speech detected
        
        # Save audio to temp file safely
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            with wave.open(tmp_file, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
            temp_path = tmp_file.name

        return temp_path

    def transcribe(self, path):
        if not path or not os.path.exists(path):
            return ''
        
        with open(path, 'rb') as f:
            resp = self.client.audio.transcriptions.create(
                file=f,
                model=self.transcription_model,
                response_format='text'
            )
        #return resp.text.strip() if resp else ""
        if resp:
            return resp.strip() if isinstance(resp, str) else resp.text.strip()
        return ""