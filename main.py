from openai import OpenAI
import sounddevice as sd
import whisper
import tempfile
import numpy as np
import scipy.io.wavfile as wav
import os
from elevenlabs import play
from elevenlabs import ElevenLabs
# Initialize Whisper model
whisper_model = whisper.load_model("base")

# Create openai client
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Create 
cli_el = ElevenLabs(
    api_key=os.environ.get("ELEVENLABS_API_KEY")
)


# Record audio from the user (using sounddevice library)
def record_audio(duration=5, fs=16000):
    print("Recording... 🎙️")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(audio_data)

# Save audio data to a temporary WAV file
def save_audio_to_file(audio_data, fs):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(temp_file.name, fs, audio_data)
    return temp_file.name

# Transcribe audio with Whisper
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    print(f"Transcription: {result['text']}")
    return result['text']

# Generate a response with ChatGPT
def generate_response(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "user",
                "content": text
            }
        ]
    )
    return response.choices[0].message.content

def play_audio(text):
    audio = cli_el.generate(text=text)
    play(audio)

# Example Workflow
while True:
    audio_data = record_audio()  # Record user input
    audio_path = save_audio_to_file(audio_data, fs=16000)  # Save audio as WAV file
    transcribed_text = transcribe_audio(audio_path)  # Transcribe with Whisper
    response_text = generate_response(transcribed_text)  # Get response from ChatGPT
    print(f"ChatGPT: {response_text}")
    play_audio(text=response_text)

