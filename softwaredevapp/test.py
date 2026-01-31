from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import soundcard as sc
import soundfile as sf
import numpy as np
import io
import os

initial_chunk_duration = 8.0
min_chunk_duration = 2.0
chunk_decrement = 1.0
sample_rate = 16000
whisper_size = "small"
mic = sc.default_microphone()

print("Loading Whisper model (this may take a while on first run)...")
try:
    model = WhisperModel(whisper_size, device="cpu", compute_type="int8")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

print("Loading speaker diarization model...")
try:
    HF_TOKEN = "hf_IxRemcmOXbUwQaONBIRLqjzsoQRWhJFVJae"
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
    print("Diarization model loaded successfully!")
except Exception as e:
    print(f"Error loading diarization model: {e}")
    print("Continuing without speaker diarization...")
    diarization_pipeline = None

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def transcribe_with_speakers(audioIn):
    temp_file = "temp_audio.wav"
    sf.write(temp_file, audioIn, sample_rate)

    if diarization_pipeline:
        diarization = diarization_pipeline(temp_file)
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
    else:
        speaker_segments = None
    
    buffer = io.BytesIO()
    sf.write(buffer, audioIn, sample_rate, format='WAV')
    buffer.seek(0)
    
    segments, info = model.transcribe(buffer, beam_size=5)
    
    transcription_parts = []
    for segment in segments:
        text = segment.text
        start_time = segment.start
        end_time = segment.end
        
        speaker = "Unknown"
        if speaker_segments:
            for spk_seg in speaker_segments:
                if (start_time >= spk_seg['start'] and start_time < spk_seg['end']) or \
                   (end_time > spk_seg['start'] and end_time <= spk_seg['end']):
                    speaker = spk_seg['speaker']
                    break
        
        transcription_parts.append(f"[{speaker}]: {text}")
    
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return "\n".join(transcription_parts)

recorded = np.zeros((0, 1), dtype=np.float32)
current_chunk_duration = initial_chunk_duration

clear_screen()

try:
    while True:
        with mic.recorder(samplerate=sample_rate, channels=1) as recorder:
            chunk = recorder.record(numframes=int(current_chunk_duration * sample_rate))

        recorded = np.concatenate((recorded, chunk))

        text = transcribe_with_speakers(recorded)
        
        clear_screen()
        print(text)
        print()

        if current_chunk_duration > min_chunk_duration:
            current_chunk_duration = max(current_chunk_duration - chunk_decrement, min_chunk_duration)

except KeyboardInterrupt:
    clear_screen()
    print("Final transcription:")
    print(text)
    print("\nStopped.")