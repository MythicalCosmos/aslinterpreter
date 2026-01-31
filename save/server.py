from flask import Flask, request, jsonify, render_template
from faster_whisper import WhisperModel
from pydub import AudioSegment
from flask_cors import CORS
import soundfile as sf
import numpy as np
import time
import os

app = Flask(__name__)
CORS(app)

SAMPLE_RATE = 16000
INITIAL_CHUNK = 4.0
MIN_CHUNK = 2.0
CHUNK_DECREMENT = 0.5
USE_CUDA = False

device = "cuda" if USE_CUDA else "cpu"
compute_type = "float16" if USE_CUDA else "int8"
print(f"Loading Whisper model device={device} compute_type={compute_type}")
model = WhisperModel("small", device=device, compute_type=compute_type)
print("Whisper model loaded")

sessions = {}

def process_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return samples

def transcribe(audio_data):
    temp_file = f"temp_{int(time.time()*1000)}.wav"
    sf.write(temp_file, audio_data, SAMPLE_RATE)
    try:
        segments, _ = model.transcribe(temp_file, beam_size=5)
        result = []
        for seg in segments:
            if seg.text.strip():
                result.append({'text': seg.text.strip(), 'start': seg.start, 'end': seg.end})
        full_text = " ".join([s['text'] for s in result])
        return result, full_text
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

@app.route('/start_session', methods=['POST'])
def start_session():
    session_id = str(int(time.time() * 1000))
    sessions[session_id] = {'audio': np.array([], dtype=np.float32), 'chunk_duration': INITIAL_CHUNK}
    return jsonify({'session_id': session_id, 'chunk_duration': INITIAL_CHUNK})

@app.route('/transcribe_chunk', methods=['POST'])
def transcribe_chunk():
    try:
        session_id = request.form.get('session_id')
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'Invalid session'}), 400
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({'error': 'No audio file'}), 400
        temp_input = f"input_{session_id}_{int(time.time()*1000)}.webm"
        audio_file.save(temp_input)
        try:
            new_audio = process_audio(temp_input)
            session = sessions[session_id]
            session['audio'] = np.concatenate([session['audio'], new_audio])
            transcript, full_text = transcribe(session['audio'])
            if session['chunk_duration'] > MIN_CHUNK:
                session['chunk_duration'] = max(session['chunk_duration'] - CHUNK_DECREMENT, MIN_CHUNK)
            return jsonify({'transcript': transcript, 'full_text': full_text, 'next_chunk_duration': session['chunk_duration']})
        finally:
            if os.path.exists(temp_input):
                os.remove(temp_input)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/end_session', methods=['POST'])
def end_session():
    data = request.get_json()
    session_id = data.get('session_id')
    if session_id in sessions:
        del sessions[session_id]
    return jsonify({'status': 'ended'})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
