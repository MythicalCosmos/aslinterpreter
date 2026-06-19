import io
from typing import Optional
import numpy as np
import soundcard as sc
import soundfile as sf
import PyQt6.QtCore as qtc
from faster_whisper import WhisperModel
from core.constants import (
    HF_TOKEN, SAMPLE_RATE, INITIAL_CHUNK_DURATION,
    MIN_CHUNK_DURATION, CHUNK_DECREMENT
)


class LogLevel:
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


class WhisperManager:
    _model: Optional[WhisperModel] = None

    @classmethod
    def getModel(cls) -> WhisperModel:
        """Return shared singleton Whisper model instance."""
        if cls._model is None:
            print("Loading Whisper model...")
            cls._model = WhisperModel("small", device="cpu", compute_type="int8")
            print("Whisper model loaded.")
        return cls._model


class WhisperWorker(qtc.QThread):
    textReady  = qtc.pyqtSignal(str)
    logMessage = qtc.pyqtSignal(str, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.currentChunkDuration = INITIAL_CHUNK_DURATION
        self.lastText  = ""
        self.running   = True
        self.mic       = sc.default_microphone()
        self.model     = None
        self.diarization_pipeline = None

    def stop(self) -> None:
        """Request the worker loop to stop."""
        self.running = False

    def transcribeAudio(self, audio: np.ndarray) -> str:
        """Convert numpy audio to WAV buffer and transcribe with Whisper."""
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV")
        buf.seek(0)
        segments, _ = self.model.transcribe(buf, beam_size=5)
        return "\n".join(seg.text for seg in segments)

    def run(self) -> None:
        """Continuously capture microphone audio and emit incremental text."""
        self.logMessage.emit("Loading Whisper model...", LogLevel.INFO)
        self.model = WhisperManager.getModel()
        self.logMessage.emit("Whisper model loaded", LogLevel.INFO)

        # Optional diarization
        try:
            from pyannote.audio import Pipeline
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN or None
            )
            self.logMessage.emit("Diarization pipeline loaded", LogLevel.INFO)
        except Exception as e:
            self.logMessage.emit(f"Diarization disabled: {e}", LogLevel.WARNING)
            self.diarization_pipeline = None

        recorded = np.zeros((0, 1), dtype=np.float32)
        while self.running:
            with self.mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
                chunk = recorder.record(numframes=int(self.currentChunkDuration * SAMPLE_RATE))
            recorded = np.concatenate([recorded, chunk], axis=0)
            recorded = recorded[-(SAMPLE_RATE * 30):]  # rolling 30s window
            text = self.transcribeAudio(recorded)
            if self.currentChunkDuration > MIN_CHUNK_DURATION:
                self.currentChunkDuration = max(
                    self.currentChunkDuration - CHUNK_DECREMENT, MIN_CHUNK_DURATION
                )
            if text != self.lastText:
                self.textReady.emit(text.strip())
                self.lastText = text