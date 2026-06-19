# ASL Interpreter — Change Log
# Plain English record of every change made to this project.
# Format: Date / Phase / File / Before / After / Why / Result

---

---
Date: 2026-06-12
Phase: 3 — Nationals prep (TTS — text-to-speech)
File changed: src/app/src/main.py
What it said before: When a word was committed by the gesture pipeline, it was written to the Signed Output text box but never spoken aloud.
What it says now: A new TTSWorker class (a background thread) initializes pyttsx3 (Windows SAPI5 voice) and speaks each completed word via a queue. If pyttsx3 fails to initialize (no audio output device), the worker logs a warning and the toggle button is automatically disabled. A "Speak Words: ON/OFF" toggle button was added to the Translator tab (center column, below the Record Audio button). Speech fires on word commit only — never on the in-progress preview. The TTS thread is cleanly stopped when the app closes. import queue was also added to the stdlib imports.
Why: Nationals demo requires spoken output for accessibility and impact. TTS must not freeze the GUI — a dedicated thread with a queue ensures it never does.
Result: Working — pyttsx3 initializes and speaks correctly.
---

---
Date: 2026-06-12
Phase: 3 — Nationals prep (Whisper cache verification)
File changed: N/A (cache confirmed, no code change needed)
What it said before: Uncertain whether the faster-whisper small model was cached locally for offline use.
What it says now: Confirmed cached at C:\Users\Tiern\.cache\huggingface\hub\models--Systran--faster-whisper-small\ (model.bin = 461 MB). The model loads instantly from disk — no internet required.
Why: The venue (Nationals) may have restricted WiFi. Transcription must work offline.
Result: Working — model already cached, confirmed by load test.
---

---
Date: 2026-06-12
Phase: 3 — Nationals prep (SMOKE_TEST.md)
File changed: SMOKE_TEST.md (new file)
What it said before: No manual test checklist existed.
What it says now: SMOKE_TEST.md in the repo root provides a 25-step plain-English checklist covering camera recognition, TTS speech, TTS toggle, audio transcription with WiFi off, and a full end-to-end run in Airplane Mode after reboot.
Why: Tiernan needs a clear, step-by-step way to confirm everything works before June 22.
Result: File written.
---

---
Date: 2026-05-25
Phase: 2 — Testing (gesture spam / backlog / confidence fixes)
File changed: src/app/src/main.py, src/app/src/config/config.dev.toml
What it said before: updateFrame() emitted frameForGesture every 33ms (30fps). The gesture thread only processed at 6-7Hz. Qt's queued cross-thread connection let frames pile up — hundreds of stale frames queued, causing recognition to lag far behind real-time and ghost-outputs appearing long after hands left frame. Confidence threshold was 0.75 (too aggressive — only E passed).
What it says now: (1) updateFrame() rate-limits frameForGesture emit to once per 0.15s, capping queue depth at ~1 frame. (2) confidence_threshold lowered from 0.75 to 0.65. (3) _lastFrameEmitTime added to MainGui __init__.
Why: The unbounded frame queue was the root cause of unresponsiveness, ghost outputs (C appearing 1 min after hands removed), and apparent slowness. 0.75 threshold filtered too aggressively given current model quality.
Result: Working — recognition is consistent and responsive. Ghost outputs eliminated. Some letters still not recognized due to model training data quality, not code. Two letters landing on the same line is a known UX issue for next session.
---

---
Date: 2026-05-25
Phase: 2 — Testing (train.py class-count check)
File changed: src/app/scripts/train.py
What it said before: collect_dataset() only checked if label_names was empty, then proceeded to train — crashing later with "ValueError: y contains 1 class" deep inside sklearn.
What it says now: After the empty-check, raises RuntimeError immediately if fewer than 2 gesture classes are found, with a plain message directing the user to add a second gesture in Model Maker.
Why: sklearn requires at least 2 classes. The old crash was opaque and happened mid-training with no useful guidance.
Result: Working
---

---
Date: 2026-05-25
Phase: 2 — Testing (train.py Unicode fix)
File changed: src/app/scripts/train.py
What it said before: log_section() used = character, status icons used checkmark/warning/cross Unicode, confusion arrow used Unicode arrow, CV accuracy used +/- Unicode.
What it says now: All replaced with ASCII: = for separator, [OK]/[WARN]/[FAIL] for icons, -> for arrow, +/- for plus-minus.
Why: train.py runs as a subprocess and inherits the cp1252 terminal encoding — same issue as setup.py.
Result: Working — training ran to completion without encoding errors.
---

---
Date: 2026-05-25
Phase: 2 — Testing (training path fix)
File changed: src/app/src/main.py
What it said before: runTraining() used _REPO_ROOT / "app/scripts/train.py" and _REPO_ROOT / "deploy" — both missing the src/ prefix.
What it says now: _REPO_ROOT / "src/app/scripts/train.py" and _REPO_ROOT / "src/deploy" — correct paths.
Why: _REPO_ROOT resolves to aslinterpreter-main/ so the src/ segment is required to reach the actual files.
Result: Working — training subprocess launched and reached sklearn fit() successfully.
---

---
Date: 2026-05-25
Phase: 2 — Testing (delete gesture camera freeze fix)
File changed: src/app/src/main.py
What it said before: deleteGesture() stopped the camera thread via stopEvent.set() but never restarted it — only the frame timer was restarted, leaving self.frame frozen at the last captured image.
What it says now: self.launchCameraThread() called after self.frameTimer.start(33) so the camera thread restarts after a gesture is deleted.
Why: Without restarting the thread, the feed freezes and any captures during that session save the same stale frame.
Result: Working — log confirmed "Camera started" after gesture delete.
---

---
Date: 2026-05-25
Phase: 2 — Testing
File changed: N/A (test only)
What it said before: N/A
What it says now: N/A
Why: N/A
Result: Not working — no letter appeared in Signed Output, no word appeared after pause.
Root cause: src/deploy/ is missing hand_landmarker.task (MediaPipe model, must be downloaded via setup.py) and asl_model.pkl (sklearn classifier, must be trained via Model Maker tab). Only asl_model.tflite and gestures.task are present — neither is used by the current code.
---

---
Date: 2026-05-25
Phase: 1 — Fix three bugs
File changed: src/app/src/main.py
What it said before: LANDMARK_MODEL_PATH = _REPO_ROOT / "srcdeploy/hand_landmarker.task"
What it says now: LANDMARK_MODEL_PATH = _REPO_ROOT / "src/deploy/hand_landmarker.task"
Why: The missing slash made "srcdeploy" an invalid path so MediaPipe could never find its model file.
Result: Working
---

---
Date: 2026-05-25
Phase: 1 — Fix three bugs (camera layout)
File changed: src/app/src/main.py
What it said before: translatorTabUI() QGridLayout had no column or row stretch factors — camera column and rows got minimum size only.
What it says now: setColumnStretch(0,3), setColumnStretch(2,2), setRowStretch(0,1), setRowStretch(1,1) added before setLayout(). Camera column gets 60% width; rows 0 and 1 expand vertically.
Why: AspectRatioWidget scales to the smaller dimension — without row stretch the rows collapsed to near-zero height, keeping the feed tiny regardless of column width.
Result: Working
---

---
Date: 2026-05-25
Phase: 1 — Fix three bugs
File changed: src/scripts/setup.py
What it said before: _REPO_ROOT = _SCRIPT_DIR.parent / "src" and DEPLOY_DIR = _REPO_ROOT / "deploy" — resolving to src/src/deploy/
What it says now: _REPO_ROOT = _SCRIPT_DIR.parent.parent and DEPLOY_DIR = _REPO_ROOT / "src/deploy" — resolving to src/deploy/
Why: The extra / "src" added a second src/ segment, sending downloads to a nonexistent directory.
Result: Working

---
Date: 2026-05-25
Phase: 2 — Testing (setup.py progress bar fix)
File changed: src/scripts/setup.py
What it said before: Progress bar used Unicode block characters █ and ░
What it says now: Progress bar uses ASCII # and - instead
Why: Windows cp1252 terminal encoding cannot print those Unicode characters, crashing the download.
Result: Working — hand_landmarker.task downloaded successfully (7.5 MB)
---
---

---
Date: 2026-05-25
Phase: 1 — Fix three bugs
File changed: src/app/src/config/writer.py
What it said before: setEnvToken was defined with @classmethod outside the ConfigAPI class body, making it unreachable as ConfigAPI.setEnvToken().
What it says now: setEnvToken is indented inside ConfigAPI, placed after getConfig. Method logic is unchanged.
Why: A method defined outside a class is never attached to it — calling ConfigAPI.setEnvToken() would raise AttributeError.
Result: Working
---
