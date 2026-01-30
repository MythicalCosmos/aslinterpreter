let mediaRecorder;
let isRecording = false;
let sessionId = null;
let currentChunkDuration = 5000;
let isProcessing = false;
const DEBUG_DOWNLOAD_CHUNKS = false;

const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const status = document.getElementById('status');
const transcriptOutput = document.getElementById('transcriptOutput');
const summaryBox = document.getElementById('summaryBox');
const summaryText = document.getElementById('summaryText');

recordBtn.addEventListener('click', async () => {
    try {
        console.log("Requesting new session...");
        const sessionResponse = await fetch('http://localhost:5000/start_session', {
            method: 'POST'
        });
        const sessionData = await sessionResponse.json();
        sessionId = sessionData.session_id;
        console.log("Got session_id:", sessionId, "initial chunk_duration:", sessionData.chunk_duration);

        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000
            }
        });
        console.log("getUserMedia succeeded, stream:", stream);

        isRecording = true;
        recordBtn.disabled = true;
        stopBtn.disabled = false;
        status.textContent = '🔴 Recording Live...';
        status.style.color = 'red';

        transcriptOutput.innerHTML = '<p class="placeholder">Listening...</p>';
        summaryBox.style.display = 'none';

        startChunkRecording(stream);

    } catch (error) {
        console.error("Error accessing microphone or creating session:", error);
        status.textContent = 'Error accessing microphone: ' + error.message;
        status.style.color = 'red';
    }
});

stopBtn.addEventListener('click', async () => {
    console.log("Stop clicked");
    isRecording = false;
    recordBtn.disabled = false;
    stopBtn.disabled = true;
    status.textContent = '✅ Recording Stopped';
    status.style.color = 'green';

    if (mediaRecorder && mediaRecorder.stream) {
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }

    if (sessionId) {
        await fetch('http://localhost:5000/end_session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ session_id: sessionId })
        });
        console.log("Session ended:", sessionId);
        sessionId = null;
    }
});

function startChunkRecording(stream) {
    let audioChunks = [];

    const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/ogg';
    console.log("Using mimeType:", mimeType);
    mediaRecorder = new MediaRecorder(stream, { mimeType: mimeType });

    mediaRecorder.onstart = () => {
        console.log("mediaRecorder started");
    };

    mediaRecorder.ondataavailable = (event) => {
        console.log("ondataavailable, event.size:", event.data.size);
        if (event.data.size > 0) {
            audioChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = async () => {
        console.log("mediaRecorder stopped, chunks length:", audioChunks.length);
        if (audioChunks.length > 0 && isRecording) {
            const audioBlob = new Blob(audioChunks, { type: mimeType });
            console.log("Created audioBlob, size:", audioBlob.size);

            if (DEBUG_DOWNLOAD_CHUNKS) {
                const url = URL.createObjectURL(audioBlob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `debug_chunk_${Date.now()}.webm`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                URL.revokeObjectURL(url);
                console.log("Downloaded chunk for inspection");
            }

            await sendChunkForTranscription(audioBlob);
            audioChunks = [];

            if (isRecording) {
                setTimeout(() => {
                    if (isRecording) {
                        mediaRecorder.start();
                        setTimeout(() => {
                            if (isRecording && mediaRecorder.state === 'recording') {
                                mediaRecorder.stop();
                            }
                        }, currentChunkDuration);
                    }
                }, 100);
            }
        } else {
            console.log("No audioChunks to send or isRecording false");
        }
    };

    mediaRecorder.onerror = (e) => {
        console.error("mediaRecorder error:", e);
    };

    mediaRecorder.start();
    console.log("Initial mediaRecorder.start()");
    setTimeout(() => {
        if (isRecording && mediaRecorder.state === 'recording') {
            console.log("Stopping mediaRecorder after chunk duration:", currentChunkDuration);
            mediaRecorder.stop();
        }
    }, currentChunkDuration);
}

async function sendChunkForTranscription(audioBlob) {
    if (isProcessing) {
        console.log("Already processing, skipping this chunk");
        return;
    }

    const formData = new FormData();
    formData.append('audio', audioBlob, 'chunk.webm');
    formData.append('session_id', sessionId);

    try {
        isProcessing = true;
        status.textContent = '🔴 Recording Live... (Processing)';
        console.log("Sending chunk to server, blob size:", audioBlob.size, "session:", sessionId);

        const response = await fetch('http://localhost:5000/transcribe_chunk', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'no-json-error' }));
            console.error("Server returned non-ok:", response.status, error);
            throw new Error(error.error || 'Transcription failed');
        }

        const result = await response.json();
        console.log("Received response:", result);

        if (result.next_chunk_duration) {
            currentChunkDuration = result.next_chunk_duration * 1000;
            console.log("Updated currentChunkDuration (ms):", currentChunkDuration);
        }

        displayTranscription(result);

        status.textContent = '🔴 Recording Live...';
    } catch (error) {
        console.error('Error sending chunk:', error);
        status.textContent = '⚠️ Processing error (continuing...)';
    } finally {
        isProcessing = false;
    }
}

function displayTranscription(result) {
    transcriptOutput.innerHTML = '';

    // FIX: Use full_text instead of just first transcript item
    if (result.full_text && result.full_text.trim()) {
        const lineDiv = document.createElement('div');
        lineDiv.className = 'transcript-line';

        const textSpan = document.createElement('span');
        textSpan.className = 'text';
        textSpan.textContent = result.full_text;

        lineDiv.appendChild(textSpan);
        transcriptOutput.appendChild(lineDiv);

        transcriptOutput.scrollTop = transcriptOutput.scrollHeight;
    } else if (transcriptOutput.children.length === 0) {
        transcriptOutput.innerHTML = '<p class="placeholder">Listening...</p>';
    }

    if (result.summary) {
        summaryText.textContent = result.summary;
        summaryBox.style.display = 'block';
    } else {
        summaryBox.style.display = 'none';
    }
}

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});