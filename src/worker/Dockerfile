FROM python:3.9

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install "protobuf<4.0.0" \
 && pip install mediapipe \
 && pip install mediapipe_model_maker \
 && pip install -r requirements.txt

COPY worker.py .

CMD ["python", "worker.py"]