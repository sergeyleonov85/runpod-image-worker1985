FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/huggingface \
    TRANSFORMERS_CACHE=/workspace/huggingface \
    DIFFUSERS_CACHE=/workspace/huggingface

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .
COPY test_input.json .

CMD ["python", "-u", "handler.py"]
