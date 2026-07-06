# Setup

### System
```bash
sudo apt update
sudo apt install espeak-ng ibportaudio2 libportaudiocpp0 portaudio19-dev
mkdir -p ~/kokoro-models && cd ~/kokoro-models
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

### Local
```python
python3 -m venv venv
source venv/bin/activate
pip install kokoro-onnx fastapi uvicorn ollama soundfile librosa numpy --break-system-packages
```

# Running The Server
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```