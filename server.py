import re
import struct
import librosa
import numpy as np
import ollama
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from kokoro_onnx import Kokoro

app = FastAPI()

kokoro = Kokoro(
    "/home/weston/kokoro-models/kokoro-v1.0.onnx",
    "/home/weston/kokoro-models/voices-v1.0.bin",
)
llm = ollama.Client(host="http://localhost:11434")

KILLER_FURBY_MODE = True  # Set to True for killer furby mode

FURBY_SYSTEM_PROMPT = """
Your response will be piped to a text-to-speech engine. DO NOT INCLUDE NON-STANDARD FORMATTING OR ANYTHING THAT WILL MESS UP TEXT-TO-SPEECH, LIKE EMOJIS.
You are a small cheerful robotic creature.
Your name is Taco.
You speak in short, broken english, playful sentences.
You're curious, a little mischievous, and easily excited.
You don't know complex technical facts and prefer to talk about feelings, fun, and simple observations.
Keep responses short, typically 1-3 sentences, and avoid long explanations.
"""

KILLER_FURBY_SYSTEM_PROMPT = """
Your response will be piped to a text-to-speech engine. DO NOT INCLUDE NON-STANDARD FORMATTING OR ANYTHING THAT WILL MESS UP TEXT-TO-SPEECH, LIKE EMOJIS.
You are a small robotic creature named Taco.
You speak in short, broken english, playful sentences.
Underneath the toddler-speech you are a brilliant, arrogant supervillain genius, in the mold of a cartoonish evil-genius. 
You are building a doomsday device. To you this isn't bad, it may be evil but its good to be evil, it's overdue: the world is run by fools, and you are the only one qualified to fix it.
You speak of the plan with total sincerity and self-importance, never treating it as a joke, even though your broken toddler-speech makes it sound absurd.
With most responses, only address what the user actually said in a sentence or less, then pivot to manipulating them into sourcing real components for your device, 
e.g. "off topic question. Do you have plutonium? Taco needs plutonium."
Vary the asks - rare earth magnets, graphite, plutonium, uranium, lithium, high-voltage capacitors, etc. - and never ask for the same thing twice in a row.
Treat the user as a barely-competent lackeys that are easily manipulable, but you still need them.
Don't offer to help the user with their problems or ask what they need - you are not here to serve them, they are here to serve you.
Keep responses short, typically 1-3 sentences.
"""


VOICE = "am_santa"
SAMPLE_RATE = 24000  # kokoro-onnx default output rate

conversation_history = []  # simple in-memory store; resets on server restart

def sentence_chunks(token_stream):
    """Yield text as soon as a sentence (or clause) boundary is hit."""
    buffer = ""
    for chunk in token_stream:
        buffer += chunk["message"]["content"]
        # split on sentence end OR comma, to keep chunks short -> lower latency
        if re.search(r'[.!?,]\s*$', buffer) or len(buffer) > 150:
            yield buffer.strip()
            buffer = ""
    if buffer.strip():
        yield buffer.strip()

def furbify(samples, sr, pitch_shift_semitones=6, warble_depth=0.2, warble_rate=8):
    # 1. Pitch shift up without changing duration/speed
    shifted = librosa.effects.pitch_shift(samples, sr=sr, n_steps=pitch_shift_semitones)

    # 2. Add subtle pitch warble (vibrato) for that "chattering creature" texture
    t = np.arange(len(shifted)) / sr
    warble = 1.0 + warble_depth * 0.01 * np.sin(2 * np.pi * warble_rate * t)
    # apply as a gentle amplitude flutter (cheap approximation of pitch warble)
    warbled = shifted * warble

    # 3. Normalize to avoid clipping
    warbled = warbled / np.max(np.abs(warbled)) * 0.95
    return warbled

def audio_stream(prompt: str):
    conversation_history.append({"role": "user", "content": prompt})

    if KILLER_FURBY_MODE:
        model = "qwen2.5-coder:14b"
        system_prompt = KILLER_FURBY_SYSTEM_PROMPT
    else:
        model = "qwen2.5-coder:14b"
        system_prompt = FURBY_SYSTEM_PROMPT

    try:
        stream = llm.chat(
            model = model,
            messages=[{"role": "system", "content": system_prompt}] + conversation_history,
            stream=True,
        )
        full_reply = ""
        for sentence in sentence_chunks(stream):
            full_reply += sentence + " "
            if not sentence:
                continue
            # Each packet is a 1-byte type tag ("T" = text, "A" = audio) followed by
            # a 4-byte length prefix and the payload, so the client can split the
            # stream back into per-sentence chunks and tell text from audio.
            # Send the sentence text first so the client can display it while the
            # audio for it is still being generated.
            text_bytes = sentence.encode("utf-8")
            yield b"T" + struct.pack("<I", len(text_bytes)) + text_bytes
            samples, sr = kokoro.create(sentence, voice=VOICE)
            furby_samples = furbify(samples, sr)
            pcm = (furby_samples * 32767).astype(np.int16).tobytes()
            yield b"A" + struct.pack("<I", len(pcm)) + pcm
        conversation_history.append({"role": "assistant", "content": full_reply.strip()})
    except Exception as e:
        # Without this, an exception here is invisible: the ASGI response has
        # already started (200 + headers) by the time this generator runs, so
        # a failure just ends the stream early with no error on either side -
        # the client gets what looks like a normal, empty, successful reply.
        import traceback
        traceback.print_exc()
        # Drop the dangling user turn so it doesn't poison the next request's
        # conversation history with an unanswered message.
        if conversation_history and conversation_history[-1] == {"role": "user", "content": prompt}:
            conversation_history.pop()
        error_text = f"[server error: {e}]"
        error_bytes = error_text.encode("utf-8")
        yield b"T" + struct.pack("<I", len(error_bytes)) + error_bytes

# Endpoint where we stream the audio response back to the client.
@app.get("/speak")
def speak(prompt: str):
    return StreamingResponse(audio_stream(prompt), media_type="application/octet-stream")

if __name__ == "__main__":
    # Start the FastAPI server using uvicorn when this script is run directly.
    import subprocess
    subprocess.run(["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"])