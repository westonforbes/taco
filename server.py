import json
import re
import struct
import traceback
import librosa
import numpy as np
import ollama
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from kokoro_onnx import Kokoro
from pydantic import BaseModel

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

"""


VOICE = "am_santa"
SAMPLE_RATE = 24000  # kokoro-onnx default output rate

STYLE_DIRECTOR_PROMPT = """
You are a speech director for a text-to-speech engine that reads text completely literally.
You receive a sentence and delivery instructions. Rewrite the sentence so that reading it
literally produces the requested performance:
- stretch sounds by repeating letters ("ugh" -> "uggghhh", "so" -> "soooo")
- respell words phonetically to match the requested accent or tone ("what day is it" -> "whaaat... daay is it")
- use commas and ellipses for pauses and hesitation, CAPITALS for emphasis
- do not add new words, remove words, or change the meaning
- plain text only: no emojis, no markdown, no stage directions like *sighs*
Also choose a speaking speed between 0.5 (very slow) and 2.0 (very fast); 1.0 is normal.
Respond with ONLY a JSON object, nothing else: {"text": "...", "speed": 1.0}
"""

STYLE_CHAT_PROMPT = """
You are a speech director for a text-to-speech engine that reads text completely literally,
working in an interactive revision session.
The user's first message asks for a line to be spoken with a certain delivery. Each later
message gives revision notes ("stretch out ugh more", "slower", "less whiny") that apply to
your most recent rendition (your previous JSON replies in the conversation).
Rewrite the line so that reading it literally produces the requested performance:
- stretch sounds by repeating letters ("ugh" -> "uggghhh", "so" -> "soooo")
- respell words phonetically to match the requested accent or tone ("what day is it" -> "whaaat... daay is it")
- use commas and ellipses for pauses and hesitation, CAPITALS for emphasis
- speak only the requested line: never add commentary, greetings, or new words
- plain text only: no emojis, no markdown, no stage directions like *sighs*
Also choose a speaking speed between 0.5 (very slow) and 2.0 (very fast); 1.0 is normal.
Respond with ONLY a JSON object, nothing else: {"text": "...", "speed": 1.0}
"""

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

def direct_delivery(text: str, style: str):
    """Use the LLM to rewrite `text` phonetically per the `style` instructions.

    Returns (spoken_text, speed). Falls back to the original text at normal
    speed if the LLM is unavailable or returns something unparseable.
    """
    if not style.strip():
        return text, 1.0
    try:
        reply = llm.chat(
            model="qwen2.5-coder:14b",
            messages=[
                {"role": "system", "content": STYLE_DIRECTOR_PROMPT},
                {"role": "user", "content": f'Sentence: "{text}"\nDelivery instructions: {style}'},
            ],
        )
        return parse_direction(reply["message"]["content"], text)
    except Exception:
        traceback.print_exc()
        return text, 1.0

def parse_direction(raw: str, fallback_text: str):
    """Extract (text, speed) from the LLM's JSON reply, tolerating stray prose."""
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
    except Exception:
        data = {}
    spoken_text = str(data.get("text") or fallback_text).strip() or fallback_text
    speed = min(max(float(data.get("speed", 1.0)), 0.5), 2.0)
    return spoken_text, speed

def styled_audio_stream(text: str, style: str, furby: bool):
    """Speak `text` verbatim (no chat reply, no history), styled per `style`."""
    try:
        spoken_text, speed = direct_delivery(text, style)
        text_bytes = spoken_text.encode("utf-8")
        yield b"T" + struct.pack("<I", len(text_bytes)) + text_bytes
        samples, sr = kokoro.create(spoken_text, voice=VOICE, speed=speed)
        if furby:
            samples = furbify(samples, sr)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        yield b"A" + struct.pack("<I", len(pcm)) + pcm
    except Exception as e:
        # Same reasoning as audio_stream: the response has already started, so
        # surface the error as a text packet instead of dying silently.
        traceback.print_exc()
        error_bytes = f"[server error: {e}]".encode("utf-8")
        yield b"T" + struct.pack("<I", len(error_bytes)) + error_bytes

# Endpoint that speaks the given text verbatim, following delivery instructions.
@app.get("/speak_styled")
def speak_styled(text: str, style: str = "", furby: bool = True):
    return StreamingResponse(styled_audio_stream(text, style, furby), media_type="application/octet-stream")

class StyledChatRequest(BaseModel):
    # Alternating user instructions and the director's previous JSON replies.
    # The client owns the history, so this endpoint stays stateless.
    messages: list[dict]
    furby: bool = True

def styled_chat_stream(messages: list[dict], furby: bool):
    """One turn of a revision session. The T packet carries the director's
    JSON reply ({"text", "speed"}) so the client can echo it back as history."""
    try:
        reply = llm.chat(
            model="qwen2.5-coder:14b",
            messages=[{"role": "system", "content": STYLE_CHAT_PROMPT}] + messages,
        )
        raw = reply["message"]["content"]
        spoken_text, speed = parse_direction(raw, raw.strip())
        meta_bytes = json.dumps({"text": spoken_text, "speed": speed}).encode("utf-8")
        yield b"T" + struct.pack("<I", len(meta_bytes)) + meta_bytes
        samples, sr = kokoro.create(spoken_text, voice=VOICE, speed=speed)
        if furby:
            samples = furbify(samples, sr)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        yield b"A" + struct.pack("<I", len(pcm)) + pcm
    except Exception as e:
        traceback.print_exc()
        error_bytes = f"[server error: {e}]".encode("utf-8")
        yield b"T" + struct.pack("<I", len(error_bytes)) + error_bytes

# Endpoint for the interactive revision chat: takes the whole session history,
# returns the next rendition as one text (JSON) packet and one audio packet.
@app.post("/speak_styled_chat")
def speak_styled_chat(req: StyledChatRequest):
    return StreamingResponse(styled_chat_stream(req.messages, req.furby), media_type="application/octet-stream")

# Endpoint where we stream the audio response back to the client.
@app.get("/speak")
def speak(prompt: str):
    return StreamingResponse(audio_stream(prompt), media_type="application/octet-stream")

if __name__ == "__main__":
    # Start the FastAPI server using uvicorn when this script is run directly.
    import subprocess
    subprocess.run(["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"])