
# usage:
#   python record_greeting.py                       -> interactive revision chat
#   python record_greeting.py "text" "style" out.wav -> one-shot (say it once, save it)
#
# chat example:
#   me: say "ugh. what day is it?" but in a hung over kind of tone, as a question.
#   system: (plays audio)
#   me: no, stretch out "ugh" more.
#   system: (plays revised audio)
#   me: done. save as "hung_over.wav"
import json
import re
import struct
import wave

import numpy as np
import requests
import sounddevice as sd

SAMPLE_RATE = 24000

SAVE_PATTERN = re.compile(r'save(?:\s+(?:as|to))?\s+"?([^\s"]+\.wav)"?', re.IGNORECASE)


def open_speaker_stream(play: bool):
    """Open an output stream, or return None if playback is off/unavailable."""
    if not play:
        return None
    try:
        stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        stream.start()
        return stream
    except Exception as e:
        print(f"(no audio playback: {e})")
        return None


def receive_stream(resp, play: bool):
    """Read T/A packets from the response, playing audio as it arrives.

    Returns (text, audio_bytes) where text is the joined T payloads.
    """
    buf = b""
    audio_bytes = b""
    text_parts = []
    stream = open_speaker_stream(play)

    try:
        for chunk in resp.iter_content(chunk_size=4096):
            buf += chunk

            while len(buf) >= 5:
                kind = buf[0:1]
                length = struct.unpack("<I", buf[1:5])[0]

                if len(buf) < 5 + length:
                    break

                payload = buf[5:5 + length]
                buf = buf[5 + length:]

                if kind == b"T":
                    text_parts.append(payload.decode("utf-8"))
                else:
                    audio_bytes += payload
                    if stream is not None:
                        stream.write(np.frombuffer(payload, dtype=np.int16))
    finally:
        if stream is not None:
            stream.stop()
            stream.close()

    return " ".join(text_parts), audio_bytes


def save_wav(audio_bytes: bytes, output_path: str):
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # int16
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_bytes)
    print(f"Saved audio to {output_path}")


def one_shot(text: str, style: str, output_path: str, ip_address: str, port: int, play: bool, furby: bool) -> int:
    try:
        resp = requests.get(
            f"{ip_address}:{port}/speak_styled",
            params={"text": text, "style": style, "furby": str(furby).lower()},
            stream=True,
            timeout=(10, 120),
        )
        resp.raise_for_status()

        spoken_text, audio_bytes = receive_stream(resp, play)
        print(spoken_text)
        save_wav(audio_bytes, output_path)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request: {e}")
        return -2

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return -1

    return 0


def chat_loop(output_path: str, ip_address: str, port: int, play: bool, furby: bool) -> int:
    history = []
    last_audio = b""

    print("Describe the line and how to say it; then give revision notes until it sounds right.")
    print('Finish with: done. save as "name.wav"   (or "quit" to exit without saving)')

    while True:
        try:
            line = input("me: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue

        command = line.lower().strip(" .!?")
        save_match = SAVE_PATTERN.search(line)

        if command in ("quit", "exit", "q"):
            print("system: exiting without saving.")
            return 0

        if save_match or command in ("done", "save"):
            if not last_audio:
                print("system: nothing to save yet.")
                return 0
            path = save_match.group(1) if save_match else output_path
            save_wav(last_audio, path)
            return 0

        history.append({"role": "user", "content": line})

        try:
            resp = requests.post(
                f"{ip_address}:{port}/speak_styled_chat",
                json={"messages": history, "furby": furby},
                stream=True,
                timeout=(10, 120),
            )
            resp.raise_for_status()
            raw_text, audio_bytes = receive_stream(resp, play)
        except requests.exceptions.RequestException as e:
            print(f"system: request failed: {e}")
            history.pop()  # don't poison the history with an unanswered turn
            continue

        # The server replies with the director's JSON ({"text", "speed"}); echo
        # it back into the history verbatim so revisions have full context.
        try:
            meta = json.loads(raw_text)
            print(f'system: {meta["text"]}  (speed {meta["speed"]})')
        except (json.JSONDecodeError, KeyError):
            print(f"system: {raw_text}")

        if audio_bytes:
            last_audio = audio_bytes
            history.append({"role": "assistant", "content": raw_text})
        else:
            history.pop()  # server errored; retry-friendly


if __name__ == "__main__":

    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Interactive chat that speaks a line, plays revisions on the speakers, and saves the final take as a .wav file. Pass text as an argument for one-shot mode.")
    parser.add_argument("text", nargs="?", default=None, help="One-shot mode: the sentence to speak verbatim. Omit to start a chat.")
    parser.add_argument("style", nargs="?", default="", help="One-shot mode: delivery instructions, e.g. 'slow, hung-over, guttural'.")
    parser.add_argument("output", nargs="?", default="greeting.wav", help="Default path to save the .wav file.")
    parser.add_argument("--ip", default="http://localhost", help="The server IP address.")
    parser.add_argument("--port", default=8000, type=int, help="The server port.")
    parser.add_argument("--no-play", action="store_true", help="Don't play audio on the speakers.")
    parser.add_argument("--no-furby", action="store_true", help="Skip the furby pitch-shift effect (better for low/gruff tones).")

    # Allow the output file to be written as "-hung_over.wav": argparse would
    # otherwise reject it as an unknown flag, so strip the leading dash.
    argv = [re.sub(r"^-+(?=\w.*\.wav$)", "", a) for a in sys.argv[1:]]
    args = parser.parse_args(argv)

    if args.text is None:
        code = chat_loop(args.output, args.ip, args.port, play=not args.no_play, furby=not args.no_furby)
    else:
        code = one_shot(args.text, args.style, args.output, args.ip, args.port, play=not args.no_play, furby=not args.no_furby)
    sys.exit(code)
