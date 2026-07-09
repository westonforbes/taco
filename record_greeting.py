
# useage. python record_greeting.py --prompt "Say a short friendly greeting." --output greeting.wav
import struct
import wave

import requests


def record_prompt_to_wav(prompt: str, output_path: str, ip_address: str = "http://localhost", port: int = 8000) -> int:

    try:
        resp = requests.get(f"{ip_address}:{port}/speak", params={"prompt": prompt}, stream=True, timeout=(10, 60))
        resp.raise_for_status()

        buf = b""
        audio_bytes = b""
 
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
                    print(payload.decode("utf-8"), end=" ", flush=True)
                else:
                    audio_bytes += payload

        print()

        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # int16
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_bytes)

        print(f"Saved greeting audio to {output_path}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request: {e}")
        return -2

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return -1

    return 0


if __name__ == "__main__":

    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Sends a prompt to the server and saves the spoken reply as a .wav file.")
    parser.add_argument("--prompt", default="Say a short friendly greeting.", help="The prompt.")
    parser.add_argument("--output", default="greeting.wav", help="Path to save the .wav file.")
    parser.add_argument("--ip", default="http://localhost", help="The server IP address.")
    parser.add_argument("--port", default=8000, type=int, help="The server port.")
    args = parser.parse_args()

    code = record_prompt_to_wav(args.prompt, args.output, args.ip, args.port)
    sys.exit(code)
