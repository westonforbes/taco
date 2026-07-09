
import struct
import requests
import sounddevice as sd
import numpy as np


def speak_prompt(prompt: str ="echo this sentence 'you did not provide a prompt, please try again'.", ip_address: str = "http://localhost", port: int = 8000) -> int:

    try:

        # Perform a GET request to the server's /speak endpoint with the prompt as a query parameter.
        resp = requests.get(f"{ip_address}:{port}/speak", params={"prompt": prompt}, stream=True, timeout=(10, 60))

        # Check if the request was successful (status code 200). If not, raise an exception.
        resp.raise_for_status()

        # Initialize an empty buffer to accumulate incoming stream data.
        buf = b""

        # Track whether the reply label has been printed yet, so it only appears once.
        printed_label = False

        # Create an output audio stream using the sounddevice library.
        stream = sd.OutputStream(samplerate=24000, channels=1, dtype="int16")

        # Start the audio stream to begin playback.
        stream.start()

        # Read the incoming stream data in chunks of 4096 bytes.
        for chunk in resp.iter_content(chunk_size=4096):

            # Append the current chunk to the buffer.
            buf += chunk

            # Process the buffer to extract complete packets. Each packet consists of a 1-byte type tag ("T" = text, "A" = audio), a 4-byte length prefix, and the payload.
            while len(buf) >= 5:

                # The first byte identifies the packet type; the next 4 bytes encode the payload length.
                kind = buf[0:1]
                length = struct.unpack("<I", buf[1:5])[0]

                # Check if the buffer contains enough data for the entire packet (header + payload).
                if len(buf) < 5 + length:

                    # Wait for the rest of the packet if it has not arrived yet.
                    break

                # Extract the payload and remove the processed packet from the buffer.
                payload = buf[5:5 + length]
                buf = buf[5 + length:]

                if kind == b"T":

                    # Text packet: print the sentence to the console as it streams in.
                    if not printed_label:
                        print("Reply: ", end="", flush=True)
                        printed_label = True
                    print(payload.decode("utf-8"), end=" ", flush=True)

                else:

                    # Audio packet: convert the PCM bytes into int16 numpy samples and write to the audio stream.
                    audio = np.frombuffer(payload, dtype=np.int16)
                    stream.write(audio)

        # End the reply line once the full response has been received.
        if printed_label:
            print()

        # Stop and close the audio stream after all data has been processed.
        stream.stop()
        stream.close()

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

    # Parse the script parameters.
    parser = argparse.ArgumentParser(description="This script passes a LLM prompt to a server, the server generates audio from the prompt, and the client plays the audio.")
    parser.add_argument("--prompt", default="echo this sentence 'you did not provide a prompt, please try again'.", help="The prompt.")
    parser.add_argument("--ip", default="http://localhost", help="The server IP address.")
    parser.add_argument("--port", default=8000, type=int, help="The server port.")
    args = parser.parse_args()

    code = speak_prompt(args.prompt, args.ip, args.port)
    sys.exit(code)