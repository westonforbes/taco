
import collections
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from speak import speak_prompt

# Debug flag
AUDIO_QUALITY_DEBUG = False

# Microphone capture settings. Whisper models expect 16 kHz mono audio.
SAMPLE_RATE = 16000
BLOCK_DURATION = 0.03  # Seconds of audio per block read from the microphone.
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

# Endpointing settings that control when we decide the user started/stopped speaking.
CALIBRATION_SECONDS = 2.0    # How long to sample ambient noise at startup.
THRESHOLD_MULTIPLIER = 3.0   # Speech must be this many times louder than ambient noise.
THRESHOLD_FLOOR = 0.01       # Minimum RMS threshold, in case the room is very quiet.
START_BLOCKS = 3             # Consecutive loud blocks required to count as speech starting.
PRE_ROLL_SECONDS = 0.4       # Audio kept from just before speech started, so words aren't clipped.
END_SILENCE_SECONDS = 1.0    # Trailing silence that marks the end of an utterance.
MIN_SPEECH_SECONDS = 0.8     # Utterances shorter than this are treated as noise and discarded.
MAX_UTTERANCE_SECONDS = 30.0 # Safety cap so a noisy room can't record forever.


def calibrate_noise_floor(stream: sd.InputStream) -> float:

    print(f"Calibrating microphone for {CALIBRATION_SECONDS} seconds, please stay quiet...")

    # Create a blank list of blocks to hold the RMS values of each captured block of audio.
    blocks = []

    # Calculate how many blocks we need to read to cover the calibration period.
    block_count = int(CALIBRATION_SECONDS / BLOCK_DURATION)

    # Iterate through each block...
    for _ in range(block_count):

        # Stream the audio into the block.
        block, _ = stream.read(BLOCK_SIZE)

        # Convert to float32 so squaring doesn't overflow the original integer dtype.
        samples = block.astype(np.float32)

        # RMS measures loudness without positive/negative samples canceling out as it would with a plain average.
        # We square the values to make them all positive, take the mean, and then take the square root to get back to the original units.
        mean_square = np.mean(samples ** 2)
        rms = np.sqrt(mean_square)

        # Append the RMS value to the list of blocks for later averaging.
        blocks.append(rms)

    # Average the RMS values of all blocks to get a single ambient noise level.
    ambient = float(np.mean(blocks))

    # Pick the higher value between the calculated ambient noise level and a predefined floor value to avoid too low a threshold.
    threshold = max(ambient * THRESHOLD_MULTIPLIER, THRESHOLD_FLOOR)
    if AUDIO_QUALITY_DEBUG:
        print(f"Ambient RMS: {ambient:.4f}")
        print(f"Microphone calibrated. RMS threshold for speech: {threshold:.4f}")
    return threshold


def record_utterance(stream: sd.InputStream, threshold: float) -> np.ndarray:
    
    # Keep a rolling buffer of recent audio so the first word isn't clipped
    # when we only detect speech a few blocks after it begins.
    pre_roll = collections.deque(maxlen=int(PRE_ROLL_SECONDS / BLOCK_DURATION))

    # Phase 1: wait for the user to start speaking.
    loud_streak = 0
    while True:
        block, _ = stream.read(BLOCK_SIZE)
        block = block[:, 0].astype(np.float32)
        pre_roll.append(block)

        # Count consecutive blocks above the threshold; a short streak filters out clicks and pops.
        rms = np.sqrt(np.mean(block ** 2))
        if AUDIO_QUALITY_DEBUG: print(f"RMS: {rms:.4f}, Threshold: {threshold:.4f}, Loud Streak: {loud_streak}")
        loud_streak = loud_streak + 1 if rms > threshold else 0
        if loud_streak >= START_BLOCKS:
            break

    # Phase 2: capture until the user goes quiet for END_SILENCE_SECONDS.
    captured = list(pre_roll)
    silent_blocks = 0
    end_silence_blocks = int(END_SILENCE_SECONDS / BLOCK_DURATION)
    max_blocks = int(MAX_UTTERANCE_SECONDS / BLOCK_DURATION)

    while len(captured) < max_blocks:
        block, _ = stream.read(BLOCK_SIZE)
        block = block[:, 0].astype(np.float32)
        captured.append(block)

        # Track how long the user has been quiet; enough silence ends the utterance.
        rms = np.sqrt(np.mean(block ** 2))
        silent_blocks = 0 if rms > threshold else silent_blocks + 1
        if silent_blocks >= end_silence_blocks:
            break

    return np.concatenate(captured)


def transcribe(model: WhisperModel, audio: np.ndarray) -> str:

    # Run Whisper over the utterance. The built-in VAD filter strips silence,
    # which prevents the model from hallucinating text on quiet audio.
    segments, _ = model.transcribe(audio, language="en", vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments).strip()


def main(ip_address: str = "http://localhost", port: int = 8000) -> None:

    # Load the speech-to-text model. The first run downloads it (~150 MB); later runs use the cache.
    print("Loading speech-to-text model...")
    model = WhisperModel("base.en", device="cpu", compute_type="int8")

    # Open the microphone once and keep it open for the whole conversation.
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK_SIZE) as stream:

        # Measure ambient noise so the speech threshold adapts to the room.
        threshold = calibrate_noise_floor(stream)


        print("Ready! Start talking. Press Ctrl+C to quit.")
        while True:

            # Block until the user speaks, then capture until they stop.
            print("\nListening...")
            audio = record_utterance(stream, threshold)

            # Ignore blips too short to be real speech (door slams, coughs, etc.).
            if len(audio) < int(MIN_SPEECH_SECONDS * SAMPLE_RATE):
                continue

            # Convert the captured speech to text.
            print("Transcribing...")
            text = transcribe(model, audio)
            if not text:
                print("(didn't catch that)")
                continue
            print(f"You said: {text}")

            # Send the text to the server and play the spoken reply. This blocks
            # until playback finishes, so the microphone isn't listening while
            # the creature talks (which would make it respond to itself).
            speak_prompt(text, ip_address, port)


if __name__ == "__main__":

    import argparse

    # Parse the script parameters.
    parser = argparse.ArgumentParser(description="Voice chat client: listens on the microphone, transcribes speech, and plays the LLM's spoken reply.")
    parser.add_argument("--ip", default="http://localhost", help="The server IP address.")
    parser.add_argument("--port", default=8000, type=int, help="The server port.")
    args = parser.parse_args()

    try:
        main(args.ip, args.port)
    except KeyboardInterrupt:
        print("\nGoodbye!")
