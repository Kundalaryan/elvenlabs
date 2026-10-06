"""
Generate ttsWordTimings JSON from existing audio using WhisperX (free, local).

WhisperX transcribes AND does forced word-level alignment against your
provided transcript, giving accurate start/end times per word -- no
ElevenLabs API or paid plan required.

SETUP (one-time):
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pip install whisperx
    # WhisperX also needs ffmpeg installed on your system:
    #   Ubuntu/Debian: sudo apt install ffmpeg
    #   Mac: brew install ffmpeg

USAGE:
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 generate_word_timings_whisperx.py narration.mp3 output.json
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 generate_word_timings_whisperx.py narration.m4a output.json

    (Transcript is not required as input -- WhisperX transcribes the audio
    itself and aligns it. If you want to force it to match YOUR exact script
    text instead of its own transcription, see the --transcript flag below.)
"""

import sys
import os
import json
import argparse
import subprocess
import tempfile
from contextlib import contextmanager

WHISPERX_PYTHON = "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


def configure_cache_directories():
    paths = {
        "MPLCONFIGDIR": os.path.join(CACHE_DIR, "matplotlib"),
        "TORCH_HOME": os.path.join(CACHE_DIR, "torch"),
        "HF_HOME": os.path.join(CACHE_DIR, "huggingface"),
    }
    for env_var, path in paths.items():
        os.environ.setdefault(env_var, path)
        os.makedirs(os.environ[env_var], exist_ok=True)


def configure_ssl_certificates():
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except ModuleNotFoundError:
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


configure_cache_directories()
configure_ssl_certificates()

try:
    import whisperx
except ModuleNotFoundError as exc:
    if exc.name != "whisperx":
        raise
    if sys.executable != WHISPERX_PYTHON and os.path.exists(WHISPERX_PYTHON):
        os.execv(WHISPERX_PYTHON, [WHISPERX_PYTHON, __file__, *sys.argv[1:]])
    print(
        "Error: whisperx is not installed for this Python interpreter.\n"
        "Run this script with the Python 3.13 interpreter that has WhisperX installed:\n"
        f"  {WHISPERX_PYTHON} generate_word_timings_whisperx.py narration.mp3 output.json",
        file=sys.stderr,
    )
    sys.exit(1)


@contextmanager
def audio_input_for_whisperx(audio_path: str):
    extension = os.path.splitext(audio_path)[1].lower()
    if extension not in {".m4a", ".mp4a"}:
        yield audio_path
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        wav_path = temp_audio.name

    try:
        print(f"Converting {extension} audio to WAV for WhisperX...")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                audio_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                wav_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        yield wav_path
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to read .m4a/.mp4a audio files.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg could not convert {audio_path}: {exc.stderr}") from exc
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def generate_timings(audio_path: str, device: str = "cpu", compute_type: str = "int8") -> list:
    print("Loading Whisper model (this may take a minute on first run)...")
    model = whisperx.load_model("base", device, compute_type=compute_type)

    print("Transcribing audio...")
    with audio_input_for_whisperx(audio_path) as prepared_audio_path:
        audio = whisperx.load_audio(prepared_audio_path)
    result = model.transcribe(audio, batch_size=8)

    print(f"Detected language: {result['language']}")
    print("Loading alignment model...")
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )

    print("Running word-level alignment...")
    aligned_result = whisperx.align(
        result["segments"], align_model, metadata, audio, device
    )

    word_timings = []
    for segment in aligned_result["segments"]:
        for word_info in segment.get("words", []):
            # Some words at segment edges can lack timing if confidence is low;
            # skip those rather than injecting fake numbers.
            if "start" not in word_info or "end" not in word_info:
                continue
            word_timings.append({
                "word": word_info["word"].strip(),
                "start": round(word_info["start"], 3),
                "end": round(word_info["end"], 3),
            })

    return word_timings


def main():
    parser = argparse.ArgumentParser(description="Generate ttsWordTimings using WhisperX")
    parser.add_argument("audio_path", help="Path to your narration audio file (mp3/wav/m4a/mp4a)")
    parser.add_argument("output_json", help="Path to write the ttsWordTimings JSON")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    args = parser.parse_args()

    word_timings = generate_timings(args.audio_path, device=args.device)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"ttsWordTimings": word_timings}, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(word_timings)} word timings to {args.output_json}")


if __name__ == "__main__":
    main()
