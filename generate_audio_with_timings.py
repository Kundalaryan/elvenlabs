"""
Generate audio + ttsWordTimings JSON in ONE call using ElevenLabs'
text-to-speech "with-timestamps" endpoint.

Usage:
    export ELEVENLABS_API_KEY="your_key_here"
    python generate_audio_with_timings.py script.txt VOICE_ID output_audio.mp3 output_timings.json

Requires:
    Python 3.8+
"""

import sys
import os
import json
import base64
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_URL_TEMPLATE = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"


def generate_with_timestamps(text: str, voice_id: str, api_key: str, model_id: str = "eleven_multilingual_v2") -> dict:
    url = API_URL_TEMPLATE.format(voice_id=voice_id)
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": model_id,
    }

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ElevenLabs API request failed ({exc.code}): {error_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach ElevenLabs API: {exc.reason}") from exc


def save_audio(base64_audio: str, output_path: str):
    audio_bytes = base64.b64decode(base64_audio)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)


def to_word_timings(alignment: dict) -> list:
    """
    Groups ElevenLabs' character-level alignment into word-level timings:
    [{ "word": "...", "start": 0.12, "end": 0.38 }, ...]
    """
    chars = alignment.get("characters", [])
    char_starts = alignment.get("character_start_times_seconds", [])
    char_ends = alignment.get("character_end_times_seconds", [])

    words = []
    current_word = ""
    word_start = None
    prev_end = None

    for ch, start, end in zip(chars, char_starts, char_ends):
        if ch.strip() == "":
            if current_word:
                words.append({
                    "word": current_word,
                    "start": round(word_start, 3),
                    "end": round(prev_end, 3),
                })
                current_word = ""
                word_start = None
        else:
            if word_start is None:
                word_start = start
            current_word += ch
            prev_end = end

    if current_word:
        words.append({
            "word": current_word,
            "start": round(word_start, 3),
            "end": round(prev_end, 3),
        })

    return words


def main():
    if len(sys.argv) != 5:
        print("Usage: python generate_audio_with_timings.py <script.txt> <voice_id> <output_audio.mp3> <output_timings.json>")
        sys.exit(1)

    script_path, voice_id, output_audio, output_json = sys.argv[1:5]

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: set ELEVENLABS_API_KEY environment variable first.")
        sys.exit(1)

    with open(script_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Calling ElevenLabs text-to-speech (with-timestamps)...")
    result = generate_with_timestamps(text, voice_id, api_key)

    # response contains: audio_base64, alignment, normalized_alignment
    save_audio(result["audio_base64"], output_audio)
    print(f"Saved audio to {output_audio}")

    alignment = result.get("alignment", result.get("normalized_alignment", {}))
    word_timings = to_word_timings(alignment)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"ttsWordTimings": word_timings}, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(word_timings)} word timings to {output_json}")


if __name__ == "__main__":
    main()
