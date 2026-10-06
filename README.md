# Audio generation and word timings

This project generates narration audio and word-level timing data for training
or e-learning content. It supports two workflows:

1. **ElevenLabs TTS** — generate an MP3 and timing data in one API request.
2. **WhisperX alignment** — create timing data locally from an existing audio
   file, without using the ElevenLabs API.

The timing files use the `ttsWordTimings` format:

```json
{
  "ttsWordTimings": [
    {
      "word": "Welcome",
      "start": 0.091,
      "end": 0.421
    }
  ]
}
```

## Requirements

- Python 3.8 or newer
- `ffmpeg` (required by WhisperX for `.m4a`/`.mp4a` input)
- An ElevenLabs API key for the ElevenLabs workflow
- WhisperX and its dependencies for local alignment

Install `ffmpeg` with Homebrew on macOS:

```bash
brew install ffmpeg
```

Install Python dependencies:

```bash
python -m pip install whisperx
```

WhisperX may have additional platform-specific requirements. A Python 3.13
interpreter is currently configured as the fallback in
`generate_word_timings_whisperx.py`; update `WHISPERX_PYTHON` in that file if
WhisperX is installed in a different location.

## ElevenLabs: generate audio and timings

Set the API key in the environment rather than placing it in source code:

```bash
export ELEVENLABS_API_KEY="your_api_key"
```

Create a plain-text script, then run:

```bash
python generate_audio_with_timings.py \
  script.txt \
  VOICE_ID \
  output/narration.mp3 \
  output/narration.json
```

The script uses the `eleven_multilingual_v2` model by default. It writes the
audio file to the requested path and stores the generated word timings in the
JSON output path.

## WhisperX: align existing audio

Use this workflow when audio already exists and only timing data is needed:

```bash
python generate_word_timings_whisperx.py \
  narration.mp3 \
  output/narration.json
```

`.mp3`, `.wav`, `.m4a`, and `.mp4a` inputs are supported. The default
configuration runs the Whisper base model on the CPU:

```bash
python generate_word_timings_whisperx.py \
  narration.m4a \
  output/narration.json \
  --device cpu
```

The first run may download the Whisper and alignment models. They are cached
under `.cache/` in the project directory.

## Project files

| File | Purpose |
| --- | --- |
| `generate_audio_with_timings.py` | Calls ElevenLabs TTS and converts character alignment into word timings. |
| `generate_word_timings_whisperx.py` | Transcribes and aligns existing audio locally with WhisperX. |
| `*.json` | Generated word-timing data; ignored by Git. |
| `*.mp3`, `*.m4a`, `*.wav` | Generated or source audio; ignored by Git. |

## Recommended workflow

1. Save the narration text in a `.txt` file.
2. Generate audio and timings with the ElevenLabs script, or start with an
   existing recording.
3. If needed, regenerate timings with WhisperX for local alignment.
4. Review the JSON timings before using them in captions, animation, or a
   video composition.

## Security and repository hygiene

- Never commit `ELEVENLABS_API_KEY` or any other credential.
- Keep generated audio files outside source control; `.gitignore` excludes
  common audio formats.
- Generated timing JSON files are ignored because they can be recreated from
  the source audio.
- Local model caches, Python bytecode, and macOS metadata are also ignored.
