#! /usr/bin/env python3

"""
Text-to-Speech Generation with Multiple Providers and Caching

Features:
1. Multiple TTS Provider Support
   - OpenAI TTS integration with voice selection
   - ElevenLabs TTS integration with voice selection
   - Model configuration and provider management
   - Voice rotation for OpenAI voices

2. Audio File Caching
   - SHA256 hash-based caching of generated audio
   - Cache invalidation based on text, model, voice and speed
   - Separate base and speed-adjusted file caching
   - Automatic cache directory management

3. Speed Adjustment
   - Time-stretching without pitch change using pyrubberband
   - Configurable speed multiplier
   - Caching of speed-adjusted files
   - Mono conversion for consistent processing

4. Async Generation Pipeline
   - Asynchronous audio generation with callbacks
   - Progress tracking and error handling
   - Provider-specific error management
   - File verification and validation

5. CLI Interface
   - Text file input support
   - Model and voice selection
   - Speed adjustment configuration
   - Custom output path specification

6. Debug Support
   - Debug logging of key operations
   - File generation tracking
   - Cache hit/miss monitoring
   - Error context preservation
"""

import os
import hashlib
import asyncio
import argparse
from typing import Callable, Optional, Coroutine, Any
from openai import OpenAI
import soundfile as sf
import pyrubberband as pyrb
from elevenlabs.client import ElevenLabs
from elevenlabs import save
from utils import dprint
import base64

# Debug flag
DEBUG = False


CACHE_DIR = ".audio_cache"
SPEED = 1.0  # Default speed multiplier

# TTS Model configurations
OPENAI_MODELS = {
    "tts-1": {"name": "tts-1", "provider": "openai"},
    "tts-1-hd": {"name": "tts-1-hd", "provider": "openai"},
    "gpt-4o-audio": {"name": "gpt-4o-audio-preview", "provider": "openai_chat"},
}

ELEVENLABS_MODELS = {
    "eleven_multilingual": {"name": "eleven_multilingual_v2", "provider": "elevenlabs"}
}

ALL_MODELS = {**OPENAI_MODELS, **ELEVENLABS_MODELS}

# OpenAI voice rotation
OPENAI_MALE_VOICES = ["ash", "echo", "fable", "onyx"]
OPENAI_FEMALE_VOICES = ["alloy", "coral", "nova", "sage", "shimmer"]
OPENAI_GOOD_VOICES = ["alloy", "nova", "sage", "shimmer"]
OPENAI_VOICES = OPENAI_GOOD_VOICES
VOICE_INDEX = 0


class TTS:
    def __init__(self):
        self.openai_client = OpenAI()
        self.elevenlabs_client = ElevenLabs()
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _gen_cache_filename(self, model: str, voice: str, text: str) -> str:
        # Include model, voice, speed in hash for unique caching
        encoded_string = (model + voice + text + str(SPEED)).encode()
        hasher = hashlib.sha256()
        hasher.update(encoded_string)
        base_hash = hasher.hexdigest()
        # For speed=1.0, we only need base file. For other speeds, we need speed-adjusted file
        if SPEED == 1.0:
            return f"{CACHE_DIR}/{base_hash}.mp3", None
        else:
            return (
                f"{CACHE_DIR}/{base_hash}_base.mp3",
                f"{CACHE_DIR}/{base_hash}_speed_{SPEED}.mp3",
            )

    def _adjust_speed(self, input_file: str, output_file: str, speed: float) -> None:
        """Adjust the speed of an audio file without changing pitch"""
        dprint(f"Adjusting speed of {input_file} to {speed}x")

        y, sr = sf.read(input_file)

        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = y.mean(axis=1)

        # Time-stretch using pyrubberband (maintains pitch)
        y_fast = pyrb.time_stretch(y, sr, speed)

        # Save the processed audio
        sf.write(output_file, y_fast, sr)
        dprint(f"Saved speed-adjusted audio to {output_file}")

    async def _generate_audio_async(
        self,
        text: str,
        model: str,
        voice: str,
        callback: Callable[
            [Optional[str], Optional[Exception]], Coroutine[Any, Any, None]
        ],
    ):
        """Async function for generating audio"""
        try:
            dprint(f"Generating audio for text: {text[:50]}...")
            base_filename, speed_filename = self._gen_cache_filename(model, voice, text)

            # If speed is 1.0, we only need the base file
            target_file = base_filename if SPEED == 1.0 else speed_filename

            if os.path.exists(target_file):
                dprint(f"Using cached audio file: {target_file}")
                await callback(target_file, None)
                return

            # Generate base file if needed
            if not os.path.exists(base_filename):
                dprint(f"Generating new audio with {model} and voice {voice}")

                if ALL_MODELS[model]["provider"] == "openai":
                    speech_response = self.openai_client.audio.speech.create(
                        model=ALL_MODELS[model]["name"],
                        voice=voice,
                        input=text,
                    )
                    speech_response.stream_to_file(base_filename)

                elif ALL_MODELS[model]["provider"] == "openai_chat":
                    # New GPT-4o audio preview implementation
                    audio_completion = self.openai_client.chat.completions.create(
                        model=ALL_MODELS[model]["name"],
                        modalities=["text", "audio"],
                        audio={"voice": voice, "format": "mp3"},
                        messages=[
                            {"role": "user", "content": f"Say this exactly: {text}"}
                        ],
                    )

                    audio_data = base64.b64decode(
                        audio_completion.choices[0].message.audio.data
                    )
                    with open(base_filename, "wb") as f:
                        f.write(audio_data)

                elif ALL_MODELS[model]["provider"] == "elevenlabs":
                    audio = self.elevenlabs_client.text_to_speech.convert(
                        text=text,
                        voice_id=voice,
                        model_id=ALL_MODELS[model]["name"],
                        output_format="mp3_44100_128",
                    )
                    save(audio, base_filename)

                    if os.path.getsize(base_filename) == 0:
                        raise Exception("Generated audio file is empty")

            # Adjust speed if needed
            if SPEED != 1.0:
                self._adjust_speed(base_filename, speed_filename, SPEED)

            await callback(target_file, None)

        except Exception as e:
            dprint(f"Error generating audio: {str(e)}")
            await callback(None, e)

    def generate_audio(
        self,
        text: str,
        callback: Callable[
            [Optional[str], Optional[Exception]], Coroutine[Any, Any, None]
        ],
        model: str = "tts-1-hd",
        voice: str = None,
        speed: float = 1.0,
    ) -> None:
        """
        Generate audio file for text and call callback with path on completion

        Args:
            text: Text to convert to speech
            callback: Async function to call with (filename, error) when complete
            model: TTS model to use
            voice: Voice to use (defaults to Maya for ElevenLabs, nova for OpenAI)
            speed: Speed multiplier for the audio (default: 1.0)
        """
        if model not in ALL_MODELS:
            raise ValueError(f"Unsupported model: {model}")

        if not voice:
            if "eleven" in model:
                voice = "XB0fDUnXU5powFXDhCwa"
            else:
                voice = OPENAI_VOICES[VOICE_INDEX]

        global SPEED
        SPEED = speed

        async def run_async():
            await self._generate_audio_async(text, model, voice, callback)

        asyncio.create_task(run_async())


async def main():
    parser = argparse.ArgumentParser(
        description="Convert text file to speech using different TTS models"
    )
    parser.add_argument("input_file", help="Path to text file to convert")
    parser.add_argument(
        "--model",
        choices=ALL_MODELS.keys(),
        default="tts-1-hd",
        help="TTS model to use",
    )
    parser.add_argument("--voice", default=None, help="Voice to use")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier")
    parser.add_argument("--output", help="Output audio file path")

    args = parser.parse_args()

    # Read input text
    with open(args.input_file, "r") as f:
        text = f.read()

    tts = TTS()
    done = asyncio.Event()
    output_file = None
    error = None

    async def callback(filename: Optional[str], err: Optional[Exception]):
        nonlocal output_file, error
        output_file = filename
        error = err
        done.set()

    tts.generate_audio(
        text, callback, model=args.model, voice=args.voice, speed=args.speed
    )
    await done.wait()

    if error:
        print(f"Error generating audio: {error}")
        return

    if args.output and output_file:
        os.rename(output_file, args.output)
        print(f"Audio saved to: {args.output}")
    else:
        print(f"Audio saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
