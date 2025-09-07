"""Faceless Content Generator

Generates a short script using OpenAI's API, converts it to speech with
``gTTS``, creates a placeholder image with OpenAI's image API, and combines
both into a simple video using ``moviepy``.

The script expects the ``OPENAI_API_KEY`` environment variable to be set.
``moviepy`` can be slow because it relies on ``ffmpeg`` under the hood; make
sure ``ffmpeg`` is installed on your system.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

import openai
from gtts import gTTS
from moviepy.editor import AudioFileClip, ImageClip


def generate_script(topic: str, *, api_key: Optional[str] = None) -> str:
    """Return a short script about ``topic`` using the ChatCompletion API."""
    openai.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Write a short script about {topic}."}],
    )
    return response["choices"][0]["message"]["content"].strip()


def text_to_speech(text: str, out_path: Path) -> None:
    """Convert ``text`` into speech and save to ``out_path``."""
    tts = gTTS(text)
    tts.save(str(out_path))


def image_for_topic(topic: str, out_path: Path, *, api_key: Optional[str] = None) -> None:
    """Generate an image for ``topic`` using the OpenAI Image API."""
    openai.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    response = openai.Image.create(prompt=topic, size="1024x1024")
    image_b64 = response["data"][0]["b64_json"]
    out_path.write_bytes(base64.b64decode(image_b64))


def create_video(image_path: Path, audio_path: Path, out_path: Path) -> None:
    """Combine ``image_path`` and ``audio_path`` into a video at ``out_path``."""
    audio_clip = AudioFileClip(str(audio_path))
    image_clip = ImageClip(str(image_path)).set_duration(audio_clip.duration)
    final_clip = image_clip.set_audio(audio_clip)
    final_clip.write_videofile(str(out_path), fps=24)


def main() -> None:
    topic = input("Topic: ")
    script = generate_script(topic)

    audio_file = Path("audio.mp3")
    image_file = Path("image.png")
    video_file = Path("video.mp4")

    text_to_speech(script, audio_file)
    image_for_topic(topic, image_file)
    create_video(image_file, audio_file, video_file)
    print(f"Created {video_file}")


if __name__ == "__main__":
    main()
