"""Faceless Content Generator

Generates a short script using Google's Gemini API, converts it to speech with
``gTTS``, and creates an engaging short-form video with multiple images.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from gtts import gTTS
from moviepy.editor import (
    AudioFileClip, ImageClip, CompositeVideoClip, TextClip,
    concatenate_videoclips, ColorClip
)
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Add FFmpeg to the PATH
os.environ['PATH'] = f"C:\\ffmpeg;{os.environ.get('PATH', '')}"

# Configuration
FONT_PATH = "arial.ttf"  # Make sure this font exists on your system
VIDEO_SIZE = (1080, 1920)  # Vertical 9:16 aspect ratio for reels/shorts
FONT_SIZE = 50
TEXT_COLOR = "white"
BACKGROUND_COLOR = "#1a1a1a"
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
DURATION = 60  # Duration of the video in seconds

if not UNSPLASH_ACCESS_KEY or UNSPLASH_ACCESS_KEY == "YOUR_UNSPLASH_ACCESS_KEY":
    print("❌ Error: Please set your Unsplash Access Key in the .env file")
    print("Get one from: https://unsplash.com/developers")
    exit(1)


def generate_script(topic: str, *, api_key: Optional[str] = None) -> List[str]:
    """Generate a script for a short video about the given topic."""
    try:
        genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = f"""Create a script for a {DURATION}-second YouTube short about {topic}.
        - Write in a natural, engaging tone as if speaking to a friend
        - Keep each segment 1-2 sentences maximum
        - Make it informative but concise
        - Don't use any labels like 'Visual:' or 'Narrator:'
        - Each line should be a complete thought that can be visualized
        - Total script should be around {DURATION} seconds when spoken naturally
        
        Example format:
        The US Open is one of tennis's most prestigious tournaments.
        Held annually in New York, it attracts the world's top players."""
        
        response = model.generate_content(prompt)
        script = response.text.strip().split('\n')
        return [line.strip() for line in script if line.strip()]
    except Exception as e:
        print(f"❌ Error generating script with Gemini: {str(e)}")
        print("\n⚠️  Falling back to a simple template-based script...")
        # Fallback template if Gemini fails
        return [
            f"Here's an interesting quote about {topic}.",
            "The greatest wisdom comes from experience.",
            "Philosophers have pondered this for centuries.",
            "What can we learn from their insights?"
        ]


def get_unsplash_image(query: str, size: tuple = (1080, 1920)) -> Image.Image:
    """Fetch a relevant image from Unsplash."""
    try:
        url = f"https://api.unsplash.com/photos/random?query={query}&client_id={UNSPLASH_ACCESS_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        image_url = response.json()["urls"]["regular"]
        
        img_response = requests.get(image_url, stream=True)
        img_response.raise_for_status()
        
        img = Image.open(img_response.raw)
        # Resize and crop to fit our video size
        img = img.resize((size[0], int(size[0] * img.height / img.width)), Image.Resampling.LANCZOS)
        if img.height < size[1]:
            # If image is not tall enough, create a background and paste
            bg = Image.new('RGB', size, BACKGROUND_COLOR)
            bg.paste(img, (0, (size[1] - img.height) // 2))
            return bg
        else:
            # Crop to center
            top = (img.height - size[1]) // 2
            return img.crop((0, top, size[0], top + size[1]))
    except Exception as e:
        print(f"Warning: Couldn't fetch image for '{query}'. Using solid color background. Error: {str(e)}")
        return Image.new('RGB', size, BACKGROUND_COLOR)


def create_image_with_text(text: str, index: int, total: int) -> Path:
    """Create an image with text overlay on top of a relevant background."""
    # Get a relevant background image
    try:
        # Extract main keywords for image search
        keywords = ' '.join([word for word in text.split() if len(word) > 3][:3])
        bg_image = get_unsplash_image(keywords or "technology", VIDEO_SIZE)
    except:
        bg_image = Image.new('RGB', VIDEO_SIZE, color=BACKGROUND_COLOR)
    
    # Create a semi-transparent overlay
    overlay = Image.new('RGBA', VIDEO_SIZE, (0, 0, 0, 180))  # Dark overlay for better text visibility
    draw = ImageDraw.Draw(overlay)
    
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except:
        font = ImageFont.load_default()
    
    # Add text with word wrap
    lines = []
    words = text.split()
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        if draw.textlength(test_line, font=font) < VIDEO_SIZE[0] - 100:  # 50px padding on each side
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    
    # Draw text
    y_position = (VIDEO_SIZE[1] - (len(lines) * FONT_SIZE * 1.5)) // 2
    
    for line in lines:
        text_width = draw.textlength(line, font=font)
        # Draw text with shadow for better visibility
        for xo, yo in [(-1, -1), (1, 1), (1, -1), (-1, 1)]:  # Shadow effect
            draw.text(
                ((VIDEO_SIZE[0] - text_width) // 2 + xo, y_position + yo),
                line,
                font=font,
                fill=(0, 0, 0, 255),
                stroke_width=0
            )
        # Draw main text
        draw.text(
            ((VIDEO_SIZE[0] - text_width) // 2, y_position),
            line,
            font=font,
            fill=TEXT_COLOR,
            stroke_width=0
        )
        y_position += FONT_SIZE * 1.5
    
    # Combine background with overlay
    if bg_image.mode != 'RGBA':
        bg_image = bg_image.convert('RGBA')
    final_img = Image.alpha_composite(bg_image, overlay)
    
    # Save the image
    img_path = Path(f"scene_{index}.png")
    final_img.convert('RGB').save(img_path)
    return img_path


def text_to_speech(text: str, out_path: Path) -> float:
    """Convert text to speech and return audio duration."""
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(str(out_path))
    return AudioFileClip(str(out_path)).duration


def create_video(script_segments: List[str], audio_path: Path, output_path: Path) -> None:
    """Create a video with multiple images and audio."""
    # Create images for each segment
    image_paths = [create_image_with_text(seg, i, len(script_segments)) 
                  for i, seg in enumerate(script_segments, 1)]
    
    # Calculate duration per segment
    audio = AudioFileClip(str(audio_path))
    total_duration = audio.duration
    segment_duration = total_duration / len(script_segments)
    
    # Create video clips
    clips = []
    for i, img_path in enumerate(image_paths):
        # Create a clip for this segment
        img_clip = ImageClip(str(img_path)).set_duration(segment_duration)
        
        # Add fade in/out (except for first/last clip)
        if i > 0:
            img_clip = img_clip.crossfadein(0.5)
        if i < len(image_paths) - 1:
            img_clip = img_clip.crossfadeout(0.5)
        
        clips.append(img_clip)
    
    # Concatenate all clips
    final_clip = concatenate_videoclips(clips, method="compose")
    
    # Set audio
    final_clip = final_clip.set_audio(audio)
    
    # Write the result to a file
    final_clip.write_videofile(
        str(output_path),
        fps=24,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True
    )
    
    # Clean up temporary files
    for img_path in image_paths:
        img_path.unlink()


def main() -> None:
    """Generate a video from a topic."""
    topic = input("Enter a topic for your short video: ")
    
    try:
        # Generate script segments
        print("Generating script segments...")
        script_segments = generate_script(topic)
        full_script = " ".join(script_segments)
        
        print(f"Generated {len(script_segments)} script segments")
        
        # Save script
        with open("script.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(f"{i+1}. {seg}" for i, seg in enumerate(script_segments)))
        
        # Generate audio
        print("Generating audio...")
        audio_path = Path("output_audio.mp3")
        audio_duration = text_to_speech(full_script, audio_path)
        
        # Create video
        print("Creating video...")
        output_path = Path("output_video.mp4")
        create_video(script_segments, audio_path, output_path)
        
        print(f"\n🎉 Video created successfully: {output_path}")
        print(f"Total duration: {audio_duration:.1f} seconds")
        print(f"Segments: {len(script_segments)}")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
    finally:
        # Clean up
        if 'audio_path' in locals() and audio_path.exists():
            audio_path.unlink()


if __name__ == "__main__":
    main()
