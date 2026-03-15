import os
import subprocess

files = [
    "advanced_intro , Data Science.mp4",
    "conclusion , Data Science.mp4",
    "intermediat , Data Science.mp4",
    "introduction_powerBi , Data Science.mp4"
]

for file in files:
    input_path = f"videos/{file}"
    output_path = f"audios/{file.replace(' ,', '').replace(' ', '_')}.mp3"  # clean naming
    subprocess.run([
        "ffmpeg", "-i", input_path, output_path
    ])
