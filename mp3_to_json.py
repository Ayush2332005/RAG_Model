import whisper
import json

# Load model
model = whisper.load_model("base")

# Transcribe audio
result = model.transcribe(
    audio="D:\\RAG\\audios\\advanced_intro_Data_Science.mp4.mp3",
    language="hi",
    task="translate",
    word_timestamps=False
)

# Create chunks for RAG
chunks = []

for segment in result["segments"]:
    chunks.append({
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"]
    })

# Print clean chunks
print("\nChunks:\n", chunks)

# Save chunks to JSON
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=4, ensure_ascii=False)