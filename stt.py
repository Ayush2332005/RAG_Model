import whisper
import json

# Load model
model = whisper.load_model("large-v2")

# Transcribe audio
result = model.transcribe(
    audio="audios/sample.mp3",
    language="hi",
    task="translate",
    word_timestamps=False
)

# Print segments
print(result["segments"])

# Create chunks for RAG
chunks = []

for segment in result["segments"]:
    chunks.append({
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"]
    })

# Print chunks
print("\nChunks:\n", chunks)

# Save chunks to JSON
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=4, ensure_ascii=False)