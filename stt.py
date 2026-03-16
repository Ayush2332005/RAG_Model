import whisper
import json

# Load model
model = whisper.load_model("small")

# Transcribe audio
result = model.transcribe(
    audio="audios/sample.mp4.mp3",
    language="hi",
    task="translate",
    fp16=False
)

# Print translated text
print(result["text"])

# Save full result to JSON
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)