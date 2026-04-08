import requests
import json
import os
import pandas as pd
import joblib


# =========================
# 🔹 Embedding Function
# =========================
def create_embeddings(text_list):
    embeddings = []

    for text in text_list:
        r = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
            }
        )

        data = r.json()

        # Safety check
        if "embedding" not in data:
            print("❌ Error in embedding response:", data)
            continue

        embeddings.append(data["embedding"])

    return embeddings

def inference(prompt):
    r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-r1",
                "prompt": prompt
            }
        )


# =========================
# 🔹 Main Processing
# =========================
def main():
    folder = "jsons"

    # Check folder exists
    if not os.path.exists(folder):
        print(f"❌ Folder '{folder}' not found")
        return

    # Skip if already created
    if os.path.exists("embeddings.joblib"):
        print("⚠️ embeddings.joblib already exists. Delete it to regenerate.")
        return

    json_files = [f for f in os.listdir(folder) if f.endswith(".json")]

    if not json_files:
        print("❌ No JSON files found in 'jsons' folder")
        return

    all_chunks = []
    chunk_id = 0

    for json_file in json_files:
        file_path = os.path.join(folder, json_file)
        print(f"📄 Processing: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)

        # Expecting list of chunks
        texts = [c["text"] for c in content]

        embeddings = create_embeddings(texts)

        if len(embeddings) != len(content):
            print("⚠️ Mismatch in embeddings and chunks length")
            continue

        for i, chunk in enumerate(content):
            chunk["chunk_id"] = chunk_id
            chunk["embedding"] = embeddings[i]
            chunk["source"] = json_file  # helpful for tracking

            chunk_id += 1
            all_chunks.append(chunk)

    # Convert to DataFrame
    df = pd.DataFrame.from_records(all_chunks)

    # Save
    joblib.dump(df, "embeddings.joblib")

    print("\n✅ Embeddings saved successfully!")
    print(f"📊 Total chunks processed: {len(df)}")


# =========================
# 🔹 Entry Point
# =========================
if __name__ == "__main__":
    main()