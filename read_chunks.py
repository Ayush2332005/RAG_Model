import requests
import json
import os 
import pandas as pd


def create_embeddings(text_list):
    embeddings = []

    for text in text_list:
        r = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",   # ✅ correct model
                "prompt": text                # ✅ single text
            }
        )

        data = r.json()

        # Debug safety
        if "embedding" not in data:
            print("Error:", data)
            continue

        embeddings.append(data["embedding"])   # ✅ correct key

    return embeddings


jsons = os.listdir("jsons")
my_dict = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}", "r", encoding="utf-8") as f:
        content = json.load(f)

    print(f"jsons/{json_file}")

    # ✅ content is a list
    texts = [c['text'] for c in content]

    embeddings = create_embeddings(texts)

    for i, chunk in enumerate(content):   # ✅ FIXED
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id += 1
        my_dict.append(chunk)


print(my_dict)

df = pd.DataFrame.from_records(my_dict)
print(df)