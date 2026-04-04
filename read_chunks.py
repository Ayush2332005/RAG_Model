import requests

def create_embeddings(text):
    r=requests.post("http://localhost:11434/api/embeddings", json={
        "model": "bge-m3",
        "prompt": text
    })

    embeddings = r.json()["embedding"]

    return embeddings

a=create_embeddings("This is a sample text to create embeddings for RAG.")
print(a)