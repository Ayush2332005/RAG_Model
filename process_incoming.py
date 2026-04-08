import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from read_chunks import create_embeddings
from embedding import EmbeddingModel, VectorStore
import requests
import json

# def main():
#         embedder = EmbeddingModel()
#         store = VectorStore()

#         incoming_query = input("Enter your query: ")
#         question_embedding = embedder.create_embeddings([incoming_query])[0]

#         df = joblib.load("embeddings.joblib")
#         store.vectors = df['text'].tolist()
#         for embedding in df['embedding']:
#             store.index.add(np.array([embedding]))
#         results = store.search(incoming_query, top_k=3)
#         print("Top results:")
#         for result in results:
#             print(result)
# if __name__ == "__main__":
#     main()



def inference(prompt):
    r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "Stream": False
            }
        )
    
    response = r.json()
    print(response)
    return response


df = joblib.load("embeddings.joblib")

incoming_query = input("Enter your query: ")
question_embedding = create_embeddings([incoming_query])[0]
#print(question_embedding)

similarities = cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()
#print(similarities)
top_results = 3
max_indx = similarities.argsort()[-3::-1][0:top_results]
#print(max_indx)


new_df = df.iloc[max_indx]
prompt = f"""
You are an assistant that answers questions based ONLY on given video subtitle chunks.

Each chunk contains:
- text spoken
- timestamp (start and end in seconds)

Here are the relevant chunks:
{new_df[['text','start','end','source']].to_string(index=False)}

User question:
{incoming_query}

Instructions:
- Identify which video and timestamp contains the answer
- Clearly mention start and end time
- If answer is not present, say:
  "This question is not related to the available video content."
"""
# print(new_df[[ "text"]])

with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("responce.txt", "w", encoding="utf-8") as f:
    f.write(response)

# for index,item in new_df.iterrows():
#     print(item['text'],item['start'],item['end'])
   
