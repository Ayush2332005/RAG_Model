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
print(new_df[[ "text"]])

'''for index,item in new_df.iterrows():
    print(item['text'],item['chunk_id'],item['start'],item['end'])'''
   
