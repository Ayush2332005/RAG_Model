from sentence_transformers import SentenceTransformer
import faiss

class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name,384)

    def create_embeddings(self, text_list):
        return self.model.encode(text_list)
    
class VectorStore:
    def __init__(self):
        self.vectors = []
        self.index = faiss.IndexFlatL2(384)  

    def add(self, vector, text):
        self.index.add(vector)
        self.vectors.extend(text)

    def search(self, query_vector, top_k=3):
        query_embedding = query_vector.encode([query_vector])
        d , i = self.index.search(query_embedding, top_k)
        return [self.vectors[idx] for idx in i[0]]
