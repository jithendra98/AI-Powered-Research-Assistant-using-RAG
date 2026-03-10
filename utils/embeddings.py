from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, documents):

        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        return [texts, embeddings.cpu().numpy()]