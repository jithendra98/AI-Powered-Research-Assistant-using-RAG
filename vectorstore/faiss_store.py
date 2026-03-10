import faiss
import os
import pickle
import numpy as np

class FaissStore:
    def __init__(self, dim, index_file = 'vectorstore/index.faiss', metadata_file = 'vectorstore/meta.pkl'):
        self.dim = dim
        self.index_file = index_file
        self.metadata_file =  metadata_file
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []

    
    def add(self, vectors, text):

        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)

        self.index.add(vectors)
        self.chunks.extend(text)
    

    def search(self, query_vector, top_k = 5):
        D, I = self.index.search(np.array([query_vector]), top_k)
        return [self.chunks[i] for i in I[0]]

    

    def save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.chunks, f)
    

    def load(self):
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, 'rb') as f:
                self.chunks = pickle.load(f)
            return True
        return False

