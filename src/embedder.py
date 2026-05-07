import ollama
import numpy as np

class Embedder:
    def __init__(self, model_name):
        self.model_name = model_name

    def embed(self, text):
        response = ollama.embed(self.model_name, text)
        return np.array(response["embeddings"], dtype=np.float32).flatten()