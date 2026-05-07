import ollama
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name):
        self.model_name = model_name

    def embed(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text to embed must be a non-empty string")

        response = ollama.embed(self.model_name, text)
        embeddings = response.get("embeddings")
        if not embeddings:
            raise RuntimeError("Embedding response did not contain embeddings")

        vector = embeddings[0] if isinstance(embeddings[0], list) else embeddings
        array = np.array(vector, dtype=np.float32).flatten()
        if array.size == 0:
            raise RuntimeError("Received an empty embedding vector")

        logger.debug("Generated embedding of size %s", array.size)
        return array