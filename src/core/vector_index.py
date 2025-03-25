import faiss
import numpy as np
from typing import List, Tuple

class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.id_map = []  # Stores image IDs in the same order as vectors

    def add(self, vectors: np.ndarray, ids: List[str]):
        """
        Add vectors and their corresponding image IDs.
        """
        assert vectors.shape[0] == len(ids)
        self.index.add(vectors.astype(np.float32))
        self.id_map.extend(ids)


    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Returns list of (image_id, distance) tuples.
        """
        if self.index.ntotal == 0:
            return []

        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i < len(self.id_map):
                results.append((self.id_map[i], dist))
        return results


    def save(self, path: str):
        faiss.write_index(self.index, path)


    def load(self, path: str):
        self.index = faiss.read_index(path)



"""
Example code from 好朋友 for how to use document_store and vector_index

from src.core.vector_index import VectorIndex
from src.models.image_db import ImageDB
from src.inference.image_encoder import encode_image  # hypothetical

image_db = ImageDB()
vector_index = VectorIndex(dim=512)

# Add all images to vector index
image_ids = []
image_vectors = []

for img in image_db.get_all_images():
    vec = encode_image(img.path)  # should return np.ndarray of shape (512,)
    image_ids.append(img.id)
    image_vectors.append(vec)

image_vectors = np.stack(image_vectors)
vector_index.add(image_vectors, image_ids)


----------------
Search

query_embedding = encode_image("query.jpg")
top_results = vector_index.search(query_embedding, top_k=5)

for img_id, distance in top_results:
    image = image_db.get_image(img_id)
    print(f"{image.title} - Distance: {distance:.2f}")


"""