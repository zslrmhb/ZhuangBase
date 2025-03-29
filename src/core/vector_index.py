# # pip install faiss-cpu
# import faiss
# import numpy as np
# from typing import List, Tuple

# class VectorIndex:
#     def __init__(self, dim: int):
#         self.dim = dim
#         self.index = faiss.IndexFlatL2(dim)
#         self.id_map = []  # Stores image IDs in the same order as vectors

#     def add(self, vectors: np.ndarray, ids: List[str]):
#         """
#         Add vectors and their corresponding image IDs.
#         """
#         assert vectors.shape[0] == len(ids)
#         self.index.add(vectors.astype(np.float32))
#         self.id_map.extend(ids)


#     def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
#         """
#         Returns list of (image_id, distance) tuples.
#         """
#         if self.index.ntotal == 0:
#             return []

#         query_vector = query_vector.reshape(1, -1).astype(np.float32)
#         distances, indices = self.index.search(query_vector, top_k)
#         results = []
#         for i, dist in zip(indices[0], distances[0]):
#             if i < len(self.id_map):
#                 results.append((self.id_map[i], dist))
#         return results


    def save(self, path: str):
        faiss.write_index(self.index, path)


    def load(self, path: str):
        self.index = faiss.read_index(path)
