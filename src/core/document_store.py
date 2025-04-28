# from __future__ import annotations
# from typing import Dict, Any




# class DocumentStore:
#     def __init__(self):
#         self.store = {}

#     def add_document(self, id: str):
#         """Add or update a document"""
#         self.store[id] = document

#     def get_document(self, id: str) -> Document | None:
#         """Retrieve a document by ID"""
#         return self.store.get(id, None)

#     def update_document(self, id: str, metadata: Dict[str, Any]):
#         """Update document metadata"""
#         if id in self.store:
#             for key, value in metadata.items():
#                 self.store[id].metadata[key] = value

#     def delete_document(self, id: str):
#         """Delete a document by ID"""
#         if id in self.store:
#             del self.store[id]

#     def get_all_documents(self) -> list[Document]:
#         """Return all documents"""
#         return list(self.store.values())