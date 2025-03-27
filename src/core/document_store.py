from __future__ import annotations
from typing import Dict, Any

from models.document import Document


class DocumentStore:
    def __init__(self):
        self.store = {}

    def add_document(self, id: str, document: Document):
        """Add or update a document"""
        self.store[id] = document

    def get_document(self, id: str) -> Document | None:
        """Retrieve a document by ID"""
        return self.store.get(id, None)

    def update_document(self, id: str, metadata: Dict[str, Any]):
        """Update document metadata"""
        if id in self.store:
            for key, value in metadata.items():
                self.store[id].metadata[key] = value

    def delete_document(self, id: str):
        """Delete a document by ID"""
        if id in self.store:
            del self.store[id]

    def get_all_documents(self) -> list[Document]:
        """Return all documents"""
        return list(self.store.values())

    # def __init__(self):
    #     self.docDB = {}

    # def add_document_with_para(self, id: str, title: str, tags: list[str], path: str):
    #     image = Document(id, title, tags, path)
    #     self.docDB[id] = image

    # def add_document_with_object(self, id: str, document: Document):
    #     self.docDB[id] = document

    # def get_document(self, id: str) -> Document | None:
    #     if id not in self.docDB:
    #         return None
    #     return self.docDB[id]

    # def update_document_with_para(self, id: str, title: str, tags: list[str], path: str):
    #     self.docDB.get(id).update(title, tags, path)

    # def update_document_with_object(self, id: str, document: Document):
    #     self.docDB[id] = document

    # def delete_document(self, id: str):
    #     if id in self.docDB:
    #         del self.docDB[id]

    # def get_all_document(self) -> list[Document]:
    #     return list(self.docDB.values())
