from __future__ import annotations

from models.query import Query
from models.document import Document
from core.document_store import DocumentStore
from core.vector_index import VectorIndex


class SearchEngine:
    def __init__(self):
        self.document_store = DocumentStore()
        self.vector_index = VectorIndex(512)  # temp value
        self.counter = 0

    def generate_doc_id(self):
        self.counter += 1
        return f"doc_{self.counter}"

    def index_document(self, document: Document):
        """Add or update a document (vector + metadata)"""
        doc_id = self.generate_doc_id()

        # add vector

        # add metadata

        pass

    def delete_document(self, id: str):
        """Remove a document by ID (vector + metadata)"""

        # delete metadata
        self.document_store.delete_document(id)
        self.counter -= 1

    def get_document(self, id: str) -> Document | None:
        """Fetch (vector + metadata) by ID"""
        return self.document_store.get_document(id)

    def list_documents(self):
        """List all indexed document IDs and metadata"""
        pass

    def search_similar(self, query: Query):
        """Search similar documents given an input"""
        input_data = query.input_data
        input_type = query.input_type
        top_k = query.top_k

    def save_index(self):
        """Persist vector index and/or metadata to disk"""
        pass

    def load_index(self):
        """Load vector index and/or metadata from disk"""
        pass

    def get_stats(self):
        """Return index size, vector dim, modality info, etc."""
        pass
