from models.query import Query

class SearchEngine:
    def __init__(self):
        pass 

    def index_document(self):
    
        """Add or update a document (vector + metadata)"""
        pass 

    def delete_document(self):
        """Remove a document by ID"""
        pass 

    def get_document(self):
        """Fetch document metadata by ID"""
        pass 

    def list_documents(self):
        """List all indexed document IDs or metadata"""
        pass 

    def search_similar(self):
        """Search similar documents given an input"""
        pass 

    def save_index(self):
        """Persist vector index and/or metadata to disk"""
        pass 

    def load_index(self):
        """Load vector index and/or metadata from disk"""
        pass 

    def get_stats(self):
        """Return index size, vector dim, modality info, etc."""
        pass 
