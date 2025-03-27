## An Image class to represent the image object
class Document:
    
    def __init__(self, id: str, title: str, tags: list[str], path: str):
        self.id = id
        self.title = title
        self.tags = tags
        self.path = path
        
    def update(self, title: str, tags: list[str], path: str):
        self.title = title
        self.tags = tags
        self.path = path
     
class DocDB:
    
    def __init__(self):
        self.docDB = {}
        
        
    def add_document_with_para(self, id: str, title: str, tags: list[str], path: str):
        image = Document(id, title, tags, path)
        self.docDB[id] = image
        
        
    def add_document_with_object(self, id: str, document: Document):
        self.docDB[id] = document
        
        
    def get_document(self, id: str) -> Document | None:
        if id not in self.docDB:
            return None
        return self.docDB[id]
    
    
    def update_document_with_para(self, id: str, title: str, tags: list[str], path: str):
        self.docDB.get(id).update(title, tags, path)
        
        
    def update_document_with_object(self, id: str, document: Document):
        self.docDB[id] = document
        
        
    def delete_document(self, id: str):
        if id in self.docDB:
            del self.docDB[id]
        
        
    def get_all_document(self) -> list[Document]:
        return list(self.docDB.values())