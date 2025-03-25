## An Image class to represent the image object
class Image:
    
    def __init__(self, id: str, title: str, tags: list[str], path: str):
        self.id = id
        self.title = title
        self.tags = tags
        self.path = path
        
    def update(self, title: str, tags: list[str], path: str):
        self.title = title
        self.tags = tags
        self.path = path
        
        
class ImageDB:
    
    def __init__(self):
        self.imageDB = {}
        
        
    def add_image_with_para(self, id: str, title: str, tags: list[str], path: str):
        image = Image(id, title, tags, path)
        self.imageDB[id] = image
        
        
    def add_image_with_object(self, id: str, image: Image):
        self.imageDB[id] = image
        
        
    def get_image(self, id: str) -> Image | None:
        return self.imageDB[id]
    
    
    def update_image_with_para(self, id: str, title: str, tags: list[str], path: str):
        self.imageDB.get(id).update(title, tags, path)
        
        
    def update_image_with_object(self, id: str, image: Image):
        self.imageDB[id] = image
        
        
    def delete_image(self, id: str):
        if id in self.imageDB:
            del self.imageDB[id]
        
        
    def get_all_images(self) -> list[Image]:
        return list(self.imageDB.values())