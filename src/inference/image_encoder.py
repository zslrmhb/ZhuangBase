from base_encoder import BaseEncoder
from image_preprocessor import ImagePreprocessor
from transformers import AutoModel, AutoImageProcessor
import torch

class ImageEncoder(BaseEncoder):

    def __init__(self, model_name = 'google/vit-base-patch16-224-in21k', device = 'cuda'): # defal model
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = ImagePreprocessor(model_name=model_name, device=self.device)


    def load_model(self, model_name=None): # can change model
        if model_name:
            self.model_name = model_name

        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        print(f"Model '{self.model_name}' loaded successfully on {self.device}.")

    def encode(self, image_path):
        image_tensor = self.processor.preprocess(image_path)
        with torch.no_grad():
            outputs = self.model(pixel_values=image_tensor)
            features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        return features # vecter


        