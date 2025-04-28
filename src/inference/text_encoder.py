# text_encoder.py
from .base_encoder import BaseEncoder #
from .text_preprocessor import TextPreprocessor # Assuming text_preprocessor.py is in the same directory
from transformers import AutoModel
import torch

class TextEncoder(BaseEncoder): #
    '''Encoder for text data using Hugging Face transformer models.'''

    def __init__(self, model_name='bert-base-uncased', device='cuda'):  # set a default model
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = TextPreprocessor(model_name=self.model_name, device=self.device)

    def load_model(self, model_name=None): #
        '''Load a pre-trained transformer model.'''
        if model_name:
            self.model_name = model_name
            self.processor = TextPreprocessor(model_name=self.model_name, device=self.device)

        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval() 
        print(f"Model '{self.model_name}' loaded successfully on {self.device}.")

    def encode(self, text_data): #
        '''Encode text data into vector representations (embeddings).'''
        encoded_input = self.processor.preprocess(text_data)

        with torch.no_grad(): 
            outputs = self.model(**encoded_input)
            last_hidden_state = outputs.last_hidden_state
            attention_mask = encoded_input['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9) 
            mean_pooled_features = sum_embeddings / sum_mask

            features = mean_pooled_features.cpu().numpy()

        return features 

