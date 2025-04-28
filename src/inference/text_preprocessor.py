from .base_preprocessor import BasePreprocessor 
from transformers import AutoTokenizer
import torch

class TextPreprocessor(BasePreprocessor): 
    '''Preprocessor for text data using Hugging Face tokenizers.'''

    def __init__(self, model_name="bert-base-uncased", use_fast=True, device=None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu") # Assign device

    def preprocess(self, text_data): #
        '''Tokenize and preprocess the input text data.'''
        if isinstance(text_data, str):
            text_data = [text_data] 
        processed = self.tokenizer(text_data, return_tensors='pt', padding=True, truncation=True)
        
        input_ids = processed['input_ids'].to(self.device)
        attention_mask = processed['attention_mask'].to(self.device)
        

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
