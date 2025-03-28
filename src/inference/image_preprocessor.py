'''
Image preprocessing 

'pip install torch torchvision transformers'
'''

from base_preprocessor import BasePreprocessor
from transformers import AutoImageProcessor
import torch
from torchvision.io import read_image


class ImagePreprocessor(BasePreprocessor):

    def __init__(self, model_name = "facebook/detr-restnet-50", use_fast = True, device = None):
         self.model = model_name
         self.processor = AutoImageProcessor.from_pretrained(model_name,use_fast=use_fast)
         self.device = device

    def preprocess(self, image_path):
         '''Load and preprocess'''
         image = read_image(image_path)
         processed = self.processor( image, return_tensor = 'pt')
         pixel_value = processed['pixel_value'].to(self.device)
         return pixel_value

