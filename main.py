from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
import torch


from src.utils import text_input

# with open("exampleText.txt", "r") as f:
#     text = f.read()

# print(word_tokenize(text))


text_input.text_input_function("Something")