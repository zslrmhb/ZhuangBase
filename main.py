from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
import torch

with open("exampleText.txt", "r") as f:
    text = f.read()

print(word_tokenize(text))

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

text = text
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)

# Mean pooling to get sentence embeddings
embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
print(embeddings)