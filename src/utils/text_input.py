from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
import torch



tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2") ## SENTENCE TOKEN
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")  ## SENTENCE MODEL

'''
text_input_function -> text to vector 
'''
def text_input_function(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    # Mean pooling to get sentence embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    print(embeddings)

if __name__ == "__main__":
    with open("exampleText.txt", "r") as f:
        text = f.read()

    print(word_tokenize(text))
