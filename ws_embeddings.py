from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
def embeddings_model(model, tokenizer, sent, idx):
    tokens = list()
    words = sent
    #This does the equivalent of tokenizer.encode(x, is_split_into_words=True)
    #allows us to locate where the target word is
    for i, word in enumerate(sent):
        #Once we've reached the target word, the length of the token list (i.e final index + 1), will be the first part of the tokenized version of our target word
        if i == idx:
          new_idx = len(tokens)
          #can break as we actually don't need to tokenize anymore, and iteration is inefficient
          break
        tokens += tokenizer.tokenize(word)
    #create embeddings
    inputs = tokenizer.encode(words, return_tensors="pt", is_split_into_words=True)
    with torch.no_grad():
        embed = model(inputs)
    #+1 because beginning of sentence token
    return embed[0][0][new_idx + 1]

def embed_bert(df):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    return df.apply(lambda x: embeddings_model(model, tokenizer, x['sent'], x['idx']), axis=1)
