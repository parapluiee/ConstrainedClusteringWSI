from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from huggingface_hub import hf_hub_download
import fasttext
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

def embed_camembert(df):
    tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    model = AutoModel.from_pretrained("camembert-base")
    return df.apply(lambda x: embeddings_model(model, tokenizer, x['sent'], x['idx']), axis=1)

def fasttext_emb(df):
    model_path = hf_hub_download(repo_id="facebook/fasttext-fr-vectors", filename="model.bin")
    model_ft = fasttext.load_model(model_path)
    return df.apply(lambda x: sentence_to_vector(x['sent'], x['idx'], model_ft), axis=1)

def sentence_to_vector(tokens, idx, model, window=5, vector_size=300):
    context = context_window(tokens, idx, window)
    vectors = []
    for word in context:
        vectors.append(model.get_word_vector(word))  # for fastText

    return np.mean(vectors, axis=0)

def context_window(tokens, idx, window_size=5):
    start = max(idx - window_size, 0)
    end = min(idx + window_size + 1, len(tokens))
    context = tokens[start:idx] + tokens[idx+1:end]
    return context
def build_vocab(sentences):
    vocab = set()
    for sentence in sentences:
        vocab.update(sentence)
    return list(vocab)
def embed_freq(df):
    freq_vectors = {}
    for lemma, group in df.groupby('lemma'):
        lemma_vocab = build_vocab(group['sent'])
        for index, row in group.iterrows():
            context = context_window(row['sent'], row['idx'])
            freq_vector = np.zeros(len(lemma_vocab))
            for word in context:
                if word in lemma_vocab:
                    freq_vector[lemma_vocab.index(word)] += 1
            freq_vectors[index] = freq_vector
    # Assign each vector to the correct row
    df['freq'] = df.index.map(freq_vectors)
    return df