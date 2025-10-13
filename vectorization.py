#!/usr/bin/env python
# coding: utf-8

# # **VECTORISATION DES PHRASES**

# In[19]:


import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from collections import Counter
import json
import joblib


# In[20]:


def tokenize(sentence, n=4):
  words = sentence.strip()
  return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]


# In[21]:


def build_vocabulary(corpus, K=5000, n=4):

    counter = Counter()

    for sentence in corpus:
        tokens = tokenize(sentence, n)
        counter.update(tokens)

    most_common = counter.most_common(K)
    vocab = {token: idx for idx, (token, _) in enumerate(most_common)}
    return vocab


# In[22]:


def compute_tf(tokens, vocab):
    tf = np.zeros(len(vocab))
    for token in tokens:
        if token in vocab:
            tf[vocab[token]] += 1
    if tf.sum() > 0:
        tf = tf / tf.sum()
    return tf


# In[23]:


def compute_tdf(corpus, vocab, n=4):
    N = len(corpus)
    idf = np.zeros(len(vocab))

    for sentence in corpus:
        tokens = set(tokenize(sentence, n))
        for token in tokens:
            if token in vocab:
                idf[vocab[token]] += 1

    idf = np.log((N + 1) / (idf + 1)) + 1
    return idf


# In[24]:


def vectorize(corpus, K=5000, n=4, vocab=None, idf=None):
    X = []
    tu = []
    if vocab == None:
        vocab = build_vocabulary(corpus, K=K, n=n)
        idf = compute_tdf(corpus, vocab, n)
        tu.append(vocab)
        tu.append(idf)
    for sentence in corpus:
        tokens = tokenize(sentence, n=n)
        tf = compute_tf(tokens, vocab)
        X.append(tf * idf)

    tu.append(X)    
    return tuple(tu)


# In[25]:


def vectorize_y(Y):
    Y_unique = set(Y)
    Y_set = {}
    for index, val in enumerate(Y_unique):
        Y_set[val] = index
    Y_final = []
    for y in Y:
        Y_final.append(Y_set[y])
    return np.array(Y_final), Y_set


# In[26]:


def save_X( X_vect, file="/content/drive/MyDrive/new_dataX.npy"):
    np.save(file, X_vect)
    print("data X sauvegardée")


# In[27]:


def save_vocab(vocab, file="/content/drive/MyDrive/vocab.json"):
    with open(file, "w") as f:
        json.dump(vocab, f)
    print("Vocabulaire X sauvegardée")


# In[28]:


def save_idf(idf_vect, file="/content/drive/MyDrive/idfVect.npy"):
    np.save(file, idf_vect)
    print("idf vecteur Y sauvegardée")


# In[29]:


def save_Y(Y, file="/content/drive/MyDrive/dataY.npy"):
    np.save(file, Y)
    print("data Y sauvegardée")


# In[30]:


def save_vocabY(Y, file="/content/drive/MyDrive/vocabY.json"):
    with open(file, "w") as f:
        json.dump(Y, f)
    print("Vocabulaire Y sauvegardée")


# In[31]:


def vectorize_one(text):
    sentence = [text]

    idf = np.load("idfFit.npy")
    json_path = "vocabFit.json"
    with open(json_path, 'r') as f:
        vocabX = json.load(f)
        
    
    sentVect = vectorize(sentence, K=5000, n=4, vocab=vocabX, idf=idf )
    sentVect = sentVect[0]
    
    svd = joblib.load("svdmodel2.plk")
    
    sent = svd.transform(sentVect)
    print(sent.shape)
    
    
    
    return sent

