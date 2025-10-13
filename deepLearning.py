#!/usr/bin/env python
# coding: utf-8

# # **Reseau de Neuronne**

# In[61]:


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from vectorization import vectorize_one


# In[65]:


def predict(text, params, vocabY, norm=False):
    
    x = vectorize_one(text)
    if norm:
        x_norm = normalizeOne(x)
    else:
        x_norm = x
    Activations = forward_propagation(x_norm, params)
    A = Activations['A'+str(len(params) // 2)]
    result = np.argmax(A[:,:1])
    print("L argmax est ", result)
    lan = ""
    for k,v in vocabY.items():
        if result == v:
            print("Langue trouve !! ") 
            return k

    print("Langue introuvable !! ")


# In[3]:


def hot_ones(y, length=None):
    if length == None:
        length = len(np.unique(y))
    if not y.shape:
        size = 1
    else:
        size = y.shape[0]
        
    y_hot_ones = np.zeros((size, length))
    for i in range(size):
        val = y[i]
        y_hot_ones[i, val] = 1
    return y_hot_ones.T


# In[4]:


def divideData(X, Y , ratio=0.2):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_size = int(X.shape[0] * ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    return X_train, X_test, Y_train, Y_test


# In[5]:


def loadData(file="/content/drive/MyDrive/"):
    X = np.load(file + "Xfit.npy")
    y = np.load(file + "Yfit.npy")
    idf = np.load(file + "idfFit.npy")
    json_path = file + "vocabFit.json"
    with open(json_path, 'r') as f:
        vocabX = json.load(f)

    json_path = file + "vocabYfit.json"
    with open(json_path, 'r') as f:
        vocabY = json.load(f)
    return X, y, idf, vocabX, vocabY


# In[6]:


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)


# In[7]:


def sigmoid(Z):
    return 1 / 1 + np.exp(-Z)

def sigmoid_derivative(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))


# In[8]:


def softmax(vect):
    maxval = np.max(vect, axis=0)
    exps = np.sum(np.exp(vect-maxval), axis=0,  keepdims = True)
    return np.exp(vect-maxval) / exps


# In[9]:


def accuracy(Y, A):
    true_classes = np.argmax(Y, axis=0)
    pred_classes = np.argmax(A, axis=0)
    return np.mean(true_classes == pred_classes)*100


# In[10]:


def check_values(dictionnaire):
    for cle,valeur in dictionnaire.items():
        print(cle, valeur.shape)
    print()


# In[11]:


def cross_entropy(activations, y, params):
    epsi = 1e-15
    couches = (len(params) // 2)
    A = activations['A' + str(couches)]
    m = y.shape[1]
    return np.sum(y * np.log(A+epsi))/ -m
    #return()


# In[12]:


def initialiser(dim):

    couches = len(dim) - 1
    parametres = {}
    for couche in range(couches):
        parametres['W' + str(couche+1)] = np.random.randn(dim[couche+1],dim[couche])*np.sqrt(2 / dim[couche])
        parametres['B' + str(couche+1)] = np.zeros((dim[couche+1], 1))
    return parametres


# In[13]:


def normalizeData(X, y, train=False):
    if train:
        X_max , X_min = X.max(axis=0) , X.min(axis=0)
        np.save('norm.npy', np.array([X_min, X_max]))
    else:
        X_norm = np.load('norm.npy')
        X_max , X_min = X_norm[1] , X_norm[0]
        
    X_train = (X - X_min) * 1/(X_max-X_min)    
    y = y.reshape(y.shape[0], 1)
    X_train = X_train.T
    y_hot = hot_ones(y, length=22)
    print(X_train.shape, y_hot.shape)
    return X_train, y_hot

def normalizeOne(X):
    X_norm = np.load('norm.npy')
    X_max , X_min = X_norm[1] , X_norm[0]
    
    X_train = (X - X_min) * 1/(X_max-X_min)
    X_train = X_train.T
    print(X_train.shape)
    return X_train


# In[14]:


def forward_propagation(X, parametres):

    couches = len(parametres) // 2
    activations = {
        'A0' : X
    }
    for couche in range(couches-1):
        W = parametres['W' + str(couche+1)]
        B = parametres['B' + str(couche+1)]
        A = activations['A' + str(couche)]
        Z = W.dot(A) + B
        activations['Z'+str(couche+1)] = Z
        activations['A' + str(couche+1)] = relu(Z)



    W = parametres['W' + str(couches)]
    B = parametres['B' + str(couches)]
    A = activations['A' + str(couches-1)]
    Z = W.dot(A) + B
    activations['Z'+str(couches)] = Z
    activations['A' + str(couches)] = softmax(Z)


    return activations


# In[15]:


def back_propagation(activations, y, parametres):

    m = y.shape[1]
    couches = len(parametres) // 2
    dZ = activations['A' + str(couches)] - y
    gradients = {}

    for i in range(couches):
        couche = couches-i
        A = activations['A' + str(couche-1)]
        gradients['dW' + str(couche)] = (1/m) * dZ.dot(A.T)
        gradients['dB' + str(couche)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        if couche>1:
            W = parametres['W' + str(couche)]
            Z = activations['Z'+str(couche-1)]
            dZ = (W.T).dot(dZ) * relu_derivative(Z)


    return gradients


# In[16]:


def update_params(parametres, gradients, learning_rate):

    couches = len(parametres) // 2

    for c in range(couches):
        couche = c+1
        W = parametres['W' + str(couche)]
        dW = gradients['dW' + str(couche)]
        parametres['W' + str(couche)] = W - (learning_rate * dW)

        B = parametres['B' + str(couche)]
        dB = gradients['dB' + str(couche)]
        parametres['B' + str(couche)]  = B - (learning_rate * dB)

    return parametres


# In[17]:


def neuronal_network(X, y, dim, n_iter, learning_rate):

    params = initialiser(dim)
    print("Les parametres")
    check_values(params)
    loss = []

    for i in tqdm(range(n_iter)):

        activations = forward_propagation(X, params)

        if i%1000:
            loss.append(cross_entropy(activations, y, params))

        gradients = back_propagation(activations, y, params)
        params = update_params(params, gradients, learning_rate)


    plt.figure(figsize=(12,7))
    plt.plot(loss)
    plt.show
    return params, loss


# In[18]:


def neuronal_network_mini_batch(X, y, dim, epoch, batch_size, learning_rate, params=None):

    if params == None:
        params = initialiser(dim)
        print("Generation des parametres")
    else:
        print("Les parametres precedants charges")
    print("Les parametres")
    check_values(params)
    loss = []


    for ep in tqdm(range(epoch)):
        mini_batches = create_mini_batches(X, y, batch_size=batch_size, shuffle=True)

        for X_batch, y_batch in mini_batches:
            activations = forward_propagation(X_batch, params)
            gradients = back_propagation(activations, y_batch, params)
            params = update_params(params, gradients, learning_rate)


        activations_full = forward_propagation(X, params)
        loss.append(cross_entropy(activations_full, y, params))

    # afficher la courbe de perte
    plt.figure(figsize=(12,7))
    plt.plot(loss)
    plt.xlabel("Itérations (x500)")
    plt.ylabel("Loss")
    plt.show()

    return params, loss


# In[19]:


def create_mini_batches(X, y, batch_size=64, shuffle=True):
    
    m = X.shape[1] 
    mini_batches = []

    if shuffle:
        permutation = np.random.permutation(m)
        X = X[:, permutation]
        y = y[:, permutation]

    num_complete_batches = m // batch_size

    for k in range(num_complete_batches):
        X_batch = X[:, k*batch_size:(k+1)*batch_size]
        y_batch = y[:, k*batch_size:(k+1)*batch_size]
        mini_batches.append((X_batch, y_batch))

    
    if m % batch_size != 0:
        X_batch = X[:, num_complete_batches*batch_size:]
        y_batch = y[:, num_complete_batches*batch_size:]
        mini_batches.append((X_batch, y_batch))

    return mini_batches


# In[67]:


X , y, idf, vocabX, vocabY = loadData(file="")


# In[73]:


text = "il faudra de la fois et du courage, une relation remplie de rage qui demarre sur des bases pourries"
json_path = "vocabYfit.json"
with open(json_path, 'r') as f:
    vocabY = json.load(f)
params = np.load("parametres4.npy", allow_pickle=True).item()
lan = predict(text, params, vocabY, norm=True)
print(lan)


# In[22]:


"""X_div, X_test, Y, y_test = divideData(X, y)
print("X train : ", X_div.shape)
print("Y train : ", Y.shape)
print("X test : ", X_test.shape)
print("Y test : ", y_test.shape)"""


# In[23]:


"""""X_train, y_hot = normalizeData(X_div, Y)"""


# In[64]:


def mainFitting():
    params0 = np.load("parametres4.npy", allow_pickle=True).item()
    print(type(params0))
    check_values(params0)

    dim = [512, 256, 128]
    dim.insert(0, X_train.shape[0])
    dim.append(y_hot.shape[0])
    batch_size = 128
    epoch = 500
    learning_rate = 0.01
    params, loss = neuronal_network_mini_batch(X_train, y_hot, dim, epoch, batch_size, learning_rate, params=params0)
    np.save("parametres5.npy", params)
    print("Parametres enregistrés")
    print(loss[-1])


# In[ ]:


def test(X, y, X_train, y_hot, X_test, y_test):
    params0 = np.load("parametres4.npy", allow_pickle=True).item()
    X_testall, y_hotall = normalizeData(X,y, train=False)
    Actiations = forward_propagation(X_train, params0)
    A = Actiations['A'+str(len(params0) // 2)]
    acc = accuracy(y_hot, A)
    print("Accuracy : ", acc)

    X_test_norm, Y_test_norm = normalizeData(X_test, y_test, train=False)
    Actiationstest = forward_propagation(X_test_norm, params0)
    A = Actiationstest['A'+str(len(params0) // 2)]
    accur = accuracy(Y_test_norm, A)
    print("Accuracy : ", accur)

