#!/usr/bin/env python
# coding: utf-8

from deepLearning import predict
import json
import numpy as np




text = "il faudra de la fois et du courage, une relation remplie de rage qui demarre sur des bases pourries"
json_path = "vocabYfit.json"
with open(json_path, 'r') as f:
    vocabY = json.load(f)
params = np.load("parametres4.npy", allow_pickle=True).item()
lan = predict(text, params, vocabY, norm=True)
print(lan)





