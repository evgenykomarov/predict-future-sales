# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:52:40 2020

@author: komarov
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product, chain
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from collections import Counter, defaultdict
from sklearn import feature_extraction
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from xgboost import XGBRegressor
from xgboost import plot_importance
import seaborn as sns
import gc
import time
import sys
import pickle
import importlib


def text_feature_to_vec(text_features, D=4, epochs=100):
    """
    naive feature generation from textual categorical features:
    (a very naive non-linear PCA)
    
    feature = bag of words {w_i}
    1. transform all categories into bag of words sparse representation
    2. train simple autoencoder with bottleneck size D
    3. return the middle layer encoded features: D-dimensional

    # for checking similar features are neighbours:
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(acx, text_features)
    # print(neigh.predict(xpca[[0], :]))
    
    i = 58
    print(text_features[i])
    nei_shops = text_features[neigh.kneighbors(acx[[i], :])[1]]
    nei_dists = neigh.kneighbors(acx[[i], :])[0]
    print(nei_shops)
    print(nei_dists)
    
    text_features = items.item_name.values
    D = 5
    """
    _ = feature_extraction.text.CountVectorizer()
    s = _.fit_transform(text_features)
    x = s.toarray().astype(bool)
    n_samples, N = s.shape
    print('=' * 30)
    print(s.count_nonzero())
    print("input dim = %s %s" % s.shape)
    print('=' * 30)
    
    class DenseTranspose(keras.layers.Layer):
        def __init__(self, dense, activation, **kwargs):
            self.dense = dense
            self.activation = keras.activations.get(activation)
            super().__init__(**kwargs)
        
        def build(self, batch_input_shape):
            # self.biases = self.add_weight(name='bias',
            #                               shape=[self.dense.input_shape[-1]],
            #                               initializer='zeros')
            super().build(batch_input_shape)
            
        def call(self, inputs):
            x = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)# + self.biases
            return self.activation(x)
    
    inputs = keras.Input(shape=(N, ))
    encode = layers.Dense(D, activation='relu')
    # activity_regularizer=keras.regularizers.l2(1e-14)
    i0 = inputs
    i1 = encode(i0)
    outputs = DenseTranspose(encode, activation='relu')(i1)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    # This builds the model for the first time:
    model.fit(x, x, batch_size=min(100, n_samples), epochs=epochs)
    model.summary()
    print('=' * 30)
    print(s.count_nonzero())
    print("input dim = %s %s" % s.shape)
    print('=' * 30)
    
    extractor = keras.Model(inputs=model.inputs, outputs=model.layers[1].output)
    
    return extractor(x).numpy().round(2).astype(np.float32)


def show_neighbors(item_names, item_features, i=None):
    # item_names = np.array of n_samples names
    # item_features = np.array of n_samples x D encoded features
    # i = item to show neighbors for
    
    if i is None:
        i = np.random.choice(range(len(item_names)))
    
    neigh = KNeighborsClassifier(n_neighbors=20)
    neigh.fit(item_features, item_names)
    # i = 10822
    nei_dists, nei_indices = neigh.kneighbors(item_features[[i], :])
    nei_items = item_names[nei_indices.squeeze()]
    print(item_names[i])
    pairs = [(n, np.round(d, 3)) for n, d in zip(nei_items.squeeze(), nei_dists.squeeze())]
    print(*pairs, sep='\n')


def norm_text(x):
    # removes non-alpha numeric
    # lower case
    # strip
    # replace many spaces with 1
    import re
    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x.lower())
    x = re.sub('\s+', ' ', x)
    return x.strip()
