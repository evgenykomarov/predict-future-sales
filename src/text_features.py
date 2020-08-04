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


def text_feature_to_vec(text_features, D=4, epochs=100, learning_rate=0.01):
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
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    # This builds the model for the first time:
    model.fit(x, x, batch_size=min(100, n_samples), epochs=epochs)
    model.summary()
    print('=' * 30)
    print(s.count_nonzero())
    print("input dim = %s %s" % s.shape)
    print('=' * 30)
    
    extractor = keras.Model(inputs=model.inputs, outputs=model.layers[1].output)
    
    return extractor(x).numpy().round(2).astype(np.float32)


def embed_categorical_features(norm=True):
    items = pd.read_csv('../data/items.csv')
    shops = pd.read_csv('../data/shops.csv')
    cats = pd.read_csv('../data/item_categories.csv')


    from nltk.stem.snowball import SnowballStemmer
    stemmer_ru = SnowballStemmer("russian")
    stemmer_en = SnowballStemmer("english")
    # stemmer_ru.stem('дракона')

    # D = 4 seems mapping different items to same vector too often
    # D = 5 starts to distinguish
    items.item_name = [norm_text(x) for x in items.item_name.values]
    item_features_filtered = filter_vocab([' '.join(stemmer_en.stem(stemmer_ru.stem(y)) for y in x.split(' '))
                                           for x in items.item_name.values], thresh=30)
    item_features = text_feature_to_vec(item_features_filtered, D=5, epochs=50, learning_rate=0.005)
    # show_neighbors(items.item_name.values, item_features, i=13691)
    show_neighbors(items.item_name.values, item_features)
    # len([x for x in item_features_filtered if 'керамичесая' in x])
    # len(np.unique(item_features_filtered))
    if norm:
        item_features = (item_features - item_features.min(axis=0)) / (item_features.max(axis=0) - item_features.min(axis=0))
        item_features = item_features.round(3)

    shops.shop_name = [norm_text(x) for x in shops.shop_name.values]
    shop_features_filtered = filter_vocab(shops.shop_name, thresh=1)
    shop_features = text_feature_to_vec(shop_features_filtered, D=3, epochs=500, learning_rate=0.01)
    show_neighbors(shops.shop_name.values, shop_features)
    if norm:
        shop_features = (shop_features - shop_features.min(axis=0)) / (shop_features.max(axis=0) - shop_features.min(axis=0))
        shop_features = shop_features.round(3)

    cats.item_category_name = [norm_text(x) for x in cats.item_category_name.values]
    cat_features_filtered = filter_vocab(cats.item_category_name, thresh=1)
    cat_features = text_feature_to_vec(cat_features_filtered, D=3, epochs=500, learning_rate=0.02)
    show_neighbors(cats.item_category_name.values, cat_features)
    if norm:
        cat_features = (cat_features - cat_features.min(axis=0)) / (cat_features.max(axis=0) - cat_features.min(axis=0))
        cat_features = cat_features.round(3)

    def merge_feats(df, idx_col, name, features):
        _ = pd.DataFrame(features, columns=['%s_%s' % (name, i) for i in range(features.shape[1])])
        _[idx_col] = df[idx_col].values
        df = df.merge(_, on=idx_col, how='left')
        return df

    items = merge_feats(items, 'item_id', 'item', item_features)
    shops = merge_feats(shops, 'shop_id', 'shop', shop_features)
    cats = merge_feats(cats, 'item_category_id', 'cat', cat_features)
    items.to_csv('../data/items_nn.csv', index=False)
    shops.to_csv('../data/shops_nn.csv', index=False)
    cats.to_csv('../data/item_categories_nn.csv', index=False)


def text_feature_to_vec3(text_features, D=4, epochs=100, learning_rate=0.01):
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

    inputs = keras.Input(shape=(N, ))
    encode = layers.Dense(D, activation='relu')
    decode = layers.Dense(N, activation='relu')
    # activity_regularizer=keras.regularizers.l2(1e-14)
    i0 = inputs
    i1 = encode(i0)
    outputs = decode(i1)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    # This builds the model for the first time:
    model.fit(x, x, batch_size=min(100, n_samples), epochs=epochs)
    model.summary()
    print('=' * 30)
    print(s.count_nonzero())
    print("input dim = %s %s" % s.shape)
    print('=' * 30)

    extractor = keras.Model(inputs=model.inputs, outputs=model.layers[1].output)

    return extractor(x).numpy().round(2).astype(np.float32)


def filter_vocab(text_features, thresh=1):
    """
    text_features: list(str): corpus of captions
    returns: list(str) where strings are filtered to words > thesh counts
    """
    vocab = Counter()
    text_features_words = [x.split(' ') for x in text_features]
    for words in text_features_words:
        for word in words:
            vocab[word] += 1
    vocab = {k: v for k, v in vocab.items() if v > thresh}
    return [' '.join([word for word in text if word in vocab]) for text in text_features_words]


def text_feature_to_sparse(data_df, cat_df, cat_col='item_name', merge_cols=['item_id'], thresh=0):
    text_features_filtered = np.array(filter_vocab(cat_df[cat_col].values, thresh=thresh))
    reconstruct_df = cat_df[merge_cols].copy()
    reconstruct_df[cat_col] = range(len(text_features_filtered))
    text_features_idx = pd.merge(data_df, reconstruct_df, on=merge_cols, how='left')[cat_col].values

    _ = feature_extraction.text.CountVectorizer()
    text_features_sparse = _.fit_transform(text_features_filtered).astype(np.int8)
    text_features_sparse[text_features_idx]
    return reconstruct_df, text_features_filtered, text_features_idx, text_features_sparse[text_features_idx]


def text_feature_to_vec2():
    items = pd.read_csv('../data/items.csv')
    shops = pd.read_csv('../data/shops.csv')
    cats = pd.read_csv('../data/item_categories.csv')

    items.item_name = [norm_text(x) for x in items.item_name.values]
    shops.shop_name = [norm_text(x) for x in shops.shop_name.values]
    cats.item_category_name = [norm_text(x) for x in cats.item_category_name.values]

    # D = 4 seems mapping different items to same vector too often
    # D = 5 starts to distinguish
    # matrix = pd.read_parquet('../data_tmp/data_2020-06-14_01.snappy')
    matrix = pd.read_parquet('../data_tmp/features_20209716_4.snappy')
    date_col = 'date_block_num'
    target = 'item_cnt_month'
    id_cols = ['item_category_id', 'shop_id', 'item_id',]
    drop_cols = ['cat_0', 'cat_1', 'cat_2', 'ID', 'item_0', 'item_1', 'item_2', 'item_3', 'item_4', 'price', 'shop_0', 'shop_1', 'shop_2', ]
    matrix.drop(drop_cols, axis=1, inplace=True)
    # x_train = matrix[matrix[date_col] < 22].drop([target, date_col], axis=1)
    t0 = 11

    train_index = matrix[(matrix[date_col] > t0) & (matrix[date_col] < 34)].index
    items_reconstruct, items_filtered, items_idx, items_features_sparse = text_feature_to_sparse(matrix.loc[train_index], items, cat_col='item_name', merge_cols=['item_id'], thresh=1)
    cats_reconstruct, cats_filtered, cats_idx, cats_features_sparse = text_feature_to_sparse(matrix.loc[train_index], cats, cat_col='item_category_name', merge_cols=['item_category_id'], thresh=0)
    shops_reconstruct, shops_filtered, shops_idx, shops_features_sparse = text_feature_to_sparse(matrix.loc[train_index], shops, cat_col='shop_name', merge_cols=['shop_id'], thresh=0)

    ts = matrix[date_col].unique()
    ts = ts[ts > t0]
    unseen = defaultdict(dict)
    cat_names = ['item_id', ] #'item_category_id', 'shop_id']
    for t in sorted(ts):
        for cat_name in cat_names:
            _t = matrix[(matrix[date_col] > t0 - 1) & (matrix[date_col] == t)][cat_name].unique()
            _before_t = matrix[(matrix[date_col] > t0 - 1) & (matrix[date_col] < t)][cat_name].unique()
            unseen[cat_name][t] = np.setdiff1d(_t, _before_t)

    cat_name = 'item_id'
    nan_index = []
    for t in sorted(unseen[cat_name]):
        _ = np.where(
            train_index.isin(matrix[(matrix[date_col] == t) & matrix[cat_name].isin(unseen[cat_name][t])].index)
        )[0]
        nan_index.append(_)
    nan_index = np.hstack(nan_index)
    nan_index.sort()

    rows = items_features_sparse.nonzero()[0][nan_index]
    cols = items_features_sparse.nonzero()[1][nan_index]
    items_features_sparse.count_nonzero()
    items_features_sparse[rows, cols] = 0
    items_features_sparse.count_nonzero()

    target = 'item_cnt_month'
    def convert_sparse_matrix_to_sparse_tensor(X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    b = []
    rest_features = matrix.columns.drop(id_cols + [target, date_col])
    for c in tqdm(rest_features):
        _ = (matrix.loc[train_index, c].fillna(0) - matrix.loc[train_index, c].fillna(0).mean()) / (1e-6 + matrix.loc[train_index, c].fillna(0).std())
        b.append(_.values)

    features_tf = [tf.sparse.reorder(convert_sparse_matrix_to_sparse_tensor(items_features_sparse)),
                   tf.sparse.reorder(convert_sparse_matrix_to_sparse_tensor(cats_features_sparse)),
                   tf.sparse.reorder(convert_sparse_matrix_to_sparse_tensor(shops_features_sparse)),
                   np.vstack(b).T]
    cat_emb_dims = [5, 3, 3]
    y_train = matrix.loc[train_index, target].values

    emb_models = []
    for cat_features, cat_emb_dim in zip(features_tf[:-1], cat_emb_dims):
        sparse_size = cat_features.shape[1]
        emb_input = keras.Input(shape=(sparse_size, ))
        encode = layers.Dense(cat_emb_dim, activation=None) # 'relu'
        emb_outputs = encode(emb_input)
        emb_model = keras.Model(inputs=emb_input, outputs=emb_outputs)
        emb_models.append(emb_model)

    rest_input = keras.Input(shape=(features_tf[-1].shape[1], ))
    comb_input = layers.concatenate([x.output for x in emb_models] + [rest_input])
    denses = [layers.Dense(sum(cat_emb_dims) + features_tf[-1].shape[1], activation='relu') for i in range(6)] + [layers.Dense(1, activation='relu')]
    o = comb_input
    for d in denses:
        o = d(o)
    model = tf.keras.Model(inputs=[x.input for x in emb_models] + [rest_input], outputs=o)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    # This builds the model for the first time:
    del matrix;
    del cat_features;
    del items_filtered;
    gc.collect()
    model.fit(features_tf, y_train, batch_size=10000, epochs=10)
    model.summary()
    # 643/643 [==============================] - 34s 53ms/step - loss: 0.4577
    maps = []
    for i, df, df_sparse, name, index_name in zip(range(3),
                                        [items_idx, cats_idx, shops_idx],
                                        [items_features_sparse, cats_features_sparse, shops_features_sparse],
                                        ['item', 'cat', 'shop'],
                                        ['item_id', 'item_category_id', 'shop_id']):
        mapa = pd.Series({j: i for i, j in enumerate(df)}).sort_index()
        extractor = keras.Model(inputs=emb_models[i].inputs, outputs=model.layers[3 + i].output)
        _ = pd.DataFrame(extractor.predict(df_sparse[mapa.values]),
                         index=mapa.index).astype(np.float16)
        _.columns = ['%s_%s' % (name, i) for i in range(_.shape[1])]
        _.index.name = index_name
        _.reset_index(inplace=True)
        # cat_df = cat_df.merge(_, on=index_name, how='left')
        maps.append(_)
    items = items.merge(maps[0], on='item_id', how='left')
    cats = cats.merge(maps[1], on='item_category_id', how='left')
    shops = shops.merge(maps[2], on='shop_id', how='left')

    items.to_csv('../data/item_nn2.csv', index=False)
    cats.to_csv('../data/item_categories_nn2.csv', index=False)
    shops.to_csv('../data/shops_nn2.csv', index=False)


def show_neighbors(item_names, item_features, i=None):
    # item_names = np.array of n_samples names
    # item_features = np.array of n_samples x D encoded features
    # i = item to show neighbors for
    
    if i is None:
        i = np.random.choice(range(len(item_names)))
        print(i)
    
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
