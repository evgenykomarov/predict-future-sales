# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 17:19:08 2020

@author: komarov
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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

seed = 123


def rf_err(model, X_train, Y_train, X_valid, Y_valid, clip_low=-np.inf, clip_high=np.inf):
    err_val = {}
    err_tr = {}
    pred_val = np.zeros(Y_valid.shape[0])
    pred_tr = np.zeros(Y_train.shape[0])
    for i in tqdm(range(model.n_estimators)):
        pred_val = (pred_val * i +  model.estimators_[i].predict(X_valid)) / (i + 1)
        pred_tr =  (pred_tr  * i +  model.estimators_[i].predict(X_train)) / (i + 1)
        err_val[i] = mean_squared_error(Y_valid.values, pred_val.clip(clip_low, clip_high))
        err_tr[i]  = mean_squared_error(Y_train.values, pred_tr.clip(clip_low, clip_high))
    
    err = pd.DataFrame({'tr': err_tr, 'val': err_val})
    print(err.iloc[-1].round(4))
    plt.figure()
    err.plot()
    plt.show()
    return err


def get_rf_importances(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-30:]
    plt.figure()
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    return {features[i]: importances[i] for i in indices}


def parse_rf_tree_interaction(t, gain_counter_glob, thresh_counter, thresh_counter2):
    # t: tree 
    # gc: global gain counter for pair interactions
    # thresh_counter = dafaultdict(list): global threshold counter for tree splits
    
    gain_counter = Counter()

    children_left = t.children_left
    children_right = t.children_right
    
    stack = [(0, 0, -1, -1)]  # (root node id, impurity gain, parent_node, parent_threshold)
    
    while len(stack) > 0:
        node_id, gain_parent, node_parent, thresh_parent = stack.pop()
        f0 = t.feature[node_id]
        fp = t.feature[node_parent]
        thresh_counter[f0].append(t.threshold[node_id])
        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            node_left = children_left[node_id]
            node_right = children_right[node_id]
            gain = t.weighted_n_node_samples[node_id] * t.impurity[node_id] - \
                    t.weighted_n_node_samples[node_left] * t.impurity[node_left] - \
                    t.weighted_n_node_samples[node_right] * t.impurity[node_right]
            if node_parent != -1:
                sort_index = np.argsort([f0, fp])
                ts = np.array([thresh_parent, t.threshold[node_id]])[sort_index]
                pair = tuple(sorted([f0, fp]))
                gain_counter[pair] += gain_parent + gain
                thresh_counter2[pair].append(ts)
            stack.append([node_left, gain, node_id, t.threshold[node_id]])
            stack.append([node_right, gain, node_id, t.threshold[node_id]])
            
    norm = sum(gain_counter.values())
    for pair in gain_counter:
        gain_counter[pair] /= norm
        gain_counter_glob[pair] += gain_counter[pair]
    
    return gain_counter


def get_splits(thresh_counter, f0, n=4):
    bins0, edges0 = np.histogram(thresh_counter[f0])
    idx = np.argsort(bins0)[-n:]
    res = np.array([edges0[i:i+1].mean() for i in idx])
    res.sort()
    return res


def get_rf_interactions(model, features, n_pairs=10, n_top=5):
        
    thresh_counter = defaultdict(list)
    thresh_counter2 = defaultdict(list)
    gain_counter = Counter()
    for i, estimator in tqdm(enumerate(model.estimators_)):
        print("parsing tree %s / %s" % (i + 1, model.n_estimators))
        t = estimator.tree_
        parse_rf_tree_interaction(t, gain_counter, thresh_counter, thresh_counter2)
        
    norm = sum(gain_counter.values())
    for pair in gain_counter:
        gain_counter[pair] /= norm    

    gain_counter = gain_counter.most_common()
    values = np.array([b for a, b in gain_counter[:n_pairs]])
    names = np.array([repr(tuple(features[np.array(a)])) for a, b in gain_counter[:n_pairs]])
    plt.figure()
    plt.title('Pair importances')
    plt.barh(range(len(values)), values[::-1], color='b', align='center')
    plt.yticks(range(len(values)), names[::-1])
    plt.xlabel('Relative Importance')
    plt.show()
    
    for feat in thresh_counter:
        thresh_counter[feat] = np.array(thresh_counter[feat])
    for pair in thresh_counter2:
        thresh_counter2[pair] = np.vstack(thresh_counter2[pair])

    best_splits2 = []
    for i in range(min(n_top, n_pairs)):
        pair = gain_counter[i][0]
        pair_thresh = thresh_counter2[pair]
        heatmap, xedges, yedges = np.histogram2d(pair_thresh[:, 0], pair_thresh[:, 1], bins=25)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        plt.figure()
        plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.xlabel(features[pair[0]])
        plt.ylabel(features[pair[1]])
        plt.show()    

        xi, yi = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
        threshs = 0.5 * (xedges[xi] + xedges[xi + 1]), 0.5 * (yedges[yi] + yedges[yi + 1])
        f0, f1 = features[pair[0]], features[pair[1]]
        best_splits2.append(((f0, f1), threshs))
    return gain_counter, thresh_counter, thresh_counter2, best_splits2


def calc_score_drop(n, model, _features, X_valid, Y_valid, base_score, n_perms=1, clip_down=None, clip_up=None):
    score_drop = {}
    importances = pd.Series({f: v for f, v in zip(_features, model.feature_importances_)}).sort_values()
    features = importances.index.values
    # going only through worst 100 features
    if n is None:
        gen = features
    else:
        gen = features[:min(n, len(features))]
    for ci in tqdm(gen):
        temp_df = X_valid.copy()
        cur_score = []
        for _ in range(n_perms):
            temp_df[ci] = np.random.permutation(temp_df[ci])
            val_pred = model.predict(temp_df)
            cur_score.append(mean_squared_error(Y_valid, val_pred.clip(clip_down, clip_up)))
        _m = np.array(cur_score).mean()
        _s = np.array(cur_score).std()
        score_drop[ci] = (_m - base_score, (_m - base_score) / (_s + 1e-10))
    score_drop = pd.DataFrame({f: {'mean_drop': a, 'tstat_drop': b} for f, (a, b) in score_drop.items()}).T.sort_values('mean_drop')
    return importances, score_drop


def feature_statistics(X_train, Y_train, X_valid, Y_valid, n_estimators=10, clip_low=-np.inf, clip_high=np.inf, n_top=5):
    
    print("building RF for %s tree" % n_estimators)
    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=10,
                                  max_features='sqrt',
                                  max_samples=0.8,
                                  random_state=seed,
                                  n_jobs=-1) # all cores
    model.fit(X_train, Y_train)
    
    err = rf_err(model, X_train, Y_train, X_valid, Y_valid, clip_low, clip_high)
    features = X_train.columns.values
    importances = get_rf_importances(model, features)
    gain_counter, thresh_counter, thresh_counter2, best_splits2 = get_rf_interactions(model, features, n_top=n_top)
    thresh_counter = {features[k]: v for k, v in  thresh_counter.items()}
    thresh_counter2 = {(features[k[0]], features[k[1]]): v for k, v in  thresh_counter2.items()}
    
    best_splits = []
    for feat, _ in sorted(importances.items(), key=lambda _: _[1], reverse=True)[:n_top]:
        plt.figure()
        plt.title('splits %s' % feat)
        bins, xedges = np.histogram(thresh_counter[feat], bins=25)
        plt.hist(thresh_counter[feat], xedges)
        splits3 = np.array([xedges[i: i + 2].mean() for i in np.argsort(bins)[-3:]])
        best_splits.append((feat, splits3))
        plt.show()
    
    return model, err, importances, features, gain_counter, thresh_counter, thresh_counter2, best_splits, best_splits2
