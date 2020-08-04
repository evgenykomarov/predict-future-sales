# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 17:19:08 2020

@author: komarov
"""


import pandas as pd
import json
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

def plot_feature_pair_splits(gain_cnt2, thresh_cnt2, n_top=5):
    gain_cnt2.sort_values(inplace=True)
    for pair in gain_cnt2.index[::-1][:n_top]:
        pair_thresh = thresh_cnt2[pair]
        heatmap, xedges, yedges = np.histogram2d(pair_thresh[:, 0], pair_thresh[:, 1], bins=10)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.figure()
        plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.xlabel(pair[0])
        plt.ylabel(pair[1])
        plt.show()


def plot_feature_splits(gain_cnt, thresh_cnt, n_top=5):
    gain_cnt.sort_values(inplace=True)
    for feat in gain_cnt.index[::-1][:n_top]:
        plt.figure()
        plt.title('splits %s' % feat)
        bins, xedges = np.histogram(thresh_cnt[feat], bins=25)
        plt.hist(thresh_cnt[feat], xedges)
        plt.show()


def parse_xgb_interaction(model):
    """
    Returns:
    gain_cnt/gain_cnt2 = dict of feature/pair gain
    thresh_cnt/thresh_cnt2 = dict of feature/pair splits
    freq_cnt2 = dict of pair split frequencies
    """
    booster = model.get_booster()

    thresh_cnt = defaultdict(list) # list of thresholds per feature
    gain_cnt = Counter() # total gain per feature
    thresh_cnt2 = defaultdict(list) # list of thresholds per feature pair
    gain_cnt2 = Counter() # total gain per feature pair
    for string in tqdm(booster.get_dump(with_stats=True, dump_format='json')):
        tree_node = json.loads(string)
        stack = [(tree_node, 0, '', -1)]  # (root node id, impurity gain, parent_feature, parent_threshold)
        while len(stack) > 0:
            tree_node, parent_gain, parent_feature, parent_thresh = stack.pop()
            tree_node.keys()
            if 'split' in tree_node:
                feature = tree_node['split']
                thresh = tree_node['split_condition']
                gain = tree_node['gain']

                thresh_cnt[feature].append(thresh)
                gain_cnt[feature] += gain
                thresh_cnt2[(parent_feature, feature)].append((parent_thresh, thresh))
                gain_cnt2[(parent_feature, feature)] += parent_gain + gain
                for c in tree_node['children']:
                    stack.append((c, gain, feature, thresh))
    gain_cnt = pd.Series({f: v / len(thresh_cnt[f]) for f, v in gain_cnt.items()})
    gain_cnt /= gain_cnt.sum()
    thresh_cnt = {f: np.array(vs) for f, vs in thresh_cnt.items()}

    # reducing to sorted pairs of feature names
    pairs = set()
    [pairs.add(tuple(sorted(pair))) for pair in gain_cnt2 if '' not in pair]
    gain_cnt2 = {pair: gain_cnt2[pair] + gain_cnt2[pair[::-1]] for pair in pairs}
    thresh_cnt2 = {pair: np.array(thresh_cnt2[pair] + thresh_cnt2[pair[::-1]][::-1]) for pair in pairs}
    # normalizing by number of occurrences of feature pair
    gain_cnt2 = pd.Series({f: v / thresh_cnt2[f].shape[0] for f, v in gain_cnt2.items()})
    gain_cnt2 /= gain_cnt2.sum()
    gain_cnt2.sort_values(inplace=True)
    # pair gain by frequency of split
    freq_cnt2 = pd.Series({pair: values.shape[0] for pair, values in thresh_cnt2.items()})
    freq_cnt2 /= freq_cnt2.sum()
    freq_cnt2.sort_values(inplace=True)

    return gain_cnt, thresh_cnt, gain_cnt2, thresh_cnt2, freq_cnt2


def calc_score_drop(n, model, importances, X_valid, Y_valid, n_perms=1, clip_down=-np.inf, clip_up=np.inf):
    score_drop = {}
    importances.sort_values(inplace=True)
    features = importances.index.values
    val_pred = model.predict(X_valid)
    base_score = mean_squared_error(Y_valid, val_pred.clip(clip_down, clip_up))

    # only worst n importances to show
    for feature in tqdm(features[:(n if n is not None else None)]):
        temp_df = X_valid.copy()
        cur_score = []
        for _ in range(n_perms):
            temp_df[feature] = np.random.permutation(temp_df[feature])
            val_pred = model.predict(temp_df)
            cur_score.append(mean_squared_error(Y_valid, val_pred.clip(clip_down, clip_up)))
        _m = np.array(cur_score).mean()
        _s = np.array(cur_score).std()
        score_drop[feature] = (_m - base_score, (_m - base_score) / (_s + 1e-10))
    score_drop = pd.DataFrame({f: {'mean_drop': a, 'tstat_drop': b}
                               for f, (a, b) in score_drop.items()}).T.sort_values('mean_drop')
    return score_drop


class FeatureStatisticsRF(object):
    """
    fit a simple tree based model and evaluate feature by
    importance on train set
    drop score on validation set
    """
    def __init__(self, X_train, Y_train, X_valid, Y_valid, **model_kwargs):
        # clip_low=-np.inf, clip_high=np.inf, n_top=5
        # model_kwargs = dict(n_estimators=10, max_depth=10, max_features='sqrt', max_samples=0.8, random_state=seed, n_jobs=-1)
        ts = time.time()
        self.model = XGBRegressor(**model_kwargs)
        eval_metric = 'rmse'
        self.model.fit(
            X_train,
            Y_train,
            eval_metric=eval_metric,
            eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
            verbose=True)
        print('trained in %.02f sec' % (time.time() - ts))

        # plot training progress
        self.err = pd.concat([pd.DataFrame(self.model.evals_result_['validation_%s' % i]) for i in range(2)], axis=1)
        self.err.columns = ['%s_%s' % (x, eval_metric) for x in ['train', 'val']]
        self.err.plot(); plt.show()

        # plot importances
        self.importances = pd.Series({f: v for f, v in zip(X_train.columns, self.model.feature_importances_)}).sort_values()
        self.gain_cnt, self.thresh_cnt, self.gain_cnt2, self.thresh_cnt2, self.freq_cnt2 = parse_xgb_interaction(self.model)
        self.score_drop = calc_score_drop(50, self.model, self.gain_cnt, X_valid, Y_valid, n_perms=1)

    def show_plots(self, n_importances=30, n_pair_importances=20):
        features_show = self.importances.index[-n_importances:]
        plt.figure()
        plt.title('Feature Importances')
        plt.barh(range(len(features_show)), self.importances.loc[features_show].values, color='b', align='center')
        plt.yticks(range(len(features_show)), features_show)
        plt.xlabel('Relative Importance')
        plt.show()

        # plot interactions
        gc2 = self.gain_cnt2.copy()
        gc2.index = [', '.join(pair) for pair in gc2.index]
        features_show = gc2.index[-n_pair_importances:]
        plt.figure()
        plt.title('Feature Interactions')
        plt.barh(range(len(features_show)), gc2.loc[features_show].values, color='b', align='center')
        plt.yticks(range(len(features_show)), features_show)
        plt.xlabel('Relative Importance')
        plt.show()

        plot_feature_splits(self.gain_cnt, self.thresh_cnt)
        plot_feature_pair_splits(self.gain_cnt2, self.thresh_cnt2)


if __file__ == '__main__':
    data = pd.read_parquet('../data_tmp/tmp.snappy')
    t_start = 11
    t_valid = [22]
    target = 'item_cnt_month'
    drop_cols=['ID', 'shop_0', 'shop_1', 'shop_2', 'cat_0', 'cat_1', 'cat_2',
               'item_0', 'item_1', 'item_2', 'item_3', 'item_4']
    add_drop = ['date_block_num', 'item_id']
    data = data[data.date_block_num > t_start]
    data.drop(drop_cols, axis=1, inplace=True)
    data.info()
    data.fillna(0, inplace=True)
    X_train = data[data.date_block_num < min(t_valid)].drop([target] + add_drop, axis=1)
    Y_train = data[data.date_block_num < min(t_valid)][target]
    X_valid = data[data.date_block_num.isin(t_valid)].drop([target] + add_drop, axis=1)
    Y_valid = data[data.date_block_num.isin(t_valid)][target]

    del data; gc.collect()
    model_kwargs = dict(max_depth=10, n_estimators=10, num_parallel_tree=10, min_child_weight=300,
                        colsample_bytree=1 / np.sqrt(X_train.shape[1]), subsample=0.8, eta=1, seed=42, tree_method='gpu_hist',)

    fs = FeatureStatisticsRF(X_train, Y_train, X_valid, Y_valid, **model_kwargs)
    # fs.show_plots(30, 20)
    fs.score_drop[fs.score_drop.mean_drop <= 0]
    fs.err
    fs.freq_cnt2

