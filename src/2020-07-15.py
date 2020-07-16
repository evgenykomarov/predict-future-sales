# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 21:21:24 2020

@author: komarov
"""

# %cd "~/kaggle/competitive-data-science-predict-future-sales/src"
import sys

import gc
import importlib
import time
from itertools import product, chain
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from xgboost import XGBRegressor
from glob import iglob
import os
from glob import iglob

import utils_rf
import utils
import text_features
import knn_features


# from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.ensemble import RandomForestRegressor
# from collections import Counter, defaultdict
# from sklearn import feature_extraction
# from tensorflow import keras
from multiprocessing import Pool, Process
# from tensorflow.keras import layers
# from sklearn.decomposition import IncrementalPCA
# from sklearn.neighbors import NearestNeighbors
# from sklearn.neighbors import KNeighborsClassifier
# import tensorflow as tf
import pickle
# from xgboost import plot_importance

importlib.reload(utils_rf)
importlib.reload(utils)
importlib.reload(text_features)
importlib.reload(knn_features)

seed = 123


# x = pd.DataFrame([[1, 1, 1, 2, 2, 2], np.random.randn(6)]).T
# x.iloc[2, 1] = 0
# x.iloc[5, 1] = 0
# x.columns = list('ab')
# x.groupby('a')['b'].cumsum() - x['b']

def load_and_clean():

    train = pd.read_csv('../data/sales_train1.csv')
    train.date = train.date.astype('str')
    # train.to_csv('../data/sales_train1.csv', index=False)
    test = pd.read_csv('../data/test.csv')

    # utils.embed_categorical_features()

    items = pd.read_csv('../data/items_nn.csv')
    shops = pd.read_csv('../data/shops_nn.csv')
    cats = pd.read_csv('../data/item_categories_nn.csv')

    plt.figure(figsize=(10, 4))
    plt.xlim(-100, 3000)
    sns.boxplot(x=train.item_cnt_day)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.xlim(train.item_price.min(), train.item_price.max()*1.1)
    sns.boxplot(x=train.item_price)
    plt.show()

    train = train[train.item_price < 100000]
    train = train[train.item_cnt_day < 1001]

    # fill missing price with median
    train.loc[train.item_price < 0, 'item_price'] = train[(train.shop_id==32) & (train.item_id==2973) & (train.date_block_num==4) & (train.item_price > 0)].item_price.median()

    # monthly sales
    test_items = set(test.item_id)
    train_items = set(train.item_id)
    print(len(test_items - train_items), len(test_items), len(test))

    return train, test, items, shops, cats

def extend_train(train, test, items, shops, cats):
    #extending the train data every month with cartesian product of shops x items
    ts = time.time()
    matrix = []
    cols = ['date_block_num','shop_id','item_id']
    for i in tqdm(train.date_block_num.unique()):
        sales = train[train.date_block_num == i]
        _ = np.fromiter(chain(*product([i], sales.shop_id.unique(), sales.item_id.unique())), 'int16')
        _.shape = len(_) // 3, 3
        matrix.append(_)

    matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
    matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
    matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
    matrix['item_id'] = matrix['item_id'].astype(np.int16)
    matrix.sort_values(cols, inplace=True)
    time.time() - ts

    # calculating target for training: monthly sales per (shop_id, item_id)
    ts = time.time()
    group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
    group.columns = ['item_cnt_month']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=cols, how='left')
    matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                    .fillna(0)
                                    .clip(0,20) # NB clip target here
                                    .astype(np.float32))
    time.time() - ts

    test['date_block_num'] = 34
    test['date_block_num'] = test['date_block_num'].astype(np.int8)
    test['shop_id'] = test['shop_id'].astype(np.int8)
    test['item_id'] = test['item_id'].astype(np.int16)

    ts = time.time()
    matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
    matrix.fillna(0, inplace=True) # 34 month
    time.time() - ts

    # my special feature should shine or fail here!
    ts = time.time()
    matrix = pd.merge(matrix, shops.drop('shop_name', axis=1).astype(np.float32), on=['shop_id'], how='left')
    matrix = pd.merge(matrix, items.drop('item_name', axis=1).astype(np.float32), on=['item_id'], how='left')
    matrix = pd.merge(matrix, cats.drop('item_category_name', axis=1).astype(np.float32), on=['item_category_id'], how='left')
    # matrix['city_code'] = matrix['city_code'].astype(np.int8)
    matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
    matrix['ID'] = matrix['ID'].astype(np.int32)
    # matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)
    time.time() - ts

    return matrix

def mean_enc(matrix, train=None, group_cols=[], group_shorts=[], enc_col='item_cnt_month', agg_fn='mean'):
    names_map = {'mean': 'avg', 'sum': ''}
    if agg_fn != 'sum':
        _ = [names_map[agg_fn], enc_col]
    else:
        _ = [enc_col]
    enc_name = '_'.join(group_shorts + _)
    if train is not None:
        group = train.groupby(group_cols).agg({enc_col: [agg_fn]})
    else:
        group = matrix.groupby(group_cols).agg({enc_col: [agg_fn]})
    group.columns = [enc_name]
    group.reset_index(inplace=True)
    matrix = pd.merge(matrix, group, on=group_cols, how='left')
    matrix[enc_name] = matrix[enc_name].astype(np.float32)
    return matrix


def evaluate_features(matrix, t0=11, t1=22, drop_cols=['ID', 'price']):
    data = matrix[matrix.date_block_num > t0]
    data.drop(drop_cols, axis=1, inplace=True)
    data.info()
    data.fillna(0, inplace=True)
    X_train = data[data.date_block_num < t1].drop(['item_cnt_month', 'date_block_num'], axis=1)
    Y_train = data[data.date_block_num < t1]['item_cnt_month']
    X_valid = data[data.date_block_num == t1].drop(['item_cnt_month', 'date_block_num'], axis=1)
    Y_valid = data[data.date_block_num == t1]['item_cnt_month']
    # X_test = data[data.date_block_num == 34].drop(['item_cnt_month', 'date_block_num', ], axis=1)
    del data
    gc.collect()

    # [62]	validation_0-rmse:0.79811	validation_1-rmse:0.89230
    model, err, importances, features, gain_counter, thresh_counter, thresh_counter2, best_splits, best_splits2 = utils_rf.feature_statistics(X_train, Y_train, X_valid, Y_valid, n_estimators=10, clip_low=0, clip_high=20)
    importances, score_drop = utils_rf.calc_score_drop(None, model, X_train.columns, X_valid, Y_valid, base_score=err.iloc[-1, 1], clip_down=0, clip_up=20)
    return err, importances, score_drop, thresh_counter, best_splits2


def get_splits(thresh_counter, f0, n_bins=4):
    bins0, edges0 = np.histogram(thresh_counter[f0])
    idx = np.argsort(bins0)[-n_bins:]
    res = np.array([edges0[i:i+1].mean() for i in idx])
    res.sort()
    return res


def mean_encode_splits(matrix, thresh_counter, best_splits2, n, n_bins=4):
    for (f0, f1), _ in tqdm(best_splits2[:n]):
        print(f0, f1)
        mean_enc_name = '_'.join([f0, f1, 'avg', 'item'])
        bins0 = [-np.inf] + list(get_splits(thresh_counter, f0, n_bins)) + [np.inf]
        bins1 = [-np.inf] + list(get_splits(thresh_counter, f1, n_bins)) + [np.inf]
        matrix['f0'] = pd.cut(matrix[f0], bins0)
        matrix['f1'] = pd.cut(matrix[f1], bins1)
        group_cols = ['f0', 'f1']
        encoded_feature = utils.expanding_mean_encoding(matrix, 'item_cnt_month', group_cols)
        matrix.drop(group_cols, axis=1, inplace=True)
        matrix[mean_enc_name] = encoded_feature
        matrix = utils.lag_feature(matrix, [1], mean_enc_name)
        matrix.drop(mean_enc_name, axis=1, inplace=True)


def expanding_mean_encode_splits(matrix, thresh_counter, best_splits2, n, n_bins=4):
    for (f0, f1), _ in tqdm(best_splits2[:n]):
        print(f0, f1)
        mean_enc_name = '_'.join([f0, 'X', f1, 'avg', 'item'])
        bins0 = [-np.inf] + list(get_splits(thresh_counter, f0, n_bins)) + [np.inf]
        bins1 = [-np.inf] + list(get_splits(thresh_counter, f1, n_bins)) + [np.inf]
        matrix['f0'] = pd.cut(matrix[f0], bins0)
        matrix['f1'] = pd.cut(matrix[f1], bins1)
        group_cols = ['f0', 'f1']
        encoded_feature = utils.lag1_expanding_mean_encoding(matrix, date_col='date_block_num', target='item_cnt_month', group_cols=group_cols, global_mean=None)
        matrix.drop(group_cols, axis=1, inplace=True)
        matrix[mean_enc_name] = encoded_feature.astype(np.float32)


def encode_splits_one_hot_label(matrix, thresh_counter, best_splits2, n, n_bins=3, type='label'):
    # n_bins = 3
    # (f0, f1), _ = best_splits2[0]

    for (f0, f1), _ in tqdm(best_splits2[:n]):
        print(f0, f1)
        mean_enc_name = '_'.join([f0, 'X', f1])
        bins0 = [-np.inf] + list(get_splits(thresh_counter, f0, n_bins)) + [np.inf]
        bins1 = [-np.inf] + list(get_splits(thresh_counter, f1, n_bins)) + [np.inf]
        # matrix['f0'] = LabelEncoder().fit_transform(pd.cut(matrix[f0], bins0))
        # matrix['f1'] = LabelEncoder().fit_transform(pd.cut(matrix[f1], bins1))
        matrix['f0'] = pd.cut(matrix[f0], bins0)
        matrix['f1'] = pd.cut(matrix[f1], bins1)
        group_cols = ['f0', 'f1']
        encoded_feature = matrix[group_cols].merge(matrix[group_cols].drop_duplicates().reset_index(drop=True).reset_index(), on=group_cols, how='left')[['index']]
        matrix.drop(group_cols, axis=1, inplace=True)
        if type == 'onehot':
            x = OneHotEncoder(sparse=False).fit_transform(encoded_feature.values).astype(np.int8)
            for i in range(x.shape[1]):
                c = '%s_oh_%s' % (mean_enc_name, i)
                matrix[c] = x[:, i]
        elif type == 'label':
            matrix[mean_enc_name] = encoded_feature['index'].values
        else:
            raise Exception


def add_some_features(matrix, train, items):
    # train, test, items, shops, cats = load_and_clean()
    # matrix = extend_train(train, test, items, shops, cats)

    ################################################
    # first sale features
    ################################################
    ts = time.time()
    matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
    matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')
    time.time() - ts

    ################################################
    # price features
    ################################################
    ts = time.time()
    group_cols = ['date_block_num', 'item_id', 'shop_id']
    group_shorts = ['date', 'item', 'shop']
    enc_col = 'item_price'
    agg_fn = 'mean'
    matrix = mean_enc(matrix, train, group_cols, group_shorts, enc_col, agg_fn)
    np.round(time.time() - ts, 1)

    ts = time.time()
    group_cols = ['date_block_num', 'item_id']
    group_shorts = ['date', 'item']
    enc_col = 'item_price'
    agg_fn = 'mean'
    matrix = mean_enc(matrix, train, group_cols, group_shorts, enc_col, agg_fn)
    np.round(time.time() - ts, 1)

    ts = time.time()
    group_cols = ['item_id']
    group_shorts = ['item']
    enc_col = 'item_price'
    agg_fn = 'mean'
    matrix = mean_enc(matrix, train, group_cols, group_shorts, enc_col, agg_fn)
    np.round(time.time() - ts, 1)

    ts = time.time()
    group_cols = ['item_id', 'shop_id']
    group_shorts = ['item', 'shop']
    enc_col = 'item_price'
    agg_fn = 'mean'
    matrix = mean_enc(matrix, train, group_cols, group_shorts, enc_col, agg_fn)
    np.round(time.time() - ts, 1)

    ts = time.time()
    group_cols = ['item_category_id', 'shop_id']
    group_shorts = ['cat', 'shop']
    enc_col = 'item_price'
    agg_fn = 'mean'
    matrix = mean_enc(matrix, train.merge(items[['item_id', 'item_category_id']], on='item_id'), group_cols, group_shorts, enc_col, agg_fn)
    np.round(time.time() - ts, 1)

    ts = time.time()
    group_cols = ['item_category_id']
    group_shorts = ['cat']
    enc_col = 'item_price'
    agg_fn = 'mean'
    matrix = mean_enc(matrix, train.merge(items[['item_id', 'item_category_id']], on='item_id'), group_cols, group_shorts, enc_col, agg_fn)
    np.round(time.time() - ts, 1)

    # filling na's for item price
    matrix['cat_shop_avg_item_price']      .fillna(matrix['cat_avg_item_price'], inplace=True)
    matrix['item_avg_item_price']          .fillna(matrix['cat_shop_avg_item_price'], inplace=True)
    matrix['date_item_avg_item_price']     .fillna(matrix['item_avg_item_price'], inplace=True)
    matrix['date_item_shop_avg_item_price'].fillna(matrix['date_item_avg_item_price'], inplace=True)
    matrix['item_shop_avg_item_price'].fillna(matrix['item_avg_item_price'], inplace=True)

    ts = time.time()
    lags = [1, 2, 3, 4, 5, 6]
    matrix = utils.lag_feature(matrix, lags, 'date_item_avg_item_price')

    ts = time.time()
    lags = [1, 2, 3, 4, 5, 6]
    matrix = utils.lag_feature(matrix, lags, 'date_item_shop_avg_item_price')

    for i in tqdm(lags):
        matrix['delta_price_lag_' + str(i)] = \
            matrix['date_item_avg_item_price_lag_' + str(i)] / matrix['item_avg_item_price'] - 1.0
        matrix['delta_shop_price_lag_' + str(i)] = \
            matrix['date_item_shop_avg_item_price_lag_' + str(i)] / matrix['item_shop_avg_item_price'] - 1.0
    gc.collect()

    matrix['delta_price_lag'] = matrix[['delta_price_lag_' + str(lag) for lag in lags]].astype('float32').fillna(method='bfill', axis=1).iloc[:, 0]
    # matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)
    matrix['delta_price_lag'].fillna(0, inplace=True)

    matrix['delta_shop_price_lag'] = matrix[['delta_shop_price_lag_' + str(lag) for lag in lags]].astype('float32').fillna(method='bfill', axis=1).iloc[:, 0]
    # matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)
    matrix['delta_shop_price_lag'].fillna(0, inplace=True)

    matrix.rename({'date_item_shop_avg_item_price': 'price'}, inplace=True, axis=1)

    fetures_to_drop = ['date_item_avg_item_price', 'item_avg_item_price', 'item_shop_avg_item_price', 'cat_shop_avg_item_price', 'cat_avg_item_price']
    for i in lags:
        fetures_to_drop += ['date_item_avg_item_price_lag_' + str(i)]
        fetures_to_drop += ['delta_price_lag_' + str(i)]
        fetures_to_drop += ['date_item_shop_avg_item_price_lag_' + str(i)]
        fetures_to_drop += ['delta_shop_price_lag_' + str(i)]
    matrix.drop(fetures_to_drop, axis=1, inplace=True)
    time.time() - ts

    ################################################
    # revenue features
    ################################################
    ts = time.time()
    train['revenue'] = train['item_price'] *  train['item_cnt_day']

    ts = time.time()
    group_cols = ['date_block_num', 'item_id', 'shop_id']
    group_shorts = ['date', 'item', 'shop']
    enc_col = 'revenue'
    agg_fn = 'sum'
    matrix = mean_enc(matrix, train, group_cols, group_shorts, enc_col, agg_fn)
    matrix['date_item_shop_revenue'].fillna(0., inplace=True)
    np.round(time.time() - ts, 1)

    ts = time.time()
    group_cols = ['date_block_num', 'shop_id']
    group_shorts = ['date', 'shop']
    enc_col = 'revenue'
    agg_fn = 'sum'
    matrix = mean_enc(matrix, train, group_cols, group_shorts, enc_col, agg_fn)
    matrix['date_shop_revenue'].fillna(0., inplace=True)
    np.round(time.time() - ts, 1)

    ts = time.time()
    group_cols = ['shop_id']
    group_shorts = ['shop']
    enc_col = 'revenue'
    agg_fn = 'mean'
    _ = train.groupby(['date_block_num', 'shop_id']).agg({'revenue': ['sum']})
    _.columns = ['revenue']
    _.reset_index(inplace=True)
    matrix = mean_enc(matrix, _, group_cols, group_shorts, enc_col, agg_fn)
    matrix['shop_avg_revenue'].fillna(0., inplace=True)
    np.round(time.time() - ts, 1)

    matrix['delta_revenue'] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
    # matrix['delta_revenue'] = matrix['delta_revenue'].astype(np.float16)
    time.time() - ts

    matrix = utils.lag_feature(matrix, [1], 'delta_revenue')
    matrix['delta_revenue_lag_1'].fillna(0.0, inplace=True)
    matrix.drop(['date_item_shop_revenue', 'date_shop_revenue','shop_avg_revenue','delta_revenue'], axis=1, inplace=True)
    time.time() - ts

    ################################################
    # lag features
    ################################################
    ts = time.time()
    matrix = utils.lag_feature(matrix, [1, 2, 3, 6, 12], 'item_cnt_month')
    # matrix = utils.lag_feature(matrix, [1, 2, 3], 'item_cnt_month')
    time.time() - ts
    for c in matrix.columns:
        if '_lag_' in c:
            matrix[c].fillna(0, inplace=True)

    matrix.to_parquet('../data_tmp/tmp1.snappy')

    # matrix = pd.read_parquet('../data_tmp/tmp1.snappy')
    #              tr       val
    # 0  0.800717  1.192927
    # 1  0.730530  1.072190
    # 2  0.705205  1.047885
    # 3  0.700927  1.031627
    # 4  0.695374  1.023219
    # 5  0.695663  1.011614
    # 6  0.689335  1.012463
    # 7  0.683831  1.008743
    # 8  0.684378  1.006106
    # 9  0.681137  1.000676

    matrix['month'] = matrix['date_block_num'] % 12
    err, importances, score_drop = evaluate_features(matrix, t0=11, t1=22, drop_cols=['ID', 'price'])

    matrix.to_parquet('../data_tmp/tmp2.snappy')

    #          tr       val
    # 0  0.742537  1.061269
    # 1  0.706614  1.019063
    # 2  0.691151  1.004994
    # 3  0.683142  1.006200
    # 4  0.685463  1.007951
    # 5  0.680613  1.008837
    # 6  0.673801  0.995782
    # 7  0.672339  0.998700
    # 8  0.671063  0.998168
    # 9  0.670680  0.996113

    ################################################
    # mean encode categories
    ################################################
    ts = time.time()
    lags = [1, 2, 3, 6, 12] #
    for col, cname in tqdm({'item_id': 'item', 'shop_id': 'shop' , 'item_category_id': 'cat'}.items()): #
        mean_enc_name = '_'.join(['date', cname, 'avg', 'item'])
        matrix[mean_enc_name], global_mean = utils.date_expanding_mean_encoding(matrix, date_col='date_block_num', target='item_cnt_month', group_cols=[col], global_mean=0)
        matrix = utils.lag_feature(matrix, lags, mean_enc_name, id_cols=['date_block_num', col])
        for c in ['%s_lag_%s' % (mean_enc_name, i) for i in lags]:
            matrix[c].fillna(0, inplace=True)
            matrix[c] = matrix[c].astype(np.float32)
        matrix.drop(mean_enc_name, axis=1, inplace=True)
    time.time() - ts

    for c in matrix.columns:
        if matrix[c].dtype == 'float64':
            matrix[c] = matrix[c].astype(np.float32)
    matrix.to_parquet('../data_tmp/tmp3.snappy')
    err, importances, score_drop = evaluate_features(matrix, t0=11, t1=22, drop_cols=['ID', 'price'])

    #          tr       val
    # 0  0.729307  1.047609
    # 1  0.692656  1.043487
    # 2  0.671874  1.028236
    # 3  0.660618  1.008712
    # 4  0.654776  0.992076
    # 5  0.649632  0.983072
    # 6  0.646305  0.978352
    # 7  0.640759  0.978253
    # 8  0.638548  0.977217
    # 9  0.637210  0.973175

    drop_cols = ['date_cat_avg_item_lag_12']
    matrix.drop(drop_cols, axis=1, inplace=True)
    err, importances, score_drop = evaluate_features(matrix, t0=11, t1=22, drop_cols=['ID', 'price'])
    matrix.to_parquet('../data_tmp/tmp4.snappy')
    #          tr       val
    # 0  0.714360  1.117559
    # 1  0.686641  1.044635
    # 2  0.661932  0.984645
    # 3  0.657309  0.977247
    # 4  0.655725  0.985918
    # 5  0.649996  0.976907
    # 6  0.647806  0.969153
    # 7  0.646006  0.967647
    # 8  0.642500  0.964019
    # 9  0.640825  0.965701

    ts = time.time()
    lags = [1, 2, 3, 6, 12]
    for cols, cnames in {('item_id', 'shop_id'): ('item', 'shop'),
                         ('item_category_id', 'shop_id'): ('cat', 'shop')}.items():
        mean_enc_name = '_'.join(['date'] + list(cnames) + ['avg', 'item'])
        matrix[mean_enc_name], global_mean = utils.date_expanding_mean_encoding(matrix, date_col='date_block_num', target='item_cnt_month', group_cols=list(cols), global_mean=0)
        matrix = utils.lag_feature(matrix, lags, mean_enc_name, id_cols=['date_block_num'] + list(cols))
        for c in ['%s_lag_%s' % (mean_enc_name, i) for i in lags]:
            matrix[c].fillna(0, inplace=True)
            matrix[c] = matrix[c].astype(np.float32)
        matrix.drop(mean_enc_name, axis=1, inplace=True)
    time.time() - ts
    matrix.to_parquet('../data_tmp/tmp5.snappy')
    #          tr       val
    # 0  0.725399  1.048693
    # 1  0.680849  1.005390
    # 2  0.667225  0.993659
    # 3  0.664457  0.999274
    # 4  0.650158  0.984977
    # 5  0.643214  0.979223
    # 6  0.636908  0.974081
    # 7  0.634623  0.961683
    # 8  0.634248  0.954822
    # 9  0.632593  0.955750

    matrix = pd.read_parquet('../data_tmp/tmp5.snappy')
    err, importances, score_drop, thresh_counter, best_splits2 = evaluate_features(matrix, t0=11, t1=22, drop_cols=['ID', 'price'])




    #          tr       val
    # 0  0.687675  1.201444
    # 1  0.655024  0.991819
    # 2  0.648814  0.976386
    # 3  0.646841  0.956301
    # 4  0.644135  0.945936
    # 5  0.638622  0.940919
    # 6  0.637341  0.936372
    # 7  0.633063  0.932086
    # 8  0.631374  0.935182
    # 9  0.628252  0.931992

    matrix = pd.read_parquet('../data_tmp/tmp5.snappy')
    err, importances, score_drop, thresh_counter, best_splits2 = evaluate_features(matrix, t0=11, t1=22, drop_cols=['ID', 'price'])
    expanding_mean_encode_splits(matrix, thresh_counter, best_splits2, n=len(best_splits2), n_bins=10)
    matrix.to_parquet('../data_tmp/tmp6.snappy')
    err, importances, score_drop, thresh_counter, best_splits2 = evaluate_features(matrix, t0=11, t1=22, drop_cols=['ID', 'price'])

    #          tr       val
    # 0  0.703115  1.082618
    # 1  0.674777  0.989672
    # 2  0.652970  0.938250
    # 3  0.650986  0.934846
    # 4  0.645391  0.940019
    # 5  0.643873  0.940444
    # 6  0.640197  0.942430
    # 7  0.635598  0.934855
    # 8  0.636822  0.935958
    # 9  0.635502  0.934361

    #### knn features
    knn_map = {'date_item': ['item_0', 'item_1', 'item_2', 'item_3', 'item_4'],
               'date_shop': ['shop_0', 'shop_1', 'shop_2'],
               'date_item_shop': ['item_0', 'item_1', 'item_2', 'item_3', 'item_4', 'shop_0', 'shop_1', 'shop_2']}
    for knn_id, knn_cols in knn_map.items():
        n_jobs = 8
        k_list = [3, 10, 20]
        metric = 'minkowski'
        target = 'item_cnt_month'
        knn_feats = []
        dates = sorted(matrix.date_block_num.unique())
        for t_prev, t in tqdm(zip(dates, dates[1:])):
            NNF = knn_features.NearestNeighborsRegFeatsByGroups(n_jobs=n_jobs, k_list=k_list, metric=metric)
            train_index = matrix.loc[matrix.date_block_num == t_prev].index
            test_index  = matrix.loc[matrix.date_block_num == t     ].index
            X = matrix.loc[train_index, knn_cols].values
            Y = matrix.loc[train_index, target].values
            X_test = matrix.loc[test_index, knn_cols].values
            # Y_reg = matrix.loc[matrix.date_block_num == t, target].values
            NNF.fit(X, Y)
            test_knn_feats, _ = NNF.predict(X_test)
            knn_feats.append(pd.DataFrame(test_knn_feats, index=test_index, columns=NNF.get_feature_names(knn_id, ['item_cnt'])))
        knn_feats = pd.concat(knn_feats, axis=0)
        knn_feats.columns = ['%s_lag_1' % c for c in knn_feats.columns]
        knn_feats.to_parquet('../data_processed/knn/%s.snappy' % (knn_id))

    knn_feats = []
    for knn_id, knn_cols in knn_map.items():
        _ = pd.read_parquet('../data_processed/knn/%s.snappy' % (knn_id))
        _.drop([c for c in _.columns if '_min_' in c], axis=1, inplace=True)
        knn_feats.append(_)
    knn_id = 'date_item_shopid'
    _ = pd.read_parquet('../data_processed/knn/%s.snappy' % (knn_id))
    _.drop([c for c in _.columns if '_min_' in c], axis=1, inplace=True)
    knn_feats.append(_)

    knn_feats = pd.concat(knn_feats, axis=1)
    matrix = matrix.merge(knn_feats, left_index=True, right_index=True, how='left')
    drop_cols = []
    err, importances, score_drop, thresh_counter, best_splits2 = evaluate_features(matrix, t0=11, t1=22, drop_cols=['ID', 'price'] + drop_cols)
    #          tr       val
    # 0  0.741812  1.158412
    # 1  0.702651  1.052534
    # 2  0.683194  1.032136
    # 3  0.669331  1.006149
    # 4  0.663625  1.005367
    # 5  0.657688  0.995047
    # 6  0.658686  0.986821
    # 7  0.650594  0.980932
    # 8  0.650515  0.980158
    # 9  0.647933  0.970870
    #          tr       val with item_shopid features
    # 0  0.740904  1.067546
    # 1  0.670665  1.025822
    # 2  0.652943  0.987056
    # 3  0.653788  0.984585
    # 4  0.653717  0.985723
    # 5  0.655951  0.988414
    # 6  0.654229  0.982954
    # 7  0.653869  0.982922
    # 8  0.654357  0.981486
    # 9  0.650522  0.977081

    drop_cols += ['date_item_knn_10_mean_item_cnt_lag_1', 'date_shop_knn_20_mean_item_cnt_lag_1', 'date_shop_knn_3_std_item_cnt_lag_1', 'date_shop_knn_20_std_item_cnt_lag_1', 'date_shop_knn_3_max_item_cnt_lag_1', 'date_shop_knn_20_max_item_cnt_lag_1', 'date_shop_knn_10_max_item_cnt_lag_1', ]
    drop_cols += ['date_item_knn_3_mean_item_cnt_lag_1', 'date_shop_knn_20_std_item_cnt_lag_1', 'date_shop_knn_10_std_item_cnt_lag_1', 'date_item_knn_20_mean_item_cnt_lag_1', 'date_shop_knn_10_mean_item_cnt_lag_1', 'date_shop_knn_20_max_item_cnt_lag_1', 'date_shop_knn_10_max_item_cnt_lag_1', 'date_shop_knn_3_max_item_cnt_lag_1', ]
    err, importances, score_drop, thresh_counter, best_splits2 = evaluate_features(matrix, t0=11, t1=22, drop_cols=['ID', 'price'] + drop_cols)
    # err, importances, score_drop, thresh_counter, best_splits2 = evaluate_features(matrix, t0=11, t1=33, drop_cols=['ID', 'price'] + drop_cols)
    expanding_mean_encode_splits(matrix, thresh_counter, best_splits2, n=len(best_splits2), n_bins=10)
    err, importances, score_drop, thresh_counter, best_splits2 = evaluate_features(matrix, t0=11, t1=22, drop_cols=['ID', 'price'] + drop_cols)
    drop_cols += ['date_item_shop_knn_10_std_item_cnt_lag_1', 'date_item_knn_20_max_item_cnt_lag_1', 'date_item_shop_knn_10_max_item_cnt_lag_1', 'date_item_shopid_knn_10_std_item_cnt_lag_1', 'date_item_shopid_knn_20_max_item_cnt_lag_1', ]
    err, importances, score_drop, thresh_counter, best_splits2 = evaluate_features(matrix, t0=11, t1=22, drop_cols=['ID', 'price'] + drop_cols)
    drop_cols += ['date_item_shop_knn_20_max_item_cnt_lag_1', 'date_item_shop_knn_20_std_item_cnt_lag_1', 'date_shop_knn_3_mean_item_cnt_lag_1', 'date_item_shop_knn_3_std_item_cnt_lag_1', ]
    err, importances, score_drop, thresh_counter, best_splits2 = evaluate_features(matrix, t0=11, t1=22, drop_cols=['ID', 'price'] + drop_cols)
    #          tr       val t == 22 with item_shopid
    # 0  0.731989  1.052323
    # 1  0.696501  1.024941
    # 2  0.686642  1.012767
    # 3  0.672317  0.984372
    # 4  0.666984  0.982737
    # 5  0.660103  0.969800
    # 6  0.655817  0.950437
    # 7  0.654492  0.951607
    # 8  0.651137  0.950105
    # 9  0.650083  0.947628

    #          tr       val extending interactions
    # 0  0.709961  1.078231
    # 1  0.663689  0.972049
    # 2  0.652830  0.961224
    # 3  0.644100  0.953036
    # 4  0.639490  0.952125
    # 5  0.639284  0.946453
    # 6  0.634258  0.938786
    # 7  0.634575  0.939513
    # 8  0.635608  0.939774
    # 9  0.636122  0.942695

    #          tr       val
    # 0  0.725707  1.105960
    # 1  0.669734  1.001679
    # 2  0.654948  0.962415
    # 3  0.651191  0.963100
    # 4  0.647619  0.959507
    # 5  0.646172  0.962321
    # 6  0.646483  0.968020
    # 7  0.647110  0.970820
    # 8  0.649260  0.972166
    # 9  0.647091  0.966589


    matrix.to_parquet('../data_tmp/2020-07-08.snappy')
    data = matrix[matrix.date_block_num > 11]
    del matrix; gc.collect()
    drop_cols = ['ID', 'price']
    drop_cols += ['date_item_knn_10_mean_item_cnt_lag_1', 'date_shop_knn_20_mean_item_cnt_lag_1', 'date_shop_knn_3_std_item_cnt_lag_1', 'date_shop_knn_20_std_item_cnt_lag_1', 'date_shop_knn_3_max_item_cnt_lag_1', 'date_shop_knn_20_max_item_cnt_lag_1', 'date_shop_knn_10_max_item_cnt_lag_1', ]
    data.drop(drop_cols, axis=1, inplace=True)
    data.info()
    data.fillna(0, inplace=True)

    X_train = data[data.date_block_num < 33].drop(['item_cnt_month', 'date_block_num'], axis=1)
    Y_train = data[data.date_block_num < 33]['item_cnt_month']
    X_valid = data[data.date_block_num == 33].drop(['item_cnt_month', 'date_block_num'], axis=1)
    Y_valid = data[data.date_block_num == 33]['item_cnt_month']
    X_test = data[data.date_block_num == 34].drop(['item_cnt_month', 'date_block_num'], axis=1)

    del data
    gc.collect()

    # [62]	validation_0-rmse:0.79811	validation_1-rmse:0.89230

    ts = time.time()
    model = XGBRegressor(
        max_depth=8,
        n_estimators=30,
        min_child_weight=300,
        colsample_bytree=0.8,
        subsample=0.8,
        eta=0.2,
        seed=42,
        nthread=7)

    model.fit(
        X_train,
        Y_train,
        eval_metric="rmse",
        eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
        verbose=True,
        early_stopping_rounds = 10)
    time.time() - ts

    best_iteration = model.get_booster().best_ntree_limit

    terr = mean_squared_error(Y_train.values.astype('float32'), model.predict(X_train, ntree_limit=best_iteration).clip(0, 20))
    verr = mean_squared_error(Y_valid.values.astype('float32'), model.predict(X_valid, ntree_limit=best_iteration).clip(0, 20))
    print('train-rmse=%.4f valid-rmse=%.4f' % (terr, verr))

    Y_pred = model.predict(X_valid, ntree_limit=best_iteration).clip(0, 20)
    Y_test = model.predict(X_test, ntree_limit=best_iteration).clip(0, 20)

    # from collections import Counter
    # from sklearn.metrics import confusion_matrix
    # Y_pred = model.predict(X_valid).clip(0, 20)
    # sorted(Counter(Y_pred.round(0)).items())
    # sorted(Counter(Y_valid).items())
    # cm = confusion_matrix(Y_valid.values.astype('int8'), Y_pred.round(0).astype('int8'))
    # pd.DataFrame(cm).to_csv('../data/errors.xgb.best.csv')

    # Y_test = model.predict(X_test).clip(0, 20)
    test = pd.read_csv('../data/test.csv')
    submission = pd.DataFrame({
        "ID": test.index,
        "item_cnt_month": Y_test
    })
    submission.to_csv('../submission/xgb_submission_2020-07-09_01.csv', index=False)
    pd.Series(Y_pred).to_csv('../submission/xgb_submission_2020-07-09_01_valid.csv', index=False)

    #         tr       val
    # 0  0.719497  1.063976
    # 1  0.680237  1.000554
    # 2  0.670583  0.977827
    # 3  0.658292  0.955831
    # 4  0.656116  0.954723
    # 5  0.655300  0.954012
    # 6  0.647821  0.939479
    # 7  0.644999  0.937383
    # 8  0.640744  0.937389
    # 9  0.640423  0.942888
    #          tr       val t = 33
    # 0  0.768883  0.891351
    # 1  0.712418  0.880349
    # 2  0.705914  0.856757
    # 3  0.700799  0.841675
    # 4  0.698489  0.837418
    # 5  0.695807  0.830791
    # 6  0.694431  0.819075
    # 7  0.694216  0.817466
    # 8  0.696895  0.817759
    # 9  0.693275  0.815480

    drop_cols += ['date_shop_knn_3_mean_item_cnt_lag_1',  'date_shop_knn_3_max_item_cnt_lag_1',  'date_shop_knn_3_std_item_cnt_lag_1',  'date_shop_knn_10_mean_item_cnt_lag_1',  'date_shop_knn_10_max_item_cnt_lag_1',  'date_shop_knn_10_std_item_cnt_lag_1',  'date_shop_knn_20_mean_item_cnt_lag_1',  'date_shop_knn_20_max_item_cnt_lag_1',  'date_shop_knn_20_std_item_cnt_lag_1']
    err, importances, score_drop, thresh_counter, best_splits2 = evaluate_features(matrix, t0=11, t1=22, drop_cols=['ID', 'price'] + drop_cols)
    score_drop[score_drop.mean_drop <= 0.0001]

    #          tr       val
    # 0  0.739528  1.207945
    # 1  0.679197  1.011331
    # 2  0.673736  0.990914
    # 3  0.664418  0.975682
    # 4  0.658474  0.971417
    # 5  0.655101  0.979347
    # 6  0.652609  0.979451
    # 7  0.649737  0.972246
    # 8  0.646250  0.968678
    # 9  0.643864  0.967476


    ii = np.random.choice(matrix.index, 10000, False)
    xx = matrix.loc[ii, ['date_item_shop_knn_10_mean_item_cnt_lag_1']].values
    yy = matrix.loc[ii, score_drop.tail(30).index]
    cc = yy.corr()['date_item_shop_knn_10_mean_item_cnt_lag_1']
    cc.sort_values()

    #############################
    # alternative groupby explicitly
    #############################
    knn_cols = ['item_0', 'item_1', 'item_2', 'item_3', 'item_4']
    group_col = 'shop_id'
    date_col = 'date_block_num'
    knn_id = 'date_item_shopid'
    n_jobs = 1
    n_jobs_exterior = 8
    k_list = [3, 10, 20]
    metric = 'minkowski'
    target = 'item_cnt_month'
    knn_feats = []
    dates = sorted(matrix.date_block_num.unique())
    list_fn_args = []
    for t_prev, t in tqdm(zip(dates, dates[1:])):
        _train = matrix.loc[(matrix[date_col] == t_prev)].index
        _test  = matrix.loc[matrix[date_col] == t     ].index
        group_values = np.intersect1d(matrix.loc[_train, group_col].values,
                                      matrix.loc[_test,  group_col].values)
        for gv in group_values:
            train_index = matrix.loc[(matrix[date_col] == t_prev) & (matrix[group_col] == gv)].index
            test_index  = matrix.loc[(matrix[date_col] == t     ) & (matrix[group_col] == gv)].index
            X = matrix.loc[train_index, knn_cols].values
            Y = matrix.loc[train_index, target].values
            # X_test = matrix.loc[test_index, knn_cols].values
            NNF = knn_features.NearestNeighborsRegFeatsByGroups(n_jobs=n_jobs, k_list=k_list, metric=metric)
            NNF.fit(X, Y)
            list_fn_args.append((NNF.predict, test_index))

    knn_feat = {}
    def on_ret(retval):
        obj, obj_id = retval
        knn_feat[obj_id] = obj
        print(obj_id)
        
    p = Pool(n_jobs_exterior)
    for i, (pred_fn, test_index) in tqdm(enumerate(list_fn_args)):
        X_test = matrix.loc[test_index, knn_cols]
        p.apply_async(pred_fn, args=(X_test, i), callback=on_ret)

    p.close(); p.join()
    big_index = np.hstack([i for _, i in list_fn_args])
    big_cols = NNF.get_feature_names(knn_id, ['item_cnt'])
    knn_feats = pd.DataFrame(np.vstack(knn_feat[i] for i in range(len(list_fn_args))), index=big_index, columns=big_cols)
    knn_feats.columns = ['%s_lag_1' % c for c in knn_feats.columns]
    knn_feats.to_parquet('../data_processed/knn/%s.snappy' % (knn_id))





    matrix = matrix.merge(knn_feats, left_index=True, right_index=True, how='left')
    matrix.to_parquet('../data_tmp/tmp7.snappy')
    # those have 0 std
    drop_cols = ['date_item_knn_20_min_item_cnt_lag_1', 'date_item_knn_10_min_item_cnt_lag_1', 'date_item_knn_3_min_item_cnt_lag_1', ]
    err, importances, score_drop, thresh_counter, best_splits2 = evaluate_features(matrix, t0=11, t1=22, drop_cols=['ID', 'price']+drop_cols)
    score_drop.head(30)

    #          tr       val
    # 0  0.713309  1.112787
    # 1  0.667688  1.045669
    # 2  0.645745  0.998628
    # 3  0.634384  0.978032
    # 4  0.626647  0.961763
    # 5  0.625309  0.950575
    # 6  0.624172  0.944746
    # 7  0.620501  0.944957
    # 8  0.619098  0.942257
    # 9  0.618158  0.941365

    ################################################
    # mean encode categories
    ################################################
    
    matrix = pd.read_parquet('../data_tmp/tmp6.snappy')
    # new_features = []
    # for agg_cols, agg_names, lags in tqdm(
    #         [
    #          # (['shop_0'],  ['shop_0'], [1, 2, 3, 6, 12]),
    #          # (['shop_1'],  ['shop_1'], [1, 2, 3, 6, 12]),
    #          # (['shop_2'],  ['shop_2'], [1, 2, 3, 6, 12]),
    #          # (['item_0'],  ['item_0'], [1, 2, 3, 6, 12]),
    #          # (['item_1'],  ['item_1'], [1, 2, 3, 6, 12]),
    #          # (['item_2'],  ['item_2'], [1, 2, 3, 6, 12]),
    #          # (['item_3'],  ['item_3'], [1, 2, 3, 6, 12]),
    #          # (['item_4'],  ['item_4'], [1, 2, 3, 6, 12]),
    #          ]):
    #     ts = time.time()
    #     mean_enc_name = '_'.join(['date'] + agg_names + ['avg', 'item'])
    #     new_features += ['%s_lag_%s' % (mean_enc_name, i) for i in lags]
    #
    #     matrix[mean_enc_name], global_mean = utils.date_expanding_mean_encoding(matrix, date_col='date_block_num', target='item_cnt_month', group_cols=agg_cols, global_mean=0)
    #     matrix = utils.lag_feature(matrix, lags, mean_enc_name, id_cols=['date_block_num'] + agg_cols)
    #     for c in ['%s_lag_%s' % (mean_enc_name, i) for i in lags]:
    #         matrix[c].fillna(0, inplace=True)
    #     matrix.drop(mean_enc_name, axis=1, inplace=True)
    #
    # for c in matrix.columns:
    #     if matrix[c].dtype == 'float64':
    #         matrix[c] = matrix[c].astype(np.float32)
    # matrix.to_parquet('../data_tmp/tmp5.snappy')
    err, importances, score_drop = evaluate_features(matrix, t0=11, t1=22, drop_cols=['ID', 'price', 'item_id'])
    err

    #          tr       val
    # 0  0.709354  0.992643
    # 1  0.667754  0.943571
    # 2  0.655671  0.950085
    # 3  0.647771  0.954242
    # 4  0.639475  0.931024
    # 5  0.635746  0.930428
    # 6  0.635117  0.932989
    # 7  0.632023  0.931826
    # 8  0.631274  0.930837
    # 9  0.631330  0.922814

    for agg_cols, agg_names, lags in tqdm(
            [
             # (['shop_0'], ['shop_0'], [1]),
             # (['shop_1'], ['shop_1'], [1]),
             # (['shop_2'], ['shop_2'], [1]),
             (['cat_0'], ['cat_0'], [1]),
             (['cat_1'], ['cat_1'], [1]),
             (['cat_2'], ['cat_2'], [1]),
             # (['item_0'], ['item_0'], [1]),
             # (['item_1'], ['item_1'], [1]),
             # (['item_2'], ['item_2'], [1]),
             # (['item_3'], ['item_3'], [1]),
             # (['item_4'], ['item_4'], [1]),
             ]):
        ts = time.time()
        mean_enc_name = '_'.join(agg_names + ['avg', 'item', 'lag_1'])
        matrix[mean_enc_name] = utils.lag1_expanding_mean_encoding(matrix, date_col='date_block_num', target='item_cnt_month', group_cols=agg_cols)
        print(time.time() - ts)

    for c in matrix.columns:
        if matrix[c].dtype == 'float16' or matrix[c].dtype == 'float64':
            matrix[c] = matrix[c].astype(np.float32)

    matrix.to_parquet('../data_tmp/tmp5.snappy')
    err, importances, score_drop = evaluate_features(matrix, t0=11, t1=22, drop_cols=['ID', 'price', 'item_id'])
    err


    #          tr       val
    # 0  0.720321  1.100728
    # 1  0.660302  0.996750
    # 2  0.641736  0.954086
    # 3  0.636053  0.945544
    # 4  0.639787  0.945419
    # 5  0.634334  0.939812
    # 6  0.637363  0.942186
    # 7  0.639440  0.949953
    # 8  0.640561  0.952878
    # 9  0.637232  0.948519

    matrix.to_parquet('../data_processed/add_some_features.snappy', compression='snappy')

    return matrix

    matrix = pd.read_parquet('../data_processed/add_some_features.snappy')

    data = matrix[matrix.date_block_num > 11]
    del matrix; gc.collect()
    drop_cols = ['ID']
    data.drop(drop_cols, axis=1, inplace=True)
    data.info()
    data.fillna(0, inplace=True)

    X_train = data[data.date_block_num < 33].drop(['item_cnt_month', 'date_block_num', 'item_id'], axis=1)
    Y_train = data[data.date_block_num < 33]['item_cnt_month']
    X_valid = data[data.date_block_num == 33].drop(['item_cnt_month', 'date_block_num', 'item_id'], axis=1)
    Y_valid = data[data.date_block_num == 33]['item_cnt_month']
    X_test = data[data.date_block_num == 34].drop(['item_cnt_month', 'date_block_num', 'item_id'], axis=1)

    del data
    gc.collect()

    # [62]	validation_0-rmse:0.79811	validation_1-rmse:0.89230

    ts = time.time()
    model = XGBRegressor(
        max_depth=8,
        n_estimators=100,
        min_child_weight=300,
        colsample_bytree=0.8,
        subsample=0.8,
        eta=0.02,
        seed=42,
        nthread=7)

    model.fit(
        X_train,
        Y_train,
        eval_metric="rmse",
        eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
        verbose=True,
        early_stopping_rounds = 10)
    time.time() - ts


    terr = mean_squared_error(Y_train.values.astype('float32'), model.predict(X_train).clip(0, 20))
    verr = mean_squared_error(Y_valid.values.astype('float32'), model.predict(X_valid).clip(0, 20))
    print('train-rmse=%.4f valid-rmse=%.4f' % (terr, verr))

    Y_pred = model.predict(X_valid).clip(0, 20)
    Y_test = model.predict(X_test).clip(0, 20)

    # from collections import Counter
    # from sklearn.metrics import confusion_matrix
    # Y_pred = model.predict(X_valid).clip(0, 20)
    # sorted(Counter(Y_pred.round(0)).items())
    # sorted(Counter(Y_valid).items())
    # cm = confusion_matrix(Y_valid.values.astype('int8'), Y_pred.round(0).astype('int8'))
    # pd.DataFrame(cm).to_csv('../data/errors.xgb.best.csv')

    # Y_test = model.predict(X_test).clip(0, 20)
    test = pd.read_csv('../data/test.csv')
    submission = pd.DataFrame({
        "ID": test.index,
        "item_cnt_month": Y_test
    })
    submission.to_csv('../submission/xgb_submission_2020-07-07_01.csv', index=False)
    pd.Series(Y_pred).to_csv('../submission/xgb_submission_2020-07-07_01_valid.csv', index=False)









def generate_item_item_shop_features(matrix):
    matrix = pd.read_parquet('../data_processed/add_some_features.snappy')
    matrix = matrix[['date_block_num',
                     'item_cnt_month', 'item_first_sale', 'date_item_shop_avg_item_price', 'date_item_shop_revenue',
                     'item_0', 'item_1', 'item_2', 'item_3', 'item_4', 'shop_0', 'shop_1', 'shop_2']]
    gc.collect(); gc.collect()
    lags = [1, 2, 3, 6, 12]
    win_sizes = [1]
    k_list = [3, 10, 50, 100]
    metric = 'minkowski'
    n_jobs = 8
    target_cols = ['item_cnt_month', 'item_first_sale', 'date_item_shop_avg_item_price', 'date_item_shop_revenue']
    target_cols_short = ['item', 'fsale', 'price', 'rev']
    knn_cols_map = {
        'item_shop': ['item_0', 'item_1', 'item_2', 'item_3', 'item_4', 'shop_0', 'shop_1', 'shop_2'],
        'item': ['item_0', 'item_1', 'item_2', 'item_3', 'item_4'],
        'shop': ['shop_0', 'shop_1', 'shop_2']
    }
    for lag in tqdm(lags):
        win_size_list = win_sizes + [12] if lag == 1 else win_sizes
        for win_size in win_size_list:
            date_col = 'date_block_num'
            for knn_prefix, knn_cols in knn_cols_map.items():
                t0 = max(lags) - 1 + max(win_sizes)
                res = knn_features.rolling_knn(matrix, t0, date_col=date_col,
                                knn_prefix=knn_prefix,
                                knn_cols=knn_cols, # columns to find nearest neighbors for
                                target_cols_short=target_cols_short, # short names for generating features names from target_cols
                                target_cols=target_cols, # columns to calculate knn averages
                                win_size=win_size, lag=lag,
                                n_jobs=n_jobs, k_list=k_list, metric=metric, eps=1e-6, seed=seed)

    for lag in lags:
        win_size_list = win_sizes + [12] if lag == 1 else win_sizes
        for win_size in win_size_list:
            for knn_prefix in knn_cols_map:
                _ = []
                bn = 'knn_%s_win%s_lag%s' % (knn_prefix, win_size, lag)
                for f in tqdm(sorted(iglob('../data_processed/knn_temp/%s_t_*.snappy' % bn))):
                    _.append(pd.read_parquet(f))
                _ = pd.concat(_, axis=0)
                _.columns = ['%s_w%s_lag_%s' % (c, win_size, lag) for c in _.columns]
                _.to_parquet('../data_processed/%s.snappy' % bn, compression='snappy')

def lag_added_features():
    matrix = pd.read_parquet('../data_processed/add_some_features.snappy')
    for col in tqdm(['date_item_shop_avg_item_price', 'date_item_avg_item_price', 'item_avg_item_price', 'cat_shop_avg_item_price', 'cat_avg_item_price', 'date_item_shop_revenue', ]):
        matrix = utils.lag_feature(matrix, [1], col)
        matrix.drop(col, axis=1, inplace=True)
    matrix.to_parquet('../data_processed/add_some_features1.snappy', compression='snappy')
    # matrix = pd.read_parquet('../data_processed/add_some_features1.snappy')

# train, test, items, shops, cats = load_and_clean()
# matrix = extend_train(train, test, items, shops, cats)
# matrix = add_some_features(matrix, train, items)
matrix = pd.read_parquet('../data_processed/add_some_features1.snappy')

knn_cols_map = {
    'item_shop': ['item_0', 'item_1', 'item_2', 'item_3', 'item_4', 'shop_0', 'shop_1', 'shop_2'],
    'item': ['item_0', 'item_1', 'item_2', 'item_3', 'item_4'],
    'shop': ['shop_0', 'shop_1', 'shop_2']
}
lags = [1, 2, 3, 6, 12]
win_sizes = [1]
target_cols_short = ['item', 'fsale', 'price', 'rev']
k_list = [3, 10, 50, 100]

def generate_trend_features(knn_cols_map, lags, win_sizes, k_list, target_cols_short):
    for knn_prefix in knn_cols_map:
        win12 = pd.read_parquet('../data_processed/knn/knn_%s_win12_lag1.snappy' % knn_prefix)
        for lag in tqdm(lags):
            win_size_list = win_sizes
            for win_size in win_size_list:
                bn = 'knn_%s_win%s_lag%s' % (knn_prefix, win_size, lag)
                bn_out = 'knn_%s_win%s_delta%s' % (knn_prefix, win_size, lag)
                # matrix = pd.concat([matrix, pd.read_parquet('../data_processed/%s.snappy' % bn)], axis=1)
                # _ = pd.read_parquet('../data_processed/%s.snappy' % bn)
                _ = pd.read_parquet('../data_processed/knn/%s.snappy' % bn)
                res = []
                for knn in k_list:
                    for fn in ['mean', 'median']:
                        for var in target_cols_short:
                            col_name = '%s_knn_%s_%s_%s_delta_%s' % (knn_prefix, knn, fn, var, lag)
                            c2 = '%s_knn_%s_%s_%s_w%s_lag_%s' % (knn_prefix, knn, fn, var, 12, 1)
                            c1 = '%s_knn_%s_%s_%s_w%s_lag_%s' % (knn_prefix, knn, fn, var,  1, lag)
                            ser = np.log1p(np.clip(_[c1] / (win12[c2] + 1e-6), -1 + 1e-6, None))
                            ser.name = col_name
                            res.append(ser)
                pd.concat(res, axis=1).to_parquet('../data_processed/knn/%s.snappy' % bn_out, compression='snappy')
                gc.collect()

# generate_trend_features(knn_cols_map, lags, win_sizes, k_list, target_cols_short)

def get_all_feature_names():
    paths = ['../data_processed/add_some_features1.snappy'] + \
            list(iglob('../data_processed/knn/knn*.snappy'))

    all_cols = []
    file_cols = {}
    for fn in tqdm(paths):
        file_cols[fn] = pd.read_parquet(fn).columns.tolist()
        all_cols += file_cols[fn]

    fn = '../data_processed/feature_selection/all_features.pkl'
    pickle.dump(all_cols, open(fn, 'wb'))
    pickle.dump(file_cols, open('../data_processed/feature_selection/file_cols.pkl', 'wb'))


def bootstrap_eliminate(min_eliminate=100):
    all_features_file = '../data_processed/feature_selection/all_features.pkl'
    file_cols = pickle.load(open('../data_processed/feature_selection/file_cols.pkl', 'rb'))
    all_features = pickle.load(open(all_features_file, 'rb'))

    for i in range(20):
        drop_paths = list(iglob('../data_processed/feature_selection/drop_*.pkl'))
        drop_features = []
        for path in drop_paths:
            drop_features += pickle.load(open(path, 'rb'))

        n_features = 300
        target = 'item_cnt_month'
        keep_features = ['date_block_num', 'item_id', 'shop_id', 'item_category_id'] + \
                        ['item_0', 'item_1', 'item_2', 'item_3', 'item_4'] + \
                        ['shop_0', 'shop_1', 'shop_2'] + \
                        ['cat_0', 'cat_1', 'cat_2']
        _ = list(set(all_features) - set(keep_features) - set(drop_features))

        try_features = np.random.choice(_, min(len(_), n_features - len(keep_features)), False).tolist() + keep_features + [target]
        len(try_features)

        paths = ['../data_processed/add_some_features1.snappy'] + \
                list(iglob('../data_processed/knn/knn*.snappy'))

        matrix = []
        for p in tqdm(paths):
            _ = pd.read_parquet(p, columns=list(set(file_cols[p]) & set(try_features)))
            if _.columns.shape[0] > 0:
                matrix.append(_)
        matrix = pd.concat(matrix, axis=1)

        utils.df_info(matrix)
        drop_cols, clusters = utils.find_duplicates_features(matrix,
                                                             id_cols=['item_cnt_month', 'date_block_num', 'shop_id', 'item_id', 'item_category_id'],
                                                             date_col='date_block_num',
                                                             date_thresh=11,
                                                             corr_thresh=0.97,
                                                             corr_sample_size=10000,
                                                             n_jobs=8,)

        n = len(list(iglob('../data_processed/feature_selection/drop_*.pkl')))
        pickle.dump(list(drop_cols), open('../data_processed/feature_selection/drop_%s.pkl' % n, 'wb'))
        matrix.drop(drop_cols, axis=1, inplace=True)
        matrix.to_parquet('../data_tmp/tmp.snappy', compression='snappy')

        matrix = pd.read_parquet('../data_tmp/tmp.snappy')

        train_index = matrix[(matrix.date_block_num < 33) & (matrix.date_block_num > 11)].index
        valid_index = matrix[matrix.date_block_num == 33].index
        del matrix; gc.collect()
        drop_cols, elimination_path, model, random_string, score_drop = utils.recursive_features_eliminate(path_to_df='../data_tmp/tmp.snappy',
                                         target='item_cnt_month',
                                         max_iter=1,
                                         always_drop=['ID'],
                                         keep_cols=['date_block_num', 'item_id', 'shop_id', 'item_category_id', 'item_0', 'item_1', 'item_2', 'item_3', 'item_4', 'shop_0', 'shop_1', 'shop_2', 'cat_0', 'cat_1', 'cat_2'],
                                         tstat_thresh=0, # tstat = (mean - base_model) / std threshold to drop features from the model
                                         max_drop_ratio=0.2, # fraction of features to drop from the non-eliminated features,
                                         min_drop=min_eliminate,
                                         train_index=train_index,
                                         valid_index=valid_index,
                                         reload_data=False, # if memory is bottleneck deleting and re-reading data on each iteration
                                         model_type='rf', # model to train and use for elimination
                                         clip_down=0,
                                         clip_up=20,
                                         max_xgb_features=200, # memory issues if >200 columns
                                         num_permutations=5, # number of permutation for feature to get stats on baseline score drop
                                         n_estimators=10,
                                         max_depth=8,
                                         max_samples=0.8,
                                         seed=123,
                                         n_jobs=8,
                                         learning_rate=0.2,
                                         )
        n = len(list(iglob('../data_processed/feature_selection/drop_*.pkl')))
        pickle.dump(list(score_drop.sort_values('mean_drop').index[:min_eliminate]), open('../data_processed/feature_selection/drop_%s.pkl' % n, 'wb'))


def fine_tune_eliminate():
    all_features_file = '../data_processed/feature_selection/all_features.pkl'
    file_cols = pickle.load(open('../data_processed/feature_selection/file_cols.pkl', 'rb'))
    all_features = pickle.load(open(all_features_file, 'rb'))
    keep_features = ['date_block_num', 'item_id', 'shop_id', 'item_category_id'] + \
                    ['item_0', 'item_1', 'item_2', 'item_3', 'item_4'] + \
                    ['shop_0', 'shop_1', 'shop_2'] + \
                    ['cat_0', 'cat_1', 'cat_2']

    n_features = 300
    target = 'item_cnt_month'

    n = len(set(all_features) - set(keep_features)) - n_features
    drop_paths = sorted(iglob('../data_processed/feature_selection/drop_*.pkl'))
    drop_features = set()
    for i in tqdm(range(21)):
        drop_features |= set(pickle.load(open('../data_processed/feature_selection/drop_%s.pkl' % i, 'rb'))) - set(keep_features)

    try_features = set(all_features) - drop_features
    len(try_features)

    paths = ['../data_processed/add_some_features1.snappy'] + \
            list(iglob('../data_processed/knn/knn*.snappy'))

    matrix = []
    for p in tqdm(paths):
        _ = pd.read_parquet(p, columns=list(set(file_cols[p]) & set(try_features)))
        if _.columns.shape[0] > 0:
            matrix.append(_)
    matrix = pd.concat(matrix, axis=1)
    del _
    gc.collect()

    utils.df_info(matrix)
    # index = np.random.choice(matrix[matrix.date_block_num > 11].index, 50001, False)
    # matrix = matrix.loc[index]
    # gc.collect()

    matrix.to_parquet('../data_tmp/tmp.snappy', compression='snappy')
    matrix = pd.read_parquet('../data_tmp/tmp.snappy')
    gc.collect()

    train_index = matrix[(matrix.date_block_num < 33) & (matrix.date_block_num > 11)].index
    valid_index = matrix[matrix.date_block_num == 33].index
    del matrix; gc.collect()
    drop_cols, elimination_path, model, random_string, score_drop = utils.recursive_features_eliminate(path_to_df='../data_tmp/tmp.snappy',
                                     target='item_cnt_month',
                                     max_iter=20,
                                     always_drop=['ID'],
                                     keep_cols=['date_block_num', 'item_id', 'shop_id', 'item_category_id', 'item_0', 'item_1', 'item_2', 'item_3', 'item_4', 'shop_0', 'shop_1', 'shop_2', 'cat_0', 'cat_1', 'cat_2'],
                                     tstat_thresh=0, # tstat = (mean - base_model) / std threshold to drop features from the model
                                     max_drop_ratio=0.2, # fraction of features to drop from the non-eliminated features,
                                     min_drop=0,
                                     train_index=train_index,
                                     valid_index=valid_index,
                                     reload_data=False, # if memory is bottleneck deleting and re-reading data on each iteration
                                     model_type='rf', # model to train and use for elimination
                                     clip_down=0,
                                     clip_up=20,
                                     max_xgb_features=200, # memory issues if >200 columns
                                     num_permutations=5, # number of permutation for feature to get stats on baseline score drop
                                     n_estimators=10,
                                     max_depth=8,
                                     max_samples=0.8,
                                     seed=123,
                                     n_jobs=8,
                                     learning_rate=0.2,
                                     )
        n = len(sorted(iglob('../data_processed/feature_selection/drop_*.pkl')))
        assert not os.path.isfile('../data_processed/feature_selection/drop_%s.pkl' % n)
        pickle.dump(list(drop_cols), open('../data_processed/feature_selection/drop_%s.pkl' % n, 'wb'))

def mean_encode_categories():
    all_features_file = '../data_processed/feature_selection/all_features.pkl'
    file_cols = pickle.load(open('../data_processed/feature_selection/file_cols.pkl', 'rb'))
    all_features = pickle.load(open(all_features_file, 'rb'))
    keep_features = ['date_block_num', 'item_id', 'shop_id', 'item_category_id'] + \
                    ['item_0', 'item_1', 'item_2', 'item_3', 'item_4'] + \
                    ['shop_0', 'shop_1', 'shop_2'] + \
                    ['cat_0', 'cat_1', 'cat_2']

    n_features = 300
    target = 'item_cnt_month'

    n = len(set(all_features) - set(keep_features)) - n_features
    drop_paths = sorted(iglob('../data_processed/feature_selection/drop_*.pkl'))
    drop_features = set()
    for i in tqdm(range(22)):
        drop_features |= set(pickle.load(open('../data_processed/feature_selection/drop_%s.pkl' % i, 'rb'))) - set(keep_features)

    try_features = set(all_features) - drop_features
    len(try_features)

    paths = ['../data_processed/add_some_features1.snappy'] + \
            list(iglob('../data_processed/knn/knn*.snappy'))

    matrix = []
    for p in tqdm(paths):
        _ = pd.read_parquet(p, columns=list(set(file_cols[p]) & set(try_features)))
        if _.columns.shape[0] > 0:
            matrix.append(_)
    matrix = pd.concat(matrix, axis=1)
    del _
    gc.collect()

    utils.df_info(matrix)
    # index = np.random.choice(matrix[matrix.date_block_num > 11].index, 50001, False)
    # matrix = matrix.loc[index]
    # gc.collect()

    for col in tqdm(['item_0', 'item_1', 'item_2', 'item_3', 'item_4', 'shop_0', 'shop_1', 'shop_2', 'cat_0', 'cat_1', 'cat_2']):
        cname = '_'.join([col, 'avg', 'item', 'lag_1'])
        if cname not in drop_features:
            matrix[cname] = utils.expanding_mean_encoding(matrix, target='item_cnt_month', group_cols=[col], global_mean=None)
        # matrix = utils.lag_feature(matrix, [1], cname)
        # matrix.drop(cname, axis=1, inplace=True)

    matrix.to_parquet('../data_tmp/tmp.snappy', compression='snappy')
    matrix = pd.read_parquet('../data_tmp/tmp.snappy')
    gc.collect()

    train_index = matrix[(matrix.date_block_num < 33) & (matrix.date_block_num > 11)].index
    valid_index = matrix[matrix.date_block_num == 33].index
    del matrix; gc.collect()
    drop_cols, elimination_path, model, random_string, score_drop = utils.recursive_features_eliminate(path_to_df='../data_tmp/tmp.snappy',
                                     target='item_cnt_month',
                                     max_iter=20,
                                     always_drop=['ID'],
                                     keep_cols=['date_block_num', 'item_id', 'shop_id', 'item_category_id', 'item_0', 'item_1', 'item_2', 'item_3', 'item_4', 'shop_0', 'shop_1', 'shop_2', 'cat_0', 'cat_1', 'cat_2'],
                                     tstat_thresh=0, # tstat = (mean - base_model) / std threshold to drop features from the model
                                     max_drop_ratio=0.2, # fraction of features to drop from the non-eliminated features,
                                     min_drop=0,
                                     train_index=train_index,
                                     valid_index=valid_index,
                                     reload_data=False, # if memory is bottleneck deleting and re-reading data on each iteration
                                     model_type='rf', # model to train and use for elimination
                                     clip_down=0,
                                     clip_up=20,
                                     max_xgb_features=200, # memory issues if >200 columns
                                     num_permutations=5, # number of permutation for feature to get stats on baseline score drop
                                     n_estimators=10,
                                     max_depth=8,
                                     max_samples=0.8,
                                     seed=123,
                                     n_jobs=8,
                                     learning_rate=0.2,
                                     )
        n = len(sorted(iglob('../data_processed/feature_selection/drop_*.pkl')))
        assert not os.path.isfile('../data_processed/feature_selection/drop_%s.pkl' % n)
        pickle.dump(list(drop_cols), open('../data_processed/feature_selection/drop_%s.pkl' % n, 'wb'))

def try_model():
    all_features_file = '../data_processed/feature_selection/all_features.pkl'
    file_cols = pickle.load(open('../data_processed/feature_selection/file_cols.pkl', 'rb'))
    all_features = pickle.load(open(all_features_file, 'rb'))
    keep_features = ['date_block_num', 'item_id', 'shop_id', 'item_category_id'] + \
                    ['item_0', 'item_1', 'item_2', 'item_3', 'item_4'] + \
                    ['shop_0', 'shop_1', 'shop_2'] + \
                    ['cat_0', 'cat_1', 'cat_2']

    target = 'item_cnt_month'

    n_features = 300
    n = len(set(all_features) - set(keep_features)) - n_features
    drop_paths = sorted(iglob('../data_processed/feature_selection/drop_*.pkl'))
    drop_features = set()
    for i in tqdm(range(22)):
        drop_features |= set(pickle.load(open('../data_processed/feature_selection/drop_%s.pkl' % i, 'rb'))) - set(keep_features)

    try_features = set(all_features) - drop_features
    len(try_features)

    paths = ['../data_processed/add_some_features1.snappy'] + \
            list(iglob('../data_processed/knn/knn*.snappy'))

    matrix = []
    for p in tqdm(paths):
        _ = pd.read_parquet(p, columns=list(set(file_cols[p]) & set(try_features)))
        if _.columns.shape[0] > 0:
            matrix.append(_)
    matrix = pd.concat(matrix, axis=1)
    del _
    gc.collect()

    utils.df_info(matrix)
    # index = np.random.choice(matrix[matrix.date_block_num > 11].index, 50001, False)
    # matrix = matrix.loc[index]
    # gc.collect()

    for col in tqdm(['item_0', 'item_1', 'item_2', 'item_3', 'item_4', 'shop_0', 'shop_1', 'shop_2', 'cat_0', 'cat_1', 'cat_2']):
        cname = '_'.join([col, 'avg', 'item', 'lag_1'])
        if cname not in drop_features:
            matrix[cname] = utils.expanding_mean_encoding(matrix, target='item_cnt_month', group_cols=[col], global_mean=None)
        # matrix = utils.lag_feature(matrix, [1], cname)
        # matrix.drop(cname, axis=1, inplace=True)

    for c in tqdm(matrix.columns):
        if matrix[c].dtype == 'float64':
            matrix[c] = matrix[c].astype(np.float32)
    matrix.to_parquet('../data_processed/item-item_shop-shop_filtered.snappy', compression='snappy')
    matrix = pd.read_parquet('../data_processed/item-item_shop-shop_filtered.snappy')

    data = matrix[matrix.date_block_num > 11]
    del matrix; gc.collect()
    drop_cols = ['ID']
    data.drop(drop_cols, axis=1, inplace=True)
    data.info()

    X_train = data[data.date_block_num < 33].drop(['item_cnt_month', 'date_block_num', ], axis=1)
    Y_train = data[data.date_block_num < 33]['item_cnt_month']
    X_valid = data[data.date_block_num == 33].drop(['item_cnt_month', 'date_block_num', ], axis=1)
    Y_valid = data[data.date_block_num == 33]['item_cnt_month']
    X_test = data[data.date_block_num == 34].drop(['item_cnt_month', 'date_block_num', ], axis=1)

    del data
    gc.collect()

    X_train.info()

    # [62]	validation_0-rmse:0.79811	validation_1-rmse:0.89230

    ts = time.time()
    model = XGBRegressor(
        max_depth=8,
        n_estimators=150,
        min_child_weight=300,
        colsample_bytree=0.8,
        subsample=0.8,
        eta=0.02,
        seed=42,
        nthread=7)

    model.fit(
        X_train,
        Y_train,
        eval_metric="rmse",
        eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
        verbose=True,
        early_stopping_rounds = 10)
    time.time() - ts


    terr = mean_squared_error(Y_train.values.astype('float32'), model.predict(X_train).clip(0, 20))
    verr = mean_squared_error(Y_valid.values.astype('float32'), model.predict(X_valid).clip(0, 20))
    print('train-rmse=%.4f valid-rmse=%.4f' % (terr, verr))

    Y_pred = model.predict(X_valid).clip(0, 20)
    Y_test = model.predict(X_test).clip(0, 20)

    # from collections import Counter
    # from sklearn.metrics import confusion_matrix
    # Y_pred = model.predict(X_valid).clip(0, 20)
    # sorted(Counter(Y_pred.round(0)).items())
    # sorted(Counter(Y_valid).items())
    # cm = confusion_matrix(Y_valid.values.astype('int8'), Y_pred.round(0).astype('int8'))
    # pd.DataFrame(cm).to_csv('../data/errors.xgb.best.csv')

    # Y_test = model.predict(X_test).clip(0, 20)
    test = pd.read_csv('../data/test.csv')
    submission = pd.DataFrame({
        "ID": test.index,
        "item_cnt_month": Y_test
    })
    submission.to_csv('../submission/xgb_submission_2020-07-06_01.csv', index=False)
    pd.Series(Y_pred).to_csv('../submission/xgb_submission_2020-07-06_01_valid.csv', index=False)

    base_score = verr
    features = X_train.columns.tolist()
    score_drop = {}
    imps = pd.Series({f: v for f, v in zip(features, model.feature_importances_)}).sort_values()
    # going only through worst 100 features
    for ci in tqdm(range(len(imps))):
        temp_df = X_valid.copy()
        cur_score = []
        for _ in range(5):
            temp_df.iloc[:, ci] = np.random.permutation(temp_df.iloc[:, ci])
            val_pred = model.predict(temp_df)
            cur_score.append(mean_squared_error(Y_valid, val_pred.clip(0, 20)))
        _m = np.array(cur_score).mean()
        _s = np.array(cur_score).std()
        score_drop[features[ci]] = (_m - base_score, (_m - base_score) / (_s + 1e-10))
    score_drop = pd.DataFrame({f: {'mean_drop': a, 'tstat_drop': b} for f, (a, b) in score_drop.items()}).T
    score_drop.sort_values('mean_drop').to_csv('../data_tmp/tmp_drop.csv')
    imps.sort_values().to_csv('../data_tmp/tmp_imp.csv')

def add_non_leafs_as_features():
    matrix = pd.read_parquet('../data_processed/item-item_shop-shop_filtered.snappy')

    drop_cols, clusters = utils.find_duplicates_features(matrix,
                                                         id_cols=['item_cnt_month', 'date_block_num', 'shop_id', 'item_id', 'item_category_id'],
                                                         date_col='date_block_num',
                                                         date_thresh=11,
                                                         corr_thresh=0.97,
                                                         corr_sample_size=10000,
                                                         n_jobs=8,)

    from sklearn.ensemble import RandomForestRegressor

    data = matrix[matrix.date_block_num > 11]
    del matrix; gc.collect()
    drop_cols = ['ID']
    data.drop(drop_cols, axis=1, inplace=True)
    data.info()
    data.fillna(0, inplace=True)

    data.isna().sum(axis=0)
    X_train = data[data.date_block_num < 33].drop(['item_cnt_month', 'date_block_num', ], axis=1)
    Y_train = data[data.date_block_num < 33]['item_cnt_month']
    X_valid = data[data.date_block_num == 33].drop(['item_cnt_month', 'date_block_num', ], axis=1)
    Y_valid = data[data.date_block_num == 33]['item_cnt_month']
    X_test = data[data.date_block_num == 34].drop(['item_cnt_month', 'date_block_num', ], axis=1)

    del data
    gc.collect()

    model = RandomForestRegressor(n_estimators=10,
                                  max_depth=10,
                                  max_features='sqrt',
                                  max_samples=0.8,
                                  random_state=seed,
                                  n_jobs=1) # all cores
    model.fit(X_train, Y_train)

    X_train.info()

    # [62]	validation_0-rmse:0.79811	validation_1-rmse:0.89230

    ts = time.time()
    model = XGBRegressor(
        max_depth=8,
        n_estimators=150,
        min_child_weight=300,
        colsample_bytree=0.8,
        subsample=0.8,
        eta=0.02,
        seed=42,
        nthread=7)

    model.fit(
        X_train,
        Y_train,
        eval_metric="rmse",
        eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
        verbose=True,
        early_stopping_rounds = 10)
    time.time() - ts


    terr = mean_squared_error(Y_train.values.astype('float32'), model.predict(X_train).clip(0, 20))
    verr = mean_squared_error(Y_valid.values.astype('float32'), model.predict(X_valid).clip(0, 20))
    print('train-rmse=%.4f valid-rmse=%.4f' % (terr, verr))

    Y_pred = model.predict(X_valid).clip(0, 20)
    Y_test = model.predict(X_test).clip(0, 20)

    # from collections import Counter
    # from sklearn.metrics import confusion_matrix
    # Y_pred = model.predict(X_valid).clip(0, 20)
    # sorted(Counter(Y_pred.round(0)).items())
    # sorted(Counter(Y_valid).items())
    # cm = confusion_matrix(Y_valid.values.astype('int8'), Y_pred.round(0).astype('int8'))
    # pd.DataFrame(cm).to_csv('../data/errors.xgb.best.csv')

    # Y_test = model.predict(X_test).clip(0, 20)
    test = pd.read_csv('../data/test.csv')
    submission = pd.DataFrame({
        "ID": test.index,
        "item_cnt_month": Y_test
    })
    submission.to_csv('../submission/xgb_submission_2020-07-06_02.csv', index=False)
    pd.Series(Y_pred).to_csv('../submission/xgb_submission_2020-07-06_02_valid.csv', index=False)

    base_score = verr
    features = X_train.columns.tolist()
    score_drop = {}
    imps = pd.Series({f: v for f, v in zip(features, model.feature_importances_)}).sort_values()
    # going only through worst 100 features
    for ci in tqdm(range(len(imps))):
        temp_df = X_valid.copy()
        cur_score = []
        for _ in range(5):
            temp_df.iloc[:, ci] = np.random.permutation(temp_df.iloc[:, ci])
            val_pred = model.predict(temp_df)
            cur_score.append(mean_squared_error(Y_valid, val_pred.clip(0, 20)))
        _m = np.array(cur_score).mean()
        _s = np.array(cur_score).std()
        score_drop[features[ci]] = (_m - base_score, (_m - base_score) / (_s + 1e-10))
    score_drop = pd.DataFrame({f: {'mean_drop': a, 'tstat_drop': b} for f, (a, b) in score_drop.items()}).T
    score_drop.sort_values('mean_drop').to_csv('../data_tmp/tmp_drop.csv')
    imps.sort_values().to_csv('../data_tmp/tmp_imp.csv')


bad_features = pd.read_csv('../data_processed/bad_features.csv')


keep_cols = set()
# per feature log return preparation
for knn in tqdm([3, 10, 50, 100]):
    for lag in [2, 3, 6, 12]:
        for prefix in ['item']: # 'item_shop',
            for fn in ['mean', 'median']:
                for var in ['item', 'fsale', 'price', 'rev']:
                    for knn_prefix in ['item', 'item_shop']:
                        col_name = '%s_knn_%s_%s_%s_delta_%s' % (knn_prefix, knn, fn, var, lag)
                        c2 = '%s_knn_%s_%s_%s_w%s_lag_%s' % (knn_prefix, knn, fn, var, 12, 1)
                        c1 = '%s_knn_%s_%s_%s_w%s_lag_%s' % (knn_prefix, knn, fn, var,  1, lag)
                        keep_cols.add(c1)
                        keep_cols.add(c2)


matrix = []
for lag in tqdm([1]):
    win_size_list = [12]
    for win_size in win_size_list:
        bn = 'knn_%s_win%s_lag%s' % (knn_prefix, win_size, lag)
        # matrix = pd.concat([matrix, pd.read_parquet('../data_processed/%s.snappy' % bn)], axis=1)
        # _ = pd.read_parquet('../data_processed/%s.snappy' % bn)
        _ = pd.read_parquet('../data_processed/%s.snappy' % bn)
        drop_cols = [c for c in _.columns if '_sum_' in c]
        matrix.append(_.drop(drop_cols, axis=1))
        gc.collect()
matrix = pd.concat(matrix, axis=1)
utils.df_info(matrix)



for lag in tqdm([1, 2, 3, 6, 12]):
    win_size_list = [1]
    for win_size in win_size_list:
        bn = 'knn_%s_win%s_lag%s' % (knn_prefix, win_size, lag)
        # matrix = pd.concat([matrix, pd.read_parquet('../data_processed/%s.snappy' % bn)], axis=1)
        # _ = pd.read_parquet('../data_processed/%s.snappy' % bn)
        _ = pd.read_parquet('../data_processed/%s.snappy' % bn)
        drop_cols = [c for c in _.columns if '_sum_' in c]
        matrix = matrix.merge(_.drop(drop_cols, axis=1), how='left', left_index=True, right_index=True)
        _cols = _.columns
        del _
        for knn in [3, 10, 50, 100]:
            for fn in ['mean', 'median']:
                for var in ['item', 'fsale', 'price', 'rev']:
                    col_name = '%s_knn_%s_%s_%s_delta_%s' % (knn_prefix, knn, fn, var, lag)
                    c2 = '%s_knn_%s_%s_%s_w%s_lag_%s' % (knn_prefix, knn, fn, var, 12, 1)
                    c1 = '%s_knn_%s_%s_%s_w%s_lag_%s' % (knn_prefix, knn, fn, var,  1, lag)
                    matrix[col_name] = np.log1p(np.clip(matrix[c1] / (matrix[c2] + 1e-6), -1 + 1e-6, None))
        drop_cols = [c for c in matrix.columns if c in bad_features and c in _cols]
        matrix.drop(drop_cols, axis=1, inplace=True)
        print('n cols=%s' % matrix.shape[1])
        gc.collect()

matrix = matrix.merge(pd.read_parquet('../data_processed/tmp.snappy'), how='right', left_index=True, right_index=True)
matrix.to_parquet('../data_processed/%s_filtered.snappy' % knn_prefix, compression='snappy')

matrix = pd.read_parquet('../data_processed/%s_filtered.snappy' % knn_prefix)
matrix = matrix[matrix.date_block_num > 11]
gc.collect()
matrix.to_parquet('../data_processed/%s_filtered1.snappy' % knn_prefix, compression='snappy')

del matrix; gc.collect()
matrix = pd.read_parquet('../data_processed/%s_filtered1.snappy' % knn_prefix)
index = np.random.choice(matrix.index, 10000, False)
matrix = matrix.loc[index]
gc.collect()
drop_cols, clusters = utils.find_duplicates_features(matrix,
                                                     id_cols=['item_cnt_month', 'date_block_num', 'ID', 'shop_id', 'item_id', 'item_category_id'],
                                                     date_col='date_block_num',
                                                     date_thresh=11,
                                                     corr_thresh=0.97,
                                                     corr_sample_size=10000,
                                                     n_jobs=8,)
del matrix; gc.collect()
matrix = pd.read_parquet('../data_processed/%s_filtered.snappy' % knn_prefix, columns=['item_shop_knn_50_max_fsale_w1_lag_1', 'item_3', 'item_shop_knn_10_median_price_w1_lag_2', 'item_shop_knn_3_min_price_w1_lag_6', 'item_shop_knn_50_min_fsale_w1_lag_2', 'item_shop_knn_50_mean_fsale_w1_lag_12', 'item_shop_knn_50_median_fsale_w1_lag_6', 'item_shop_knn_100_median_rev_w1_lag_6', 'item_shop_knn_100_median_price_delta_6', 'item_shop_knn_3_max_price_w1_lag_1', 'item_shop_knn_3_mean_item_w1_lag_6', 'item_shop_knn_10_mean_rev_w1_lag_3', 'item_shop_knn_10_min_item_w1_lag_3', 'item_shop_knn_50_max_price_w12_lag_1', 'item_shop_knn_10_min_fsale_w1_lag_3', 'item_shop_knn_50_mean_fsale_w1_lag_6', 'item_shop_knn_100_mean_item_w1_lag_12', 'item_shop_knn_100_mean_item_delta_1', 'item_shop_knn_50_mean_rev_w1_lag_12', 'item_shop_knn_10_max_rev_w1_lag_6', 'item_shop_knn_100_mean_price_w1_lag_2', 'item_shop_knn_100_max_price_w1_lag_6', 'item_shop_knn_100_median_price_delta_3', 'item_shop_knn_10_mean_price_w1_lag_6', 'item_shop_knn_10_mean_fsale_w1_lag_1', 'item_shop_knn_50_mean_fsale_w1_lag_3', 'item_shop_knn_3_min_price_w1_lag_12', 'item_shop_knn_10_min_fsale_w12_lag_1', 'item_shop_knn_50_mean_price_delta_3', 'item_shop_knn_100_min_item_w12_lag_1', 'item_shop_knn_10_max_item_w1_lag_3', 'item_shop_knn_3_mean_price_w12_lag_1', 'item_shop_knn_3_median_fsale_delta_2', 'item_shop_knn_50_min_price_w1_lag_3', 'date_block_num', 'item_shop_knn_10_median_item_w1_lag_2', 'item_shop_knn_10_min_item_w1_lag_2', 'item_shop_knn_50_median_price_w1_lag_1', 'item_cnt_month', 'item_shop_knn_50_min_rev_w1_lag_6', 'item_shop_knn_50_mean_fsale_delta_2', 'item_shop_knn_100_median_rev_w1_lag_3', 'item_shop_knn_10_mean_fsale_delta_12', 'item_shop_knn_50_min_price_w12_lag_1', 'item_2', 'item_shop_knn_100_mean_fsale_delta_3', 'item_shop_knn_100_max_item_w12_lag_1', 'item_shop_knn_10_median_fsale_delta_1', 'item_shop_knn_10_min_rev_w1_lag_6', 'item_shop_knn_3_max_price_w1_lag_6', 'item_shop_knn_3_median_item_delta_1', 'item_shop_knn_100_median_price_delta_2', 'item_shop_knn_3_median_rev_w1_lag_1', 'item_shop_knn_3_mean_price_delta_6', 'item_shop_knn_100_mean_rev_w1_lag_3', 'item_shop_knn_50_min_fsale_w1_lag_3', 'item_shop_knn_50_mean_item_w1_lag_6', 'item_shop_knn_10_mean_fsale_w1_lag_3', 'item_shop_knn_100_min_price_w1_lag_3', 'item_shop_knn_100_mean_fsale_w12_lag_1', 'item_shop_knn_100_mean_rev_w1_lag_12', 'item_4', 'item_shop_knn_100_median_fsale_w1_lag_2', 'item_shop_knn_10_min_item_w1_lag_1', 'item_shop_knn_100_median_item_delta_6', 'item_shop_knn_3_median_item_w1_lag_6', 'item_shop_knn_10_median_price_w1_lag_3', 'item_shop_knn_50_max_item_w1_lag_12', 'item_shop_knn_10_mean_price_w1_lag_2', 'item_shop_knn_3_mean_price_w1_lag_1', 'item_shop_knn_3_min_rev_w1_lag_6', 'item_shop_knn_3_median_price_w1_lag_6', 'item_shop_knn_100_mean_price_w1_lag_12', 'item_shop_knn_10_median_price_delta_1', 'item_shop_knn_10_max_price_w1_lag_3', 'item_shop_knn_10_max_rev_w1_lag_12', 'item_shop_knn_3_median_item_delta_3', 'item_shop_knn_3_median_price_delta_3', 'item_shop_knn_50_min_rev_w1_lag_3', 'item_shop_knn_3_min_rev_w1_lag_12', 'item_shop_knn_10_max_price_w1_lag_2', 'item_shop_knn_3_median_item_w12_lag_1', 'item_shop_knn_100_median_price_delta_12', 'item_shop_knn_10_max_item_w1_lag_2', 'item_shop_knn_100_mean_fsale_w1_lag_1', 'item_shop_knn_100_median_fsale_w1_lag_1', 'item_shop_knn_10_mean_fsale_delta_3', 'item_shop_knn_10_median_item_w12_lag_1', 'item_shop_knn_50_min_item_w1_lag_3', 'item_shop_knn_50_median_price_w1_lag_2', 'cat_0', 'item_shop_knn_10_mean_item_w1_lag_1', 'item_shop_knn_3_mean_item_delta_3', 'item_shop_knn_10_max_item_w12_lag_1', 'item_shop_knn_50_mean_item_w1_lag_2', 'item_shop_knn_50_median_rev_w1_lag_3', 'item_shop_knn_100_min_rev_w1_lag_6', 'item_shop_knn_10_mean_rev_w12_lag_1', 'item_shop_knn_50_mean_price_delta_12', 'item_first_sale', 'item_shop_knn_3_max_fsale_w1_lag_6', 'item_shop_knn_100_min_rev_w1_lag_2', 'item_shop_knn_3_min_price_w1_lag_1', 'item_shop_knn_50_min_fsale_w1_lag_6', 'item_shop_knn_100_min_price_w1_lag_12', 'item_shop_knn_100_max_item_w1_lag_1', 'item_shop_knn_10_mean_item_delta_6', 'item_shop_knn_10_min_rev_w1_lag_3', 'item_shop_knn_3_median_price_delta_1', 'item_shop_knn_10_mean_item_w1_lag_2', 'item_shop_knn_10_max_price_w1_lag_12', 'item_shop_knn_3_min_fsale_w1_lag_12', 'item_shop_knn_50_median_price_delta_1', 'item_shop_knn_50_mean_item_w1_lag_12', 'item_shop_knn_50_min_price_w1_lag_2', 'item_shop_knn_3_median_rev_w1_lag_6', 'item_shop_knn_10_median_fsale_delta_12', 'item_shop_knn_50_median_price_delta_3', 'item_shop_knn_100_median_item_w1_lag_6', 'item_shop_knn_10_mean_price_w1_lag_12', 'item_shop_knn_3_max_item_w1_lag_6', 'item_shop_knn_3_mean_fsale_w1_lag_12', 'item_shop_knn_50_min_rev_w1_lag_12', 'item_shop_knn_10_max_fsale_w1_lag_3', 'item_shop_knn_50_mean_price_w1_lag_2', 'item_shop_knn_3_mean_fsale_w1_lag_2', 'item_shop_knn_50_median_fsale_delta_3', 'item_shop_knn_10_min_price_w1_lag_12', 'item_shop_knn_50_median_item_delta_1', 'date_shop_revenue', 'date_item_shop_avg_item_price', 'item_shop_knn_50_mean_fsale_w1_lag_1', 'item_shop_knn_3_median_fsale_w1_lag_6', 'item_shop_knn_3_max_rev_w1_lag_12', 'item_shop_knn_100_min_rev_w1_lag_3', 'item_shop_knn_50_median_fsale_w1_lag_12', 'item_shop_knn_10_median_item_delta_3', 'item_shop_knn_50_max_rev_w1_lag_6', 'item_shop_knn_10_mean_price_delta_1', 'item_shop_knn_10_median_price_delta_12', 'item_shop_knn_50_mean_price_w1_lag_3', 'item_shop_knn_3_median_price_delta_12', 'item_shop_knn_10_min_rev_w12_lag_1', 'item_shop_knn_3_mean_fsale_w1_lag_3', 'item_shop_knn_10_mean_rev_w1_lag_1', 'item_shop_knn_10_median_price_delta_6', 'item_shop_knn_3_max_price_w1_lag_2', 'item_shop_knn_50_mean_fsale_delta_6', 'item_shop_knn_50_median_item_delta_6', 'item_shop_knn_50_mean_fsale_delta_12', 'item_shop_knn_100_mean_fsale_w1_lag_2', 'item_shop_knn_50_max_item_w1_lag_2', 'item_shop_knn_100_median_price_w12_lag_1', 'item_shop_knn_3_mean_rev_w1_lag_3', 'item_shop_knn_10_median_item_delta_1', 'item_shop_knn_50_max_price_w1_lag_1', 'item_shop_knn_3_median_item_w1_lag_12', 'item_shop_knn_100_max_item_w1_lag_2', 'item_shop_knn_10_median_fsale_w1_lag_2', 'item_shop_knn_100_mean_item_w12_lag_1', 'item_shop_knn_100_max_item_w1_lag_6', 'item_shop_knn_100_mean_rev_w1_lag_2', 'item_shop_knn_10_mean_fsale_w1_lag_2', 'item_shop_knn_10_min_price_w1_lag_2', 'item_shop_knn_10_max_fsale_w1_lag_6', 'item_shop_knn_3_max_item_w1_lag_3', 'item_shop_knn_3_median_rev_w1_lag_12', 'item_shop_knn_50_mean_item_delta_2', 'item_shop_knn_100_mean_fsale_delta_12', 'item_shop_knn_10_mean_rev_w1_lag_12', 'item_shop_knn_3_median_item_w1_lag_3', 'item_shop_knn_3_max_fsale_w1_lag_12', 'item_shop_knn_50_min_rev_w1_lag_2', 'item_shop_knn_3_mean_rev_w12_lag_1', 'item_shop_knn_100_max_price_w1_lag_2', 'item_shop_knn_3_min_fsale_w1_lag_6', 'item_shop_knn_100_median_fsale_w1_lag_12', 'item_shop_knn_50_median_rev_w1_lag_6', 'item_shop_knn_50_median_price_delta_6', 'item_shop_knn_3_mean_fsale_w1_lag_1', 'item_shop_knn_100_median_item_w12_lag_1', 'item_shop_knn_50_min_fsale_w12_lag_1', 'item_shop_knn_10_max_item_w1_lag_6', 'item_shop_knn_100_median_rev_w1_lag_2', 'item_shop_knn_50_median_item_w12_lag_1', 'item_shop_knn_50_median_item_w1_lag_2', 'item_shop_knn_3_min_price_w1_lag_2', 'item_shop_knn_100_mean_item_delta_3', 'item_shop_knn_50_mean_item_delta_12', 'item_shop_knn_10_median_rev_w1_lag_6', 'item_shop_knn_10_median_price_delta_2', 'item_shop_knn_3_median_fsale_delta_3', 'item_shop_knn_10_min_price_w1_lag_3', 'item_cnt_month_lag_6', 'item_shop_knn_10_mean_price_delta_12', 'item_shop_knn_3_median_fsale_delta_12', 'item_shop_knn_10_min_rev_w1_lag_2', 'item_shop_knn_50_max_rev_w1_lag_3', 'item_shop_knn_10_mean_fsale_delta_2', 'item_shop_knn_10_max_price_w1_lag_6', 'item_shop_knn_100_min_price_w1_lag_6', 'item_shop_knn_10_mean_item_delta_3', 'item_shop_knn_3_max_rev_w1_lag_2', 'item_shop_knn_3_median_item_delta_2', 'item_shop_knn_50_median_item_w1_lag_12', 'item_shop_knn_3_max_item_w1_lag_2', 'item_shop_knn_10_min_fsale_w1_lag_6', 'item_shop_knn_100_min_fsale_w1_lag_2', 'item_shop_knn_100_max_rev_w12_lag_1', 'item_shop_knn_3_max_item_w1_lag_1', 'item_shop_knn_50_min_item_w1_lag_12', 'item_shop_knn_3_min_item_w1_lag_12', 'item_shop_knn_50_mean_price_delta_2', 'item_shop_knn_10_mean_item_delta_2', 'item_shop_knn_10_median_rev_w12_lag_1', 'item_shop_knn_3_max_item_w1_lag_12', 'item_shop_knn_50_mean_item_w1_lag_3', 'item_shop_knn_100_median_fsale_delta_6', 'item_shop_knn_3_mean_rev_w1_lag_12', 'item_cnt_month_lag_12', 'item_shop_knn_50_median_item_delta_12', 'item_shop_knn_10_median_price_w1_lag_1', 'item_shop_knn_10_median_item_delta_2', 'item_shop_knn_100_min_rev_w12_lag_1', 'item_shop_knn_50_min_price_w1_lag_1', 'item_cnt_month_lag_3', 'item_shop_knn_10_median_rev_w1_lag_3', 'date_item_shop_revenue', 'item_shop_knn_50_mean_price_w12_lag_1', 'item_shop_knn_3_max_fsale_w1_lag_2', 'item_shop_knn_100_median_item_w1_lag_2', 'item_shop_knn_50_mean_price_delta_6', 'item_shop_knn_50_median_rev_w1_lag_1', 'item_cnt_month_lag_2', 'item_shop_knn_3_min_price_w1_lag_3', 'cat_1', 'item_shop_knn_10_max_fsale_w1_lag_12', 'item_shop_knn_10_median_fsale_delta_6', 'item_shop_knn_10_median_item_w1_lag_3', 'item_shop_knn_10_max_fsale_w1_lag_1', 'item_shop_knn_100_median_price_w1_lag_2', 'item_category_id', 'shop_2', 'item_shop_knn_3_max_fsale_w1_lag_1', 'item_shop_knn_10_mean_fsale_delta_6', 'item_shop_knn_50_median_item_w1_lag_3', 'item_shop_knn_10_max_fsale_w12_lag_1', 'delta_revenue', 'item_shop_knn_3_mean_rev_w1_lag_6', 'item_shop_knn_3_max_price_w1_lag_3', 'item_shop_knn_10_min_fsale_w1_lag_2', 'item_shop_knn_3_mean_price_w1_lag_3', 'item_shop_knn_3_min_fsale_w1_lag_1', 'item_shop_knn_10_median_item_delta_12', 'item_shop_knn_10_mean_item_delta_1', 'item_shop_knn_50_mean_price_w1_lag_12', 'item_shop_knn_10_max_price_w12_lag_1', 'item_shop_knn_10_median_fsale_w1_lag_1', 'item_shop_knn_100_mean_item_w1_lag_3', 'item_shop_knn_50_median_price_w1_lag_6', 'item_shop_knn_10_median_fsale_w1_lag_12', 'item_shop_knn_100_median_price_delta_1', 'item_shop_knn_10_max_fsale_w1_lag_2', 'item_shop_knn_100_min_fsale_w12_lag_1', 'item_shop_knn_50_median_item_delta_2', 'item_shop_knn_10_min_rev_w1_lag_1', 'ID', 'item_shop_knn_100_mean_fsale_delta_2', 'item_shop_knn_3_mean_price_delta_12', 'item_shop_knn_3_median_price_delta_6', 'item_shop_knn_100_min_item_w1_lag_12', 'item_shop_knn_100_median_rev_w1_lag_12', 'item_shop_knn_3_mean_item_delta_6', 'item_shop_knn_3_mean_item_delta_1', 'item_shop_knn_10_median_item_w1_lag_12', 'item_shop_knn_50_median_item_w1_lag_1', 'item_shop_knn_100_median_fsale_w1_lag_3', 'item_shop_knn_100_min_fsale_w1_lag_1', 'item_shop_knn_100_mean_rev_w12_lag_1', 'item_shop_knn_10_median_fsale_w1_lag_6', 'item_shop_knn_3_median_rev_w1_lag_2', 'item_shop_knn_100_median_fsale_w1_lag_6', 'item_shop_knn_10_mean_price_w1_lag_1', 'shop_0', 'item_shop_knn_100_median_price_w1_lag_12', 'item_shop_knn_100_mean_price_delta_1', 'item_shop_knn_100_median_item_delta_3', 'item_1', 'item_shop_knn_10_min_price_w12_lag_1', 'item_shop_knn_50_median_fsale_delta_2', 'item_shop_knn_3_mean_fsale_delta_1', 'item_shop_knn_50_max_price_w1_lag_6', 'item_shop_knn_100_min_fsale_w1_lag_3', 'item_shop_knn_100_mean_item_delta_2', 'item_shop_knn_10_min_fsale_w1_lag_12', 'item_shop_knn_100_mean_price_delta_6', 'item_shop_knn_3_median_item_w1_lag_1', 'item_shop_knn_50_median_fsale_delta_1', 'item_shop_knn_50_mean_rev_w1_lag_1', 'item_shop_knn_3_median_fsale_delta_1', 'item_shop_knn_100_max_price_w1_lag_12', 'item_shop_knn_10_min_fsale_w1_lag_1', 'item_shop_knn_10_min_item_w12_lag_1', 'item_shop_knn_100_mean_price_delta_3', 'item_shop_knn_10_max_rev_w1_lag_1', 'item_shop_knn_3_median_fsale_w1_lag_3', 'item_0', 'item_shop_knn_3_mean_item_delta_2', 'item_shop_knn_50_min_rev_w1_lag_1', 'item_shop_knn_100_median_fsale_delta_2', 'item_shop_knn_3_max_rev_w1_lag_3', 'item_shop_knn_50_median_price_delta_2', 'item_shop_knn_100_median_item_delta_12', 'item_shop_knn_50_median_price_delta_12', 'item_shop_knn_50_mean_fsale_w1_lag_2', 'item_shop_knn_50_median_fsale_delta_12', 'item_shop_knn_3_median_price_delta_2', 'item_shop_knn_100_median_price_w1_lag_6', 'item_shop_knn_3_max_fsale_w12_lag_1', 'item_shop_knn_50_min_fsale_w1_lag_12', 'cat_2', 'item_shop_knn_50_max_price_w1_lag_3', 'item_shop_knn_3_min_item_w1_lag_3', 'item_shop_knn_10_mean_fsale_w1_lag_6', 'item_shop_knn_100_median_rev_w1_lag_1', 'item_shop_knn_10_mean_price_delta_3', 'item_shop_knn_50_median_fsale_w1_lag_3', 'item_shop_knn_3_mean_fsale_w12_lag_1', 'item_shop_knn_100_mean_rev_w1_lag_6', 'item_shop_knn_3_max_rev_w1_lag_6', 'item_shop_knn_100_min_fsale_w1_lag_12', 'item_shop_knn_3_mean_price_w1_lag_2', 'item_shop_knn_100_max_rev_w1_lag_12', 'item_shop_knn_50_min_price_w1_lag_12', 'item_shop_knn_100_min_price_w1_lag_1', 'item_shop_knn_10_median_item_delta_6', 'item_shop_knn_3_min_item_w1_lag_1', 'item_shop_knn_3_min_fsale_w1_lag_2', 'item_shop_knn_100_median_item_w1_lag_1', 'item_shop_knn_100_mean_fsale_delta_1', 'item_shop_knn_100_median_rev_w12_lag_1', 'item_shop_knn_3_mean_rev_w1_lag_2', 'item_shop_knn_3_min_fsale_w1_lag_3', 'item_shop_knn_3_min_item_w12_lag_1', 'item_shop_knn_10_mean_rev_w1_lag_6', 'shop_1', 'item_shop_knn_10_mean_rev_w1_lag_2', 'item_shop_knn_10_min_price_w1_lag_6', 'item_shop_knn_10_max_price_w1_lag_1', 'item_shop_knn_3_min_rev_w1_lag_1', 'item_shop_knn_100_mean_price_w1_lag_3', 'item_shop_knn_50_max_item_w1_lag_1', 'item_shop_knn_100_mean_item_w1_lag_6', 'item_shop_knn_3_mean_item_w1_lag_2', 'item_shop_knn_50_mean_price_w1_lag_1', 'item_shop_knn_100_median_price_w1_lag_3', 'item_shop_knn_10_min_item_w1_lag_12', 'item_shop_knn_10_median_rev_w1_lag_1', 'item_shop_knn_100_max_price_w1_lag_3', 'item_shop_knn_10_median_fsale_w1_lag_3', 'item_shop_knn_3_median_fsale_w1_lag_12', 'item_shop_knn_50_mean_rev_w1_lag_6', 'item_shop_knn_100_max_item_w1_lag_3', 'item_shop_knn_100_mean_fsale_delta_6', 'item_shop_knn_3_mean_price_delta_2', 'item_shop_knn_3_mean_price_delta_3', 'item_shop_knn_3_median_item_w1_lag_2', 'item_shop_knn_10_mean_item_w1_lag_12', 'item_shop_knn_50_median_price_w12_lag_1', 'item_shop_knn_100_median_item_w1_lag_3', 'item_shop_knn_50_median_item_delta_3', 'item_shop_knn_100_mean_item_delta_6', 'item_shop_knn_50_median_item_w1_lag_6', 'item_shop_knn_3_median_fsale_delta_6', 'item_shop_knn_10_mean_fsale_w1_lag_12', 'item_shop_knn_3_max_price_w1_lag_12', 'item_shop_knn_50_mean_item_delta_6', 'item_shop_knn_100_mean_price_delta_12', 'item_shop_knn_50_max_item_w12_lag_1', 'item_shop_knn_10_median_price_w1_lag_12', 'item_shop_knn_50_max_price_w1_lag_2', 'item_shop_knn_10_mean_item_delta_12', 'item_shop_knn_3_median_rev_w12_lag_1', 'item_shop_knn_50_mean_item_w1_lag_1', 'item_shop_knn_100_min_price_w12_lag_1', 'item_shop_knn_50_min_fsale_w1_lag_1', 'item_shop_knn_100_mean_item_w1_lag_1', 'item_shop_knn_3_mean_price_w1_lag_6', 'item_shop_knn_50_mean_fsale_w12_lag_1', 'item_shop_knn_10_median_item_w1_lag_1', 'item_shop_knn_100_max_fsale_w1_lag_1', 'item_shop_knn_10_median_fsale_delta_3', 'item_shop_knn_100_min_rev_w1_lag_1', 'item_shop_knn_10_max_item_w1_lag_1', 'item_shop_knn_3_mean_price_w1_lag_12', 'item_shop_knn_10_median_price_w1_lag_6', 'item_shop_knn_3_median_fsale_w1_lag_1', 'item_shop_knn_10_mean_fsale_delta_1', 'item_shop_knn_3_mean_item_w12_lag_1', 'item_shop_knn_10_mean_price_delta_2', 'item_shop_knn_3_min_item_w1_lag_2', 'shop_avg_revenue', 'item_shop_knn_100_median_price_w1_lag_1', 'item_shop_knn_3_min_fsale_w12_lag_1', 'item_shop_knn_50_min_item_w12_lag_1', 'item_shop_knn_50_mean_fsale_delta_3', 'item_shop_knn_3_median_price_w1_lag_3', 'item_shop_knn_10_mean_item_w12_lag_1', 'item_shop_knn_50_min_rev_w12_lag_1', 'item_shop_knn_3_mean_item_w1_lag_3', 'item_shop_knn_100_mean_item_w1_lag_2', 'item_shop_knn_50_max_item_w1_lag_3', 'item_shop_knn_100_max_fsale_w12_lag_1', 'item_shop_knn_50_median_rev_w1_lag_12', 'item_shop_knn_3_mean_fsale_w1_lag_6', 'item_shop_knn_50_max_price_w1_lag_12', 'item_shop_knn_10_mean_price_w1_lag_3', 'item_shop_knn_100_max_item_w1_lag_12', 'item_shop_knn_3_mean_item_w1_lag_1', 'item_shop_knn_100_max_price_w12_lag_1', 'item_shop_knn_10_median_rev_w1_lag_12', 'item_shop_knn_10_mean_item_w1_lag_3', 'item_shop_knn_50_max_item_w1_lag_6', 'item_shop_knn_50_mean_rev_w12_lag_1', 'item_shop_knn_100_mean_rev_w1_lag_1', 'item_shop_knn_50_mean_price_delta_1', 'item_shop_knn_100_min_price_w1_lag_2', 'item_shop_knn_100_max_rev_w1_lag_3', 'item_shop_knn_50_mean_price_w1_lag_6', 'item_shop_knn_3_median_price_w1_lag_1', 'item_shop_knn_50_mean_item_delta_3', 'item_shop_knn_50_min_item_w1_lag_1', 'item_shop_knn_50_mean_rev_w1_lag_3', 'item_shop_knn_50_max_rev_w12_lag_1', 'item_shop_knn_100_median_fsale_delta_12', 'item_shop_knn_100_max_rev_w1_lag_1', 'item_shop_knn_3_median_item_delta_6', 'item_shop_knn_3_mean_rev_w1_lag_1', 'item_shop_knn_10_min_item_w1_lag_6', 'item_shop_knn_100_mean_price_w1_lag_1', 'item_shop_knn_10_mean_fsale_w12_lag_1', 'item_shop_knn_100_median_item_delta_2', 'item_shop_knn_100_min_fsale_w1_lag_6', 'item_shop_knn_3_min_rev_w12_lag_1', 'item_shop_knn_50_max_rev_w1_lag_1', 'item_shop_knn_3_mean_fsale_delta_2', 'item_shop_knn_100_median_item_w1_lag_12', 'item_shop_knn_50_max_fsale_w12_lag_1', 'item_shop_knn_100_median_fsale_delta_1', 'item_shop_knn_10_max_rev_w1_lag_3', 'item_shop_knn_10_median_rev_w1_lag_2', 'item_shop_knn_10_median_price_delta_3', 'item_shop_knn_3_mean_item_w1_lag_12', 'item_cnt_month_lag_1', 'item_shop_knn_3_median_price_w1_lag_2', 'shop_id', 'item_shop_first_sale', 'item_shop_knn_10_max_item_w1_lag_12', 'item_shop_knn_50_median_rev_w1_lag_2', 'item_shop_knn_100_max_rev_w1_lag_6', 'item_shop_knn_10_max_rev_w12_lag_1', 'item_shop_knn_10_min_rev_w1_lag_12', 'item_id', 'item_shop_knn_3_median_price_w1_lag_12', 'item_shop_knn_50_mean_rev_w1_lag_2', 'item_shop_knn_3_median_item_delta_12', 'item_shop_knn_3_max_item_w12_lag_1', 'item_shop_knn_3_mean_fsale_delta_3', 'item_shop_knn_50_min_price_w1_lag_6', 'item_shop_knn_50_median_fsale_w1_lag_2', 'item_shop_knn_3_min_item_w1_lag_6', 'item_shop_knn_10_mean_item_w1_lag_6', 'item_shop_knn_100_min_rev_w1_lag_12', 'item_shop_knn_50_median_fsale_w1_lag_1', 'item_shop_knn_50_max_rev_w1_lag_2', 'item_shop_knn_50_mean_fsale_delta_1', 'item_shop_knn_10_mean_price_delta_6', 'item_shop_knn_50_median_fsale_delta_6', 'item_shop_knn_10_median_fsale_delta_2', 'item_shop_knn_100_max_price_w1_lag_1', 'item_shop_knn_3_median_fsale_w1_lag_2', 'item_shop_knn_50_mean_item_w12_lag_1', 'item_shop_knn_3_mean_fsale_delta_12', 'item_shop_knn_100_median_fsale_delta_3', 'item_shop_knn_3_mean_fsale_delta_6', 'item_shop_knn_3_mean_price_delta_1', 'item_shop_knn_3_mean_item_delta_12', 'item_shop_knn_3_max_fsale_w1_lag_3', 'item_shop_knn_50_mean_item_delta_1', 'item_shop_knn_10_median_item_w1_lag_6', 'item_shop_knn_50_median_rev_w12_lag_1', 'item_shop_knn_100_mean_price_w1_lag_6', 'item_shop_knn_100_median_item_delta_1', 'item_shop_knn_50_median_price_w1_lag_12', 'item_shop_knn_100_mean_price_w12_lag_1', 'item_shop_knn_50_max_rev_w1_lag_12', 'item_shop_knn_100_mean_fsale_w1_lag_3', 'item_shop_knn_100_mean_item_delta_12', 'item_shop_knn_50_min_item_w1_lag_6', 'item_shop_knn_10_min_price_w1_lag_1', 'item_shop_knn_50_median_price_w1_lag_3', 'item_shop_knn_100_mean_price_delta_2', 'item_shop_knn_100_max_rev_w1_lag_2', ])
matrix.drop(drop_cols, axis=1).to_parquet('../data_processed/%s_filtered_minus_correlated.snappy' % knn_prefix, compression='snappy')
matrix[matrix.date_block_num > 11].to_parquet('../data_processed/%s_filtered_minus_correlated1.snappy' % knn_prefix, compression='snappy')


matrix.to_parquet('../data_processed/%s_features.snappy' % knn_prefix, compression='snappy')
matrix = pd.read_parquet('../data_processed/%s_features.snappy' % knn_prefix)

if knn_prefix == 'item_shop':
    drop_cols = ['item_shop_knn_50_max_fsale_w1_lag_6', 'item_shop_knn_10_mean_rev_delta_12', 'item_shop_knn_3_median_price_w12_lag_1', 'item_shop_knn_3_min_price_w12_lag_1', 'item_shop_knn_3_median_rev_delta_6', 'item_shop_knn_3_max_price_w12_lag_1', 'item_shop_knn_100_mean_rev_delta_3', 'item_shop_knn_50_median_rev_delta_12', 'item_shop_knn_10_median_rev_delta_12', 'item_shop_knn_3_max_rev_w1_lag_2', 'item_shop_knn_100_median_rev_delta_12', 'item_shop_knn_100_mean_rev_delta_2', 'item_shop_knn_10_mean_rev_delta_2', 'item_shop_knn_100_median_rev_delta_3', 'item_shop_knn_10_median_price_w12_lag_1', 'item_shop_knn_3_max_rev_w12_lag_1', 'item_shop_knn_10_median_rev_delta_2', 'item_shop_knn_50_mean_rev_delta_2', 'item_shop_knn_3_median_fsale_w12_lag_1', 'item_avg_item_price', 'item_shop_knn_10_median_fsale_w12_lag_1', 'item_shop_knn_50_mean_rev_delta_3', 'item_shop_knn_100_median_rev_delta_6', 'item_shop_knn_100_max_fsale_w1_lag_2', 'item_shop_knn_100_mean_rev_delta_12', 'item_shop_knn_50_max_fsale_w1_lag_12', 'item_shop_knn_3_median_rev_delta_2', 'item_shop_knn_50_median_rev_delta_6', 'item_shop_knn_50_mean_rev_delta_6', 'item_shop_knn_50_max_fsale_w1_lag_2', 'item_shop_knn_3_median_rev_delta_12', 'item_shop_knn_50_median_rev_delta_2', 'item_shop_knn_100_max_fsale_w1_lag_12', 'item_shop_knn_100_median_rev_delta_2', 'item_shop_knn_50_max_fsale_w1_lag_3', 'item_shop_knn_10_median_rev_delta_6', 'item_shop_knn_50_median_fsale_w12_lag_1', 'item_shop_knn_10_median_rev_delta_3', 'item_shop_knn_100_max_fsale_w1_lag_3', 'item_shop_knn_100_mean_fsale_w1_lag_6', 'item_shop_knn_100_max_fsale_w1_lag_6', 'item_shop_knn_50_mean_rev_delta_12', 'item_shop_knn_100_mean_rev_delta_6', 'item_shop_knn_100_median_fsale_w12_lag_1', 'item_shop_knn_3_max_rev_w1_lag_1', 'date_item_avg_item_price', 'item_shop_knn_3_max_rev_w1_lag_6', 'item_shop_knn_50_median_rev_delta_3', 'item_shop_knn_100_mean_fsale_w1_lag_12', ]
else:
    drop_cols = ['date_item_avg_item_price', 'item_avg_item_price', 'item_knn_100_mean_fsale_w12_lag_1', 'item_knn_100_mean_item_w12_lag_1', 'item_knn_100_mean_price_w12_lag_1', 'item_knn_100_mean_rev_w12_lag_1', 'item_knn_100_median_fsale_w12_lag_1', 'item_knn_100_median_price_w12_lag_1', 'item_knn_100_min_fsale_w12_lag_1', 'item_knn_10_max_fsale_w1_lag_1', 'item_knn_10_max_fsale_w1_lag_12', 'item_knn_10_max_fsale_w1_lag_2', 'item_knn_10_max_fsale_w1_lag_3', 'item_knn_10_max_fsale_w1_lag_6', 'item_knn_10_max_price_w12_lag_1', 'item_knn_10_max_price_w1_lag_1', 'item_knn_10_max_price_w1_lag_12', 'item_knn_10_max_price_w1_lag_2', 'item_knn_10_max_price_w1_lag_3', 'item_knn_10_max_price_w1_lag_6', 'item_knn_10_mean_fsale_w12_lag_1', 'item_knn_10_mean_fsale_w1_lag_1', 'item_knn_10_mean_fsale_w1_lag_12', 'item_knn_10_mean_fsale_w1_lag_2', 'item_knn_10_mean_fsale_w1_lag_3', 'item_knn_10_mean_fsale_w1_lag_6', 'item_knn_10_mean_price_w12_lag_1', 'item_knn_10_mean_price_w1_lag_1', 'item_knn_10_mean_price_w1_lag_12', 'item_knn_10_mean_price_w1_lag_2', 'item_knn_10_mean_price_w1_lag_3', 'item_knn_10_mean_price_w1_lag_6', 'item_knn_10_median_fsale_w12_lag_1', 'item_knn_10_median_fsale_w1_lag_1', 'item_knn_10_median_fsale_w1_lag_12', 'item_knn_10_median_fsale_w1_lag_2', 'item_knn_10_median_fsale_w1_lag_3', 'item_knn_10_median_fsale_w1_lag_6', 'item_knn_10_median_price_w12_lag_1', 'item_knn_10_median_price_w1_lag_1', 'item_knn_10_median_price_w1_lag_12', 'item_knn_10_median_price_w1_lag_2', 'item_knn_10_median_price_w1_lag_3', 'item_knn_10_median_price_w1_lag_6', 'item_knn_10_median_rev_w1_lag_12', 'item_knn_10_min_fsale_w1_lag_1', 'item_knn_10_min_fsale_w1_lag_12', 'item_knn_10_min_fsale_w1_lag_2', 'item_knn_10_min_fsale_w1_lag_3', 'item_knn_10_min_fsale_w1_lag_6', 'item_knn_10_min_price_w12_lag_1', 'item_knn_10_min_price_w1_lag_1', 'item_knn_10_min_price_w1_lag_12', 'item_knn_10_min_price_w1_lag_2', 'item_knn_10_min_price_w1_lag_3', 'item_knn_10_min_price_w1_lag_6', 'item_knn_3_max_fsale_w12_lag_1', 'item_knn_3_max_fsale_w1_lag_1', 'item_knn_3_max_fsale_w1_lag_12', 'item_knn_3_max_fsale_w1_lag_2', 'item_knn_3_max_fsale_w1_lag_3', 'item_knn_3_max_fsale_w1_lag_6', 'item_knn_3_max_price_w12_lag_1', 'item_knn_3_max_price_w1_lag_1', 'item_knn_3_max_price_w1_lag_12', 'item_knn_3_max_price_w1_lag_2', 'item_knn_3_max_price_w1_lag_3', 'item_knn_3_max_price_w1_lag_6', 'item_knn_3_max_rev_w12_lag_1', 'item_knn_3_max_rev_w1_lag_12', 'item_knn_3_median_fsale_w12_lag_1', 'item_knn_3_median_fsale_w1_lag_1', 'item_knn_3_median_fsale_w1_lag_12', 'item_knn_3_median_fsale_w1_lag_2', 'item_knn_3_median_fsale_w1_lag_3', 'item_knn_3_median_fsale_w1_lag_6', 'item_knn_3_median_price_w12_lag_1', 'item_knn_3_median_price_w1_lag_1', 'item_knn_3_median_price_w1_lag_12', 'item_knn_3_median_price_w1_lag_2', 'item_knn_3_median_price_w1_lag_3', 'item_knn_3_median_price_w1_lag_6', 'item_knn_3_min_fsale_w1_lag_1', 'item_knn_3_min_fsale_w1_lag_12', 'item_knn_3_min_fsale_w1_lag_2', 'item_knn_3_min_fsale_w1_lag_3', 'item_knn_3_min_fsale_w1_lag_6', 'item_knn_3_min_price_w12_lag_1', 'item_knn_3_min_price_w1_lag_1', 'item_knn_3_min_price_w1_lag_12', 'item_knn_3_min_price_w1_lag_2', 'item_knn_3_min_price_w1_lag_3', 'item_knn_3_min_price_w1_lag_6', 'item_knn_50_max_price_w12_lag_1', 'item_knn_50_mean_fsale_w12_lag_1', 'item_knn_50_mean_fsale_w1_lag_1', 'item_knn_50_mean_fsale_w1_lag_12', 'item_knn_50_mean_fsale_w1_lag_2', 'item_knn_50_mean_fsale_w1_lag_3', 'item_knn_50_mean_fsale_w1_lag_6', 'item_knn_50_mean_price_w12_lag_1', 'item_knn_50_mean_price_w1_lag_1', 'item_knn_50_mean_price_w1_lag_12', 'item_knn_50_mean_price_w1_lag_2', 'item_knn_50_mean_price_w1_lag_3', 'item_knn_50_mean_price_w1_lag_6', 'item_knn_50_mean_rev_w1_lag_12', 'item_knn_50_median_fsale_w12_lag_1', 'item_knn_50_median_fsale_w1_lag_1', 'item_knn_50_median_fsale_w1_lag_12', 'item_knn_50_median_fsale_w1_lag_2', 'item_knn_50_median_fsale_w1_lag_3', 'item_knn_50_median_fsale_w1_lag_6', 'item_knn_50_median_price_w12_lag_1', 'item_knn_50_median_price_w1_lag_1', 'item_knn_50_median_price_w1_lag_12', 'item_knn_50_median_price_w1_lag_2', 'item_knn_50_median_price_w1_lag_3', 'item_knn_50_median_price_w1_lag_6', 'item_knn_50_median_rev_w1_lag_1', 'item_knn_50_median_rev_w1_lag_12', 'item_knn_50_median_rev_w1_lag_2', 'item_knn_50_median_rev_w1_lag_3', 'item_knn_50_median_rev_w1_lag_6', 'item_knn_50_min_fsale_w12_lag_1', ]
matrix.drop([c for c in drop_cols if c in matrix.columns], axis=1, inplace=True)
# matrix.to_parquet('../data_processed/%s_minus_correlated.snappy' % knn_prefix, compression='snappy')
matrix[matrix.date_block_num > 11].to_parquet('../data_processed/%s_minus_correlated1.snappy' % knn_prefix, compression='snappy')

for c in tqdm(matrix.columns):
    if matrix[c].dtype == 'float16':
        matrix[c] = matrix[c].astype(np.float32)

matrix = matrix[matrix.date_block_num > 11]
gc.collect()

keep_cols = set()
for knn in tqdm([3, 10, 50, 100]):
    for lag in [2, 3, 6, 12]:
        for prefix in ['item']: # 'item_shop',
            for fn in ['mean', 'median']:
                for var in ['item', 'fsale', 'price', 'rev']:
                    col_name = '%s_knn_%s_%s_%s_delta_%s' % (prefix, knn, fn, var, lag)
                    if col_name in drop_cols:
                        continue
                    c2 = '%s_knn_%s_%s_%s_w%s_lag_%s' % (prefix, knn, fn, var, 12, 1)
                    c1 = '%s_knn_%s_%s_%s_w%s_lag_%s' % (prefix, knn, fn, var,  1, lag)
                    keep_cols.add(c1)
                    keep_cols.add(c2)
drop_cols1 = list(set(drop_cols) - keep_cols)
matrix.drop([c for c in matrix.columns if c in drop_cols1], axis=1, inplace=True)
gc.collect()
for knn in tqdm([3, 10, 50, 100]):
    for lag in [2, 3, 6, 12]:
        for prefix in ['item_shop']: # 'item',
            for fn in ['mean', 'median']:
                for var in ['item', 'fsale', 'price', 'rev']:
                    col_name = '%s_knn_%s_%s_%s_delta_%s' % (prefix, knn, fn, var, lag)
                    if col_name in drop_cols:
                        continue
                    c2 = '%s_knn_%s_%s_%s_w%s_lag_%s' % (prefix, knn, fn, var, 12, 1)
                    c1 = '%s_knn_%s_%s_%s_w%s_lag_%s' % (prefix, knn, fn, var,  1, lag)
                    matrix[col_name] = np.log1p(matrix[c1] / (matrix[c2] + 1e-6))
                    gc.collect()
matrix.drop([c for c in matrix.columns if c in drop_cols], axis=1, inplace=True)



matrix.drop(drop_cols, axis=1, inplace=True)
matrix.to_parquet('../data_processed/item_shop_minus_correlated.snappy', compression='snappy')
matrix = pd.read_parquet('../data_processed/item_shop_minus_correlated.snappy')

res = {}
index = np.random.choice(range(matrix.shape[0]), 10000, False)
v0 = matrix['item_cnt_month'].iloc[index].values
for c in matrix.columns:
    if c == 'item_cnt_month':
        continue
    v = matrix[c].iloc[index].values
    mask = (~np.isnan(v0)) & (~np.isnan(v))
    res[c] = np.corrcoef(v0[mask], v[mask])[0, 1]

from collections import Counter
Counter(res).most_common()

for knn_prefix in ['item_shop']:
    matrix_path = '../data_processed/%s_filtered_minus_correlated1.snappy' % knn_prefix
    matrix = pd.read_parquet(matrix_path, columns=['date_block_num'])
    train_index = matrix[(matrix.date_block_num < 33) & (matrix.date_block_num > 11)].index
    valid_index = matrix[matrix.date_block_num == 33].index
    del matrix; gc.collect()

    init_drop = []
    drop_cols, elimination_path, model, random_string = utils.recursive_features_eliminate(path_to_df=matrix_path,
                                     target='item_cnt_month',
                                     max_iter=100,
                                     always_drop=init_drop + ['ID', 'date_item_shop_avg_item_price', 'date_item_shop_revenue', 'date_shop_revenue', 'delta_revenue', 'shop_avg_revenue', ],
                                     keep_cols=['date_block_num', 'item_id', 'shop_id', 'item_category_id', 'item_0', 'item_1', 'item_2', 'item_3', 'item_4', 'shop_0', 'shop_1', 'shop_2', 'cat_0', 'cat_1', 'cat_2'],
                                     tstat_thresh=-1, # tstat = (mean - base_model) / std threshold to drop features from the model
                                     max_drop_ratio=0.2, # fraction of features to drop from the non-eliminated features,
                                     min_drop=50,
                                     train_index=train_index,
                                     valid_index=valid_index,
                                     reload_data=True, # if memory is bottleneck deleting and re-reading data on each iteration
                                     model_type='rf', # model to train and use for elimination
                                     clip_down=0,
                                     clip_up=20,
                                     max_xgb_features=200, # memory issues if >200 columns
                                     num_permutations=5, # number of permutation for feature to get stats on baseline score drop
                                     n_estimators=10,
                                     max_depth=8,
                                     max_samples=0.8,
                                     seed=123,
                                     n_jobs=8,
                                     learning_rate=0.2,
                                     )

    pd.Series(elimination_path).to_pickle('../data_tmp/%s_path02.pkl' % knn_prefix)

always_drop = ['ID', 'date_item_shop_avg_item_price', 'date_item_shop_revenue', 'date_shop_revenue', 'delta_revenue', 'shop_avg_revenue']
paths = {
    'item': ['../data_tmp/7a1f425d_27.pkl'],
    'item_shop': ['../data_tmp/853eddfa_61.pkl']
}

# pd.Series(always_drop)
bad_features = []
bad_featues = [pd.read_csv('../data_processed/bad_features.csv').iloc[:, 1]]
for knn_prefix in paths:
    for pkl in paths[knn_prefix]:
        bad_features.append(pd.read_pickle(pkl))
bad_featues = pd.concat(bad_features, axis=0).sort_values().reset_index(drop=True)

# bad_featues.to_csv('../data_processed/bad_features1.csv')

bad_features = list(pd.read_csv('../data_processed/bad_features1.csv').iloc[:, 1].values)
matrix = []
existing_cols = set()
for knn_prefix in ['item_shop', 'item']:
    matrix_path = '../data_processed/%s_filtered_minus_correlated.snappy' % knn_prefix
    _ = pd.read_parquet(matrix_path)
    _.drop([c for c in _.columns if c in (set(bad_features) | existing_cols)], axis=1, inplace=True)
    existing_cols |= set(_.columns)
    matrix.append(_)
    del _
    gc.collect()
matrix = pd.concat(matrix, axis=1)

data = matrix[matrix.date_block_num > 11]
del matrix
gc.collect()
data.to_parquet('../data_processed/2020-07-02.snappy', compression='snappy')

data = pd.read_parquet('../data_processed/2020-07-02.snappy')
drop_cols = ['ID']
drop_cols += ['date_item_shop_avg_item_price', 'date_item_shop_revenue', 'date_shop_revenue', 'item_shop_first_sale', 'shop_avg_revenue', ]
data.drop(drop_cols, axis=1, inplace=True)
data.info()

X_train = data[data.date_block_num < 33].drop(['item_cnt_month', 'date_block_num', ], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month', 'date_block_num', ], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month', 'date_block_num', ], axis=1)

del data
gc.collect()


X_train.info()

# [62]	validation_0-rmse:0.79811	validation_1-rmse:0.89230

ts = time.time()
model = XGBRegressor(
    max_depth=8,
    n_estimators=67,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.02,
    seed=42)

model.fit(
    X_train,
    Y_train,
    eval_metric="rmse",
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
    verbose=True,
    early_stopping_rounds = 10)
time.time() - ts


terr = mean_squared_error(Y_train.values.astype('float32'), model.predict(X_train).clip(0, 20))
verr = mean_squared_error(Y_valid.values.astype('float32'), model.predict(X_valid).clip(0, 20))
print('train-rmse=%.4f valid-rmse=%.4f' % (terr, verr))

Y_pred = model.predict(X_valid).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)

# from collections import Counter
# from sklearn.metrics import confusion_matrix
# Y_pred = model.predict(X_valid).clip(0, 20)
# sorted(Counter(Y_pred.round(0)).items())
# sorted(Counter(Y_valid).items())
# cm = confusion_matrix(Y_valid.values.astype('int8'), Y_pred.round(0).astype('int8'))
# pd.DataFrame(cm).to_csv('../data/errors.xgb.best.csv')

# Y_test = model.predict(X_test).clip(0, 20)
test = pd.read_csv('../data/test.csv')
submission = pd.DataFrame({
    "ID": test.index,
    "item_cnt_month": Y_test
})
submission.to_csv('../submission/xgb_submission_2020-07-03_03.csv', index=False)
pd.Series(Y_pred).to_csv('../submission/xgb_submission_2020-07-03_03_valid.csv', index=False)








































matrix.to_parquet('../data_processed/item_shop_thick0.snappy', compression='snappy')

matrix = utils.reduce_mem_usage(matrix)
matrix = matrix[matrix.date_block_num > 11]
matrix = matrix.merge()

for knn_prefix in ['item', 'item_shop']:
    for lag in tqdm([1, 2, 3, 6, 12]):
        fn = 'knn_%s_win1_lag%s.snappy' % (knn_prefix, lag)
        df = pd.read_parquet('../data_processed/' + fn)
        drop_cols = [c for c in df.columns if '_std_' in c or 'min_fsale' in c]
        df.drop(drop_cols, axis=1, inplace=True)
        df.columns = [c + '_lag_%s' % lag for c in df.columns]
        df = utils.reduce_mem_usage(df.loc[df.index.isin(matrix.index)])
        matrix = pd.concat([matrix, df], axis=1)
        # print(matrix.info())

for c in tqdm(matrix.columns):
    if matrix[c].dtype == 'float16':
        matrix[c] = matrix[c].astype(np.float32)
matrix.to_parquet('../data_processed/many_features.snappy', compression='snappy')
matrix = pd.read_parquet('../data_processed/many_features.snappy')

for c in matrix.columns:
    if 'fsale' in c:
        matrix[c] = matrix[c] - matrix['date_block_num']


train_index = matrix[(matrix.date_block_num < 33) & (matrix.date_block_num > 11)].index
valid_index = matrix[matrix.date_block_num == 33].index
del matrix; gc.collect()

drop_cols, elimination_path, model, random_string = utils.recursive_features_eliminate(path_to_df='../data_processed/many_shop_features.snappy',
                                 target='item_cnt_month',
                                 max_iter=20,
                                 drop_cols=['ID'],
                                 keep_cols=['date_block_num', 'item_id', 'shop_id', 'item_category_id', 'item_0', 'item_1', 'item_2', 'item_3', 'item_4', 'shop_0', 'shop_1', 'shop_2', 'cat_0', 'cat_1', 'cat_2'],
                                 tstat_thresh=-1, # tstat = (mean - base_model) / std threshold to drop features from the model
                                 max_drop_ratio=0.2, # fraction of features to drop from the non-eliminated features,
                                 train_index=train_index,
                                 valid_index=valid_index,
                                 reload_data=False, # if memory is bottleneck deleting and re-reading data on each iteration
                                 model_type='rf', # model to train and use for elimination
                                 clip_down=0,
                                 clip_up=20,
                                 max_xgb_features=200, # memory issues if >200 columns
                                 num_permutations=5, # number of permutation for feature to get stats on baseline score drop
                                 n_estimators=10,
                                 max_depth=8,
                                 max_samples=0.8,
                                 seed=123,
                                 n_jobs=8,
                                 learning_rate=0.2,
                                 )


# cs = ['item_shop_knn_50_min_item_lag_1', 'item_shop_knn_100_min_item_lag_12', 'item_knn_50_min_item_lag_12', 'item_shop_knn_50_min_item_lag_2', 'item_shop_knn_50_min_item_lag_3', 'item_shop_knn_50_min_item_lag_6', ]
# pd.concat([matrix[cs].std(), matrix[cs].min(), matrix[cs].max()], axis=1)

drop_cols = ['ID'] # 'shop_name', 'item_name', 'item_category_name'
drop_cols += ['item_knn_50_min_item_lag_1', 'item_knn_100_min_item_lag_1', 'item_knn_500_min_item_lag_1', 'item_knn_50_min_item_lag_2', 'item_knn_100_min_item_lag_2', 'item_knn_500_min_item_lag_2', 'item_knn_50_min_item_lag_3', 'item_knn_100_min_item_lag_3', 'item_knn_500_min_item_lag_3', 'item_knn_50_min_item_lag_6', 'item_knn_100_min_item_lag_6', 'item_knn_500_min_item_lag_6', 'item_knn_100_min_item_lag_12', 'item_knn_500_min_item_lag_12', 'item_shop_knn_500_min_item_lag_1', 'item_shop_knn_500_min_item_lag_2', 'item_shop_knn_500_min_item_lag_3', 'item_shop_knn_500_min_item_lag_6', 'item_shop_knn_500_min_item_lag_12', ]
drop_cols += ['item_knn_10_mean_fsale_lag_2']
drop_cols += ['item_shop_knn_10_median_fsale_lag_3', 'item_knn_50_median_item_lag_1', 'item_shop_knn_50_mean_price_delta_2', 'item_knn_500_min_price_lag_2', 'item_shop_knn_100_mean_price_lag_12', 'item_knn_50_median_item_lag_12', 'item_shop_knn_100_max_fsale_lag_6', 'item_knn_10_min_item_lag_2', 'item_shop_knn_10_median_item_lag_1', 'item_shop_knn_50_median_price_lag_1', 'item_shop_knn_500_mean_item_lag_2', 'item_shop_knn_500_median_fsale_lag_12', 'item_knn_50_median_item_lag_6', 'item_shop_knn_10_mean_price_delta_2', 'item_shop_knn_500_max_item_lag_2', 'item_shop_knn_100_max_item_lag_1', 'item_knn_100_mean_item_lag_1', 'item_shop_knn_10_min_price_lag_2', 'item_shop_knn_100_mean_item_lag_2', 'item_knn_100_min_price_lag_3', 'item_shop_knn_100_mean_item_lag_6', 'item_knn_500_max_fsale_lag_1', 'item_shop_knn_100_min_price_lag_12', 'item_shop_knn_500_mean_fsale_lag_1', 'item_knn_500_median_price_delta_6', 'item_shop_knn_10_median_price_delta_6', 'item_shop_knn_50_mean_price_lag_1', 'item_knn_10_max_item_lag_3', 'item_knn_10_median_item_lag_3', 'item_knn_50_mean_item_lag_1', 'item_shop_knn_500_mean_fsale_lag_12', 'item_shop_knn_10_median_price_lag_1', 'item_knn_500_mean_price_lag_6', 'item_shop_knn_500_max_price_lag_1', 'item_knn_500_median_price_lag_3', 'item_knn_500_max_item_lag_3', 'item_knn_50_max_fsale_lag_12', 'item_shop_knn_10_median_price_lag_3', 'item_knn_100_mean_fsale_lag_12', 'item_shop_knn_50_median_item_lag_1', 'item_shop_knn_50_max_price_lag_2', 'item_knn_50_min_price_lag_12', 'item_knn_100_max_price_lag_6', 'item_knn_500_max_price_lag_1', 'item_knn_500_mean_price_lag_3', 'item_shop_knn_500_mean_price_lag_1', 'item_knn_100_median_fsale_lag_6', 'item_knn_100_mean_fsale_lag_6', 'item_shop_knn_10_max_price_lag_12', 'item_shop_knn_500_max_item_lag_12', 'item_shop_knn_500_max_price_lag_2', ]
drop_cols += ['item_knn_100_max_fsale_lag_1', 'item_knn_10_mean_price_lag_2', 'item_knn_100_max_price_lag_2', 'item_knn_500_mean_fsale_lag_3', 'item_knn_50_max_fsale_lag_6', 'item_knn_100_mean_item_lag_12', 'item_knn_100_max_item_lag_12', 'item_knn_500_median_item_lag_12', 'item_shop_knn_50_max_price_lag_1', 'item_shop_knn_100_median_item_lag_2', 'item_shop_knn_100_median_item_lag_3', 'item_shop_knn_100_mean_price_lag_3', 'item_shop_knn_10_min_item_lag_12', 'item_shop_knn_50_max_price_lag_12', 'item_knn_10_mean_price_delta_12', 'item_knn_100_median_price_delta_6', 'item_knn_500_mean_price_delta_2', 'item_shop_knn_500_mean_price_delta_2']
drop_cols += ['item_knn_10_min_item_lag_1', 'item_knn_100_median_item_lag_1', 'item_knn_500_mean_fsale_lag_1', 'item_knn_10_max_item_lag_2', 'item_knn_50_median_item_lag_3', 'item_knn_100_min_price_lag_12', 'item_shop_knn_100_max_price_lag_2', 'item_shop_knn_500_median_item_lag_12', 'item_shop_knn_10_min_price_lag_12', 'item_shop_knn_100_median_price_delta_2', 'item_shop_knn_500_mean_price_delta_12']
drop_cols += ['item_knn_10_median_item_lag_1', 'item_knn_50_median_item_lag_2', 'item_knn_500_median_item_lag_2', 'item_knn_10_mean_item_lag_3', 'item_knn_500_mean_item_lag_3', 'item_knn_50_max_fsale_lag_3', 'item_knn_10_max_item_lag_6', 'item_knn_500_mean_fsale_lag_12', 'item_knn_100_mean_price_lag_12', 'item_knn_500_min_price_lag_12', 'item_shop_knn_100_median_price_lag_1', 'item_shop_knn_50_max_fsale_lag_2', 'item_shop_knn_100_mean_item_lag_3', 'item_shop_knn_10_median_price_lag_6', 'item_shop_knn_50_median_fsale_lag_12', 'item_knn_10_mean_price_delta_6']
drop_cols += ['item_knn_10_median_item_lag_2', 'item_knn_50_mean_item_lag_2', 'item_knn_50_max_fsale_lag_2', 'item_knn_500_min_price_lag_3', 'item_knn_10_mean_item_lag_6', 'item_knn_10_min_item_lag_6', 'item_knn_10_min_item_lag_12', 'item_knn_100_median_item_lag_12', 'item_knn_10_mean_fsale_lag_12', 'item_knn_100_median_fsale_lag_12', 'item_knn_100_median_price_lag_12', 'item_shop_knn_50_min_item_lag_12', 'item_shop_knn_10_mean_fsale_lag_12', 'item_shop_knn_50_median_price_lag_12', 'item_shop_knn_10_mean_price_delta_12', 'item_knn_100_mean_price_delta_2']
drop_cols += ['item_knn_100_max_fsale_lag_12','item_knn_100_max_fsale_lag_6','item_knn_100_max_price_lag_1','item_knn_100_mean_price_lag_1','item_knn_100_median_fsale_lag_2','item_knn_100_median_item_lag_6','item_knn_100_median_price_lag_1','item_knn_10_max_item_lag_12','item_knn_10_mean_item_lag_12','item_knn_10_mean_item_lag_2','item_knn_10_mean_price_delta_2','item_knn_10_median_item_lag_12','item_knn_10_min_item_lag_3','item_knn_500_max_fsale_lag_6','item_knn_500_mean_item_lag_12','item_knn_500_median_fsale_lag_12','item_knn_500_median_price_delta_2','item_knn_500_median_price_lag_2','item_knn_50_max_price_lag_12','item_knn_50_max_price_lag_6','item_knn_50_mean_item_lag_12','item_shop_knn_100_max_price_lag_3','item_shop_knn_100_median_fsale_lag_12','item_shop_knn_100_median_item_lag_6','item_shop_knn_100_median_price_lag_3','item_shop_knn_10_max_fsale_lag_1','item_shop_knn_10_mean_fsale_lag_6','item_shop_knn_10_median_item_lag_6','item_shop_knn_500_mean_price_lag_12','item_shop_knn_500_median_fsale_lag_6','item_shop_knn_500_median_item_lag_6','item_shop_knn_500_median_price_delta_12','item_shop_knn_50_max_price_lag_3','item_shop_knn_50_mean_fsale_lag_1','item_shop_knn_50_mean_fsale_lag_6','item_shop_knn_50_median_fsale_lag_6','item_shop_knn_50_median_item_lag_3',]
drop_cols += ['item_knn_100_median_fsale_lag_1', 'item_knn_50_max_price_lag_1', 'item_knn_500_median_price_lag_1', 'item_knn_100_mean_item_lag_2', 'item_knn_100_median_item_lag_2', 'item_knn_50_min_price_lag_2', 'item_knn_100_median_item_lag_3', 'item_knn_500_median_item_lag_3', 'item_knn_100_median_fsale_lag_3', 'item_knn_10_median_item_lag_6', 'item_knn_50_mean_item_lag_6', 'item_knn_100_mean_item_lag_6', 'item_knn_10_mean_fsale_lag_6', 'item_knn_500_mean_fsale_lag_6', 'item_knn_500_min_price_lag_6', 'item_knn_500_max_item_lag_12', 'item_shop_knn_100_median_item_lag_1', 'item_shop_knn_10_max_price_lag_1', 'item_shop_knn_100_mean_price_lag_1', 'item_shop_knn_500_median_item_lag_2', 'item_shop_knn_10_max_fsale_lag_2', 'item_shop_knn_100_median_fsale_lag_2', 'item_shop_knn_100_mean_price_lag_2', 'item_shop_knn_100_median_price_lag_2', 'item_shop_knn_100_min_price_lag_2', 'item_shop_knn_100_median_fsale_lag_3', 'item_shop_knn_500_max_item_lag_6', 'item_shop_knn_10_median_fsale_lag_6', 'item_shop_knn_50_max_item_lag_12', 'item_shop_knn_10_mean_price_lag_12', 'item_shop_knn_50_min_price_lag_12', 'item_knn_50_mean_price_delta_6', 'item_shop_knn_100_mean_price_delta_2', ]
drop_cols += ['item_knn_500_max_fsale_lag_3', 'item_knn_500_mean_fsale_lag_2', 'item_knn_50_min_price_lag_6', ]
drop_cols += ['item_knn_100_mean_fsale_lag_1', 'item_knn_100_min_price_lag_6', 'item_knn_10_mean_fsale_lag_1', 'item_knn_10_mean_fsale_lag_3', 'item_knn_10_mean_price_lag_12', 'item_knn_500_max_fsale_lag_2', 'item_knn_500_max_price_lag_12', 'item_knn_500_max_price_lag_3', 'item_knn_500_mean_price_lag_1', 'item_knn_500_median_fsale_lag_3', 'item_knn_500_median_fsale_lag_6', 'item_knn_500_median_price_lag_6', 'item_knn_50_max_price_lag_3', 'item_shop_knn_100_max_fsale_lag_2', 'item_shop_knn_100_max_item_lag_6', 'item_shop_knn_100_max_price_lag_1', 'item_shop_knn_100_mean_price_lag_6', 'item_shop_knn_100_median_fsale_lag_1', 'item_shop_knn_100_median_price_lag_12', 'item_shop_knn_10_max_fsale_lag_3', 'item_shop_knn_10_max_item_lag_12', 'item_shop_knn_10_max_price_lag_3', 'item_shop_knn_10_mean_fsale_lag_2', 'item_shop_knn_10_mean_item_lag_2', 'item_shop_knn_10_median_fsale_lag_12', 'item_shop_knn_10_min_item_lag_1', 'item_shop_knn_10_min_item_lag_2', 'item_shop_knn_500_mean_price_delta_6', 'item_shop_knn_500_median_price_lag_12', 'item_shop_knn_50_mean_fsale_lag_12', 'item_shop_knn_50_mean_price_lag_2', 'item_shop_knn_50_median_fsale_lag_1', 'item_shop_knn_50_median_price_delta_2', 'item_shop_knn_50_median_price_lag_6', 'item_shop_knn_50_min_price_lag_1', ]
drop_cols += ['item_knn_100_max_price_lag_3', 'item_knn_100_mean_price_delta_6', 'item_knn_100_mean_price_lag_3', 'item_knn_100_median_price_lag_3', 'item_knn_100_min_price_lag_1', 'item_knn_500_max_item_lag_1', 'item_knn_500_max_price_lag_2', 'item_knn_500_mean_item_lag_6', 'item_knn_500_mean_price_delta_12', 'item_knn_500_mean_price_lag_12', 'item_knn_500_median_price_lag_12', 'item_knn_50_max_fsale_lag_1', 'item_knn_50_max_item_lag_12', 'item_knn_50_max_item_lag_6', 'item_knn_50_max_price_lag_2', 'item_shop_knn_100_max_fsale_lag_1', 'item_shop_knn_100_max_item_lag_12', 'item_shop_knn_10_max_fsale_lag_12', 'item_shop_knn_10_mean_fsale_lag_1', 'item_shop_knn_10_mean_price_delta_6', 'item_shop_knn_10_mean_price_lag_1', 'item_shop_knn_10_mean_price_lag_2', 'item_shop_knn_10_median_fsale_lag_1', 'item_shop_knn_10_min_price_lag_1', 'item_shop_knn_10_min_price_lag_3', 'item_shop_knn_500_max_price_lag_6', 'item_shop_knn_500_median_fsale_lag_1', 'item_shop_knn_500_median_price_lag_6', 'item_shop_knn_50_mean_fsale_lag_2', 'item_shop_knn_50_mean_fsale_lag_3', 'item_shop_knn_50_mean_price_delta_12', 'item_shop_knn_50_mean_price_lag_12', 'item_shop_knn_50_mean_price_lag_6', 'item_shop_knn_50_median_fsale_lag_2', 'item_shop_knn_50_median_fsale_lag_3', ]
drop_cols += ['item_knn_100_max_fsale_lag_3', 'item_knn_100_max_price_lag_12', 'item_knn_100_mean_fsale_lag_3', 'item_knn_100_mean_price_lag_2', 'item_knn_100_mean_price_lag_6', 'item_knn_100_median_price_lag_6', 'item_knn_500_mean_price_delta_6', 'item_knn_500_median_item_lag_1', 'item_knn_500_median_item_lag_6', 'item_knn_50_min_price_lag_1', 'item_shop_knn_100_max_price_lag_12', 'item_shop_knn_100_median_price_lag_6', 'item_shop_knn_10_mean_price_lag_3', 'item_shop_knn_10_mean_price_lag_6', 'item_shop_knn_10_median_fsale_lag_2', 'item_shop_knn_500_min_price_lag_1', 'item_shop_knn_50_max_item_lag_6', 'item_shop_knn_50_median_item_lag_2', 'item_shop_knn_50_median_price_delta_12', 'item_shop_knn_50_min_price_lag_2', ]
drop_cols += ['item_knn_500_max_item_lag_2', 'item_knn_500_max_item_lag_6', 'item_knn_500_median_fsale_lag_2', 'item_knn_50_max_item_lag_2', 'item_shop_knn_100_median_fsale_lag_6', 'item_shop_knn_10_median_price_delta_2', 'item_shop_knn_10_median_price_lag_12', 'item_shop_knn_500_median_item_lag_3', ]
drop_cols += ['item_knn_100_median_price_delta_2', 'item_knn_100_min_price_lag_2', 'item_knn_500_mean_item_lag_1', 'item_knn_500_mean_price_lag_2', 'item_shop_knn_100_min_price_lag_6', 'item_shop_knn_10_max_fsale_lag_6', 'item_shop_knn_10_min_price_lag_6', 'item_shop_knn_500_max_price_lag_12', 'item_shop_knn_500_mean_item_lag_6', 'item_shop_knn_50_max_price_lag_6', 'item_shop_knn_50_mean_price_lag_3', 'item_shop_knn_50_median_item_lag_12', 'item_shop_knn_50_median_item_lag_6', 'item_shop_knn_50_median_price_lag_2', ]
drop_cols += ['item_knn_10_mean_price_lag_1', 'item_knn_100_max_fsale_lag_2', 'item_shop_knn_500_median_price_lag_1', 'item_shop_knn_10_median_item_lag_3', 'item_shop_knn_500_median_price_delta_2', 'item_shop_knn_100_min_price_lag_3']
drop_cols += ['item_knn_100_mean_fsale_lag_2', 'item_knn_100_mean_item_lag_3', 'item_knn_10_mean_price_lag_3', 'item_knn_50_max_item_lag_3', 'item_shop_knn_10_median_item_lag_12', 'item_shop_knn_10_median_price_lag_2', ]
drop_cols += ['item_knn_50_mean_item_lag_3', 'item_shop_knn_100_mean_price_delta_12', 'item_shop_knn_10_mean_fsale_lag_3', 'item_shop_knn_10_mean_item_lag_12', 'item_shop_knn_10_median_item_lag_2', 'item_shop_knn_50_mean_item_lag_6', 'item_shop_knn_50_median_price_delta_6', ]
drop_cols += ['item_shop_knn_50_min_price_lag_6']
drop_cols += ['item_shop_knn_10_min_item_lag_3', 'item_shop_knn_500_max_item_lag_1', 'item_shop_knn_500_median_price_delta_6', 'item_shop_knn_500_min_price_lag_6', ]
# drop_cols += [c for c in matrix.columns if "_std_" in c or '_median_' in c]
# drop_cols += ['item_knn_500_min_price_lag_12', 'item_knn_10_mean_item_lag_12', 'item_knn_50_mean_price_delta_12', 'item_shop_knn_500_max_item_lag_1', 'item_knn_100_min_price_lag_1', 'item_shop_knn_50_min_fsale_lag_1', 'item_shop_knn_10_mean_fsale_lag_1', 'item_shop_knn_50_mean_price_delta_6', 'item_shop_knn_100_mean_fsale_lag_1', 'item_shop_knn_10_min_price_lag_6', 'item_shop_knn_100_min_fsale_lag_1', 'item_shop_knn_10_mean_price_delta_12', 'item_shop_knn_500_min_item_lag_1', 'item_shop_knn_100_min_item_lag_1', 'item_shop_knn_500_min_fsale_lag_1', 'item_shop_knn_50_min_item_lag_1', 'item_knn_10_min_price_lag_12', 'item_shop_knn_500_min_fsale_lag_12', 'item_shop_knn_100_min_fsale_lag_12', 'item_shop_knn_100_min_item_lag_6', 'item_shop_knn_500_min_item_lag_12', 'item_shop_knn_100_min_item_lag_12', 'item_shop_knn_100_min_fsale_lag_6', 'item_shop_knn_50_min_item_lag_12', 'item_shop_knn_500_min_fsale_lag_6', 'item_shop_knn_500_min_item_lag_6', ]
matrix.drop(drop_cols, axis=1, inplace=True)
# drop_cols += ['item_knn_500_avg_sale', 'item_knn_100_med_price', 'item_knn_100_avg_sale', 'item_knn_1000_max_sale', 'item_knn_1000_max_price', 'item_knn_500_std_price', 'item_knn_100_std_price', 'item_knn_500_med_sale', 'item_knn_100_max_sale', 'item_knn_100_avg_price', 'item_knn_1000_std_sale', 'item_knn_1000_avg_sale', 'item_knn_1000_avg_item', 'item_knn_100_max_price', 'item_knn_1000_std_item', 'item_knn_500_std_sale', 'item_knn_1000_med_item', 'item_knn_1000_med_sale', 'item_knn_1000_med_price', 'date_item_avg_item_price', 'item_knn_1000_std_price', 'item_knn_500_std_item', 'item_knn_1000_max_item', 'item_knn_100_std_sale', ]
target = 'item_cnt_month'
# matrix = matrix.iloc[np.where(matrix.date_block_num > 11)[0]]
data = matrix#.drop(drop_cols, axis=1)

gc.collect()
del matrix;
data.fillna(0, inplace=True)

# features = data.drop([target], axis=1).columns
X_train = data[data.date_block_num < 33].drop([target], axis=1)
Y_train = data[data.date_block_num < 33][target]
X_valid = data[data.date_block_num == 33].drop([target], axis=1)
Y_valid = data[data.date_block_num == 33][target]
# X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1).values
del data; gc.collect()
gc.collect()
gc.collect()

#################################################
model, err, importances, features, gain_counter, thresh_counter, thresh_counter2, best_splits, best_splits2 = utils_rf.feature_statistics(X_train, Y_train, X_valid, Y_valid, n_estimators=20, clip_low=0, clip_high=20)
pd.Series(importances).sort_values(ascending=False).to_csv('../data_tmp/tmp_rf.csv')

# imps = pd.Series(importances).sort_values(ascending=False)
# # imps = score_drop['mean'].sort_values(ascending=False)
# cs = imps.index[:50]
# corrs = np.corrcoef(X_train[cs].values.astype(np.float64).T)
# rows, cols = np.where((corrs > 0.9))
# features = X_train.columns
# pairs = [(features[r], features[c], np.round(corrs[r, c], 2)) for r, c in zip(rows, cols) if r < c]

base_score = mean_squared_error(Y_valid, model.predict(X_valid).clip(0, 20))
score_drop = {}
# features = X_train.columns
for ci, c in tqdm(enumerate(features)):
    temp_df = X_valid.copy()

    # if temp_df[c].dtypes.name != 'category':
    cur_score = []
    for _ in range(5):
        temp_df.iloc[:, ci] = np.random.permutation(temp_df.iloc[:, ci])
        val_pred = model.predict(temp_df) # [features]
        cur_score.append( mean_squared_error(Y_valid, val_pred.clip(0, 20)) )
    _m = np.array(cur_score).mean()
    _s = np.array(cur_score).std()
    cur_score = (_m - base_score, (_m - base_score) / (_s + 1e-10))
    # If our current rmse score is less than base score
    # it means that feature most probably is a bad one
    # and our model is learning on noise
    score_drop[c] = cur_score
    # print(c, score_drop[c])
# sorted(score_drop.items(), key=lambda x: x[1][0], reverse=True)
score_drop = pd.DataFrame({f: {'mean': a, 'tstat': b} for f, (a, b) in score_drop.items()}).T.sort_values('tstat')
score_drop.to_csv('../data_tmp/tmp1_rf.csv')
# pd.Series({f: v for f, v in zip(features, model.feature_importances_)}).sort_values(ascending=False).to_csv('../data_tmp/tmp_rf.csv')
###############################################
thresh = min(-1, np.quantile(score_drop.tstat.values, 0.2))
score_drop[score_drop.tstat <= thresh].index.values


# X_train.info()
ts = time.time()
model = XGBRegressor(
    max_depth=8,
    n_estimators=40,
    min_child_weight=300, 
    colsample_bytree=0.5,
    subsample=0.8,
    eta=0.2,
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds=5)
time.time() - ts

model, err, importances, features, gain_counter, thresh_counter, thresh_counter2, best_splits, best_splits2 = utils_rf.feature_statistics(X_train, Y_train, X_valid, Y_valid, n_estimators=10, clip_low=0, clip_high=20)
pd.Series(importances).sort_values(ascending=False).to_csv('../data_tmp/tmp.csv')

base_score = mean_squared_error(Y_valid, model.predict(X_valid).clip(0, 20))
score_drop = {}
# features = X_train.columns
for ci, c in tqdm(enumerate(features)):
    temp_df = X_valid.copy()

    # if temp_df[c].dtypes.name != 'category':
    cur_score = []
    for _ in range(5):
        temp_df[:, ci] = np.random.permutation(temp_df[:, ci])
        val_pred = model.predict(temp_df) # [features]
        cur_score.append( mean_squared_error(Y_valid, val_pred.clip(0, 20)) )
    _m = np.array(cur_score).mean()
    _s = np.array(cur_score).std()
    cur_score = (_m - base_score, (_m - base_score) / (_s + 1e-10))
    # If our current rmse score is less than base score
    # it means that feature most probably is a bad one
    # and our model is learning on noise
    score_drop[c] = cur_score
    # print(c, score_drop[c])
# sorted(score_drop.items(), key=lambda x: x[1][0], reverse=True)
score_drop = pd.DataFrame({f: {'mean': a, 'tstat': b} for f, (a, b) in score_drop.items()}).T
pd.DataFrame({f: {'mean': a, 'tstat': b} for f, (a, b) in score_drop.items()}).T.sort_values('tstat').to_csv('../data_tmp/tmp1.csv')
pd.Series({f: v for f, v in zip(features, model.feature_importances_)}).sort_values(ascending=False).to_csv('../data_tmp/tmp.csv')

# category is in top features, mean encode it and RF again!
mean_enc_name = '_'.join(['cat', 'avg', 'item'])
encoded_feature = utils.expanding_mean_encoding(matrix, 'item_cnt_month', ['item_category_id'])
matrix[mean_enc_name] = encoded_feature
matrix = utils.lag_feature(matrix, [1, 2, 12], mean_enc_name)
matrix.drop([mean_enc_name], axis=1, inplace=True)

# try shop mean encode, drop and RF again!
mean_enc_name = '_'.join(['shop', 'avg', 'item'])
encoded_feature = utils.expanding_mean_encoding(matrix, 'item_cnt_month', ['shop_id'])
matrix[mean_enc_name] = encoded_feature
matrix = utils.lag_feature(matrix, [1, 2, 12], mean_enc_name)
matrix.drop([mean_enc_name], axis=1, inplace=True)

# try shop x cat mean encode, drop and RF again!
mean_enc_name = '_'.join(['cat', 'shop', 'avg', 'item'])
encoded_feature = utils.expanding_mean_encoding(matrix, 'item_cnt_month', ['item_category_id', 'shop_id'])
matrix[mean_enc_name] = encoded_feature
matrix = utils.lag_feature(matrix, [1, 2, 12], mean_enc_name)
matrix.drop([mean_enc_name], axis=1, inplace=True)

matrix.to_pickle('../data/tmp3.pkl')
# matrix = pd.read_pickle('../data/tmp3.pkl')

mean_enc_name = '_'.join(['item', 'avg', 'item'])
encoded_feature = utils.expanding_mean_encoding(matrix, 'item_cnt_month', ['item_id'])
matrix[mean_enc_name] = encoded_feature
matrix = utils.lag_feature(matrix, [1, 2, 12], mean_enc_name)
matrix.drop([mean_enc_name], axis=1, inplace=True)

matrix.to_pickle('../data/tmp4.pkl')
# matrix = pd.read_pickle('../data/tmp4.pkl')

for agg_cols, agg_names, lags in tqdm(
        [(['date_block_num'], ['date'], [1]),
         (['date_block_num', 'item_id'], ['date', 'item'], [1, 2, 3, 6, 12]),
         (['date_block_num', 'shop_id'], ['date', 'shop'], [1, 2, 3, 6, 12]),
         (['date_block_num', 'item_category_id'], ['date', 'cat'], [1, 2, 3, 12]),
         (['date_block_num', 'shop_id', 'item_category_id'], ['date', 'shop', 'cat'], [1, 2, 3, 12]),
         (['date_block_num', 'shop_0'], ['date', 'shop_0'], [1]),
         (['date_block_num', 'shop_1'], ['date', 'shop_1'], [1]),
         (['date_block_num', 'shop_2'], ['date', 'shop_2'], [1]),
         (['date_block_num', 'cat_0'],  ['date', 'cat_0'], [1]),
         (['date_block_num', 'cat_1'],  ['date', 'cat_1'], [1]),
         (['date_block_num', 'cat_2'],  ['date', 'cat_2'], [1]),]):
    ts = time.time()
    mean_enc_name = '_'.join(agg_names + ['avg', 'item_cnt'])
    group = matrix.groupby(agg_cols).agg({'item_cnt_month': ['mean']})
    group.columns = [mean_enc_name]
    group.reset_index(inplace=True)

    if mean_enc_name in matrix.columns:
        raise Exception

    matrix = pd.merge(matrix, group, on=agg_cols, how='left')
        
    matrix[mean_enc_name] = matrix[mean_enc_name].astype(np.float16)
    matrix = utils.lag_feature(matrix, lags, mean_enc_name)
    matrix.drop([mean_enc_name], axis=1, inplace=True)
    if any(c.endswith('_x') for c in matrix.columns):
        raise Exception
    print(time.time() - ts)

matrix.to_pickle('../data/tmp5.pkl')
# matrix = pd.read_pickle('../data/tmp5.pkl')

for agg_cols, agg_names, lags in tqdm(
        [
         (['shop_0'], ['shop_0'], [1]),
         (['shop_1'], ['shop_1'], [1]),
         (['shop_2'], ['shop_2'], [1]),
         (['cat_0'], ['cat_0'], [1]),
         (['cat_1'], ['cat_1'], [1]),
         (['cat_2'], ['cat_2'], [1]),
         (['item_0'], ['item_0'], [1]),
         (['item_1'], ['item_1'], [1]),
         (['item_2'], ['item_2'], [1]),
         (['item_3'], ['item_3'], [1]),
         (['item_4'], ['item_4'], [1]),
         ]):
    ts = time.time()
    mean_enc_name = '_'.join(agg_names + ['avg', 'item'])
    encoded_feature = utils.expanding_mean_encoding(matrix, 'item_cnt_month', agg_cols)
    matrix[mean_enc_name] = encoded_feature
    matrix = utils.lag_feature(matrix, lags, mean_enc_name)
    matrix.drop([mean_enc_name], axis=1, inplace=True)
    print(time.time() - ts)

matrix.to_pickle('../data/tmp6.pkl')
# matrix = pd.read_pickle('../data/tmp6.pkl')

for c in matrix.columns:
    if np.issubdtype(matrix[c], np.number):
        s = (~np.isfinite(matrix[c]) & ~np.isnan(matrix[c])).sum()
        if s > 0:
            print(c)
            matrix.loc[np.where(~np.isfinite(matrix[c]))[0], c] = np.nan






ts = time.time()
train['revenue'] = train['item_price'] *  train['item_cnt_day']
group = train.groupby(['date_block_num','shop_id']).agg({'revenue': ['sum']})
group.columns = ['date_shop_revenue']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32)

group = group.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})
group.columns = ['shop_avg_revenue']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['shop_id'], how='left')
matrix['shop_avg_revenue'] = matrix['shop_avg_revenue'].astype(np.float32)

matrix['delta_revenue'] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
matrix['delta_revenue'] = matrix['delta_revenue'].astype(np.float16)


matrix = utils.lag_feature(matrix, [1], 'delta_revenue')
matrix['delta_revenue_lag_1'].fillna(0.0, inplace=True)
matrix.drop(['date_shop_revenue','shop_avg_revenue','delta_revenue'], axis=1, inplace=True)
time.time() - ts


matrix['month'] = matrix['date_block_num'] % 12
days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
matrix['days'] = matrix['month'].map(days).astype(np.int8)


ts = time.time()
# matrix = matrix[matrix.date_block_num > 11]
time.time() - ts

ts = time.time()
def fill_na(df):
    for col in tqdm(df.columns):
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('item_cnt' in col):
                df[col].fillna(0, inplace=True)
    return df

matrix = fill_na(matrix)
time.time() - ts

matrix.to_pickle('../data/tmp7.pkl')
matrix = pd.read_pickle('../data/tmp7.pkl')

drop_cols = ['ID', 'shop_name', 'item_name', 'item_category_name']
target = 'item_cnt_month'
m = matrix.drop(drop_cols, axis=1)
m = m[m.date_block_num > 11]
m.fillna(0, inplace=True)

X_train = m[m.date_block_num < 33].drop([target], axis=1)
Y_train = m[m.date_block_num < 33][target]
X_valid = m[m.date_block_num == 33].drop([target], axis=1)
Y_valid = m[m.date_block_num == 33][target]

model, err, importances, features, gain_counter, thresh_counter, thresh_counter2, best_splits, best_splits2 = utils_rf.feature_statistics(X_train, Y_train, X_valid, Y_valid, n_estimators=10, clip_low=0, clip_high=20)
pd.Series(importances).sort_values(ascending=False).to_csv('../tmp/tmp.csv')

base_score = mean_squared_error(Y_valid.values, model.predict(X_valid).clip(0, 20))
score_drop = {}
features = X_train.columns
for c in features:
    temp_df = X_valid.copy()

    if temp_df[c].dtypes.name != 'category':
        temp_df[c] = np.random.permutation(temp_df[c].values)
        val_pred = model.predict(temp_df[features])
        cur_score = mean_squared_error(Y_valid.values, val_pred.clip(0, 20))

        # If our current rmse score is less than base score
        # it means that feature most probably is a bad one
        # and our model is learning on noise
        score_drop[c] = np.round(cur_score - base_score, 4)
        print(c, score_drop[c])
sorted(score_drop.items(), key=lambda x: x[1], reverse=True)
pd.Series(score_drop).sort_values().to_csv('../tmp/tmp1.csv')




matrix.info()
for c in matrix.columns:
    if matrix[c].dtype == 'float64':
        print(c)
        matrix[c] = matrix[c].astype('float32')

for c in matrix.columns:
    if matrix[c].isna().any():
        print(c)
        matrix[c].fillna(0, inplace=True)

matrix.to_pickle('../data/tmp8.pkl')
# matrix = pd.read_pickle('../data/tmp8.pkl')


# drop_cols = ['ID', 'shop_name', 'item_name', 'item_category_name']
# target = 'item_cnt_month'
# m = matrix.drop(drop_cols, axis=1)
# m = m[m.date_block_num > 11]
# m.fillna(0, inplace=True)

# X_train = m[m.date_block_num < 33].drop([target], axis=1).astype(np.float32)
# Y_train = m[m.date_block_num < 33][target].astype(np.float32)
# X_valid = m[m.date_block_num == 33].drop([target], axis=1).astype(np.float32)
# Y_valid = m[m.date_block_num == 33][target].astype(np.float32)

# model, err, importances, features, gain_counter, thresh_counter, thresh_counter2, best_splits, best_splits2 = utils_rf.feature_statistics(X_train, Y_train, X_valid, Y_valid, n_estimators=10, clip_low=0, clip_high=20)
# pd.Series(importances).sort_values(ascending=False).round(3)


pkl_name = 'data_2020-06-24_01.pkl'
matrix.to_pickle('../data/%s' % pkl_name)


del matrix
del items
del shops
del cats
del train
# leave test for submission
gc.collect()


%cd "~/kaggle/competitive-data-science-predict-future-sales/src"

import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
import gc
import time


test = pd.read_csv('../data/test.csv')

data = pd.read_pickle('../data/%s' % pkl_name)


sorted(data.columns)
drop_cols = ['ID', 'item_category_name', 'item_name', 'shop_name']
# drop_cols = ['ID', 'shop_id', 'shop_name', 'item_name', 'item_id', 'item_category_name', 'item_category_id']
# drop_cols += ['si_dist_6_lag_1', 'si_dist_4_lag_1', 'si_dist_2_lag_1', 'si_dist_3_lag_1', 'si_dist_7_lag_1', 'si_dist_59_lag_1', 'si_dist_57_lag_1', 'si_dist_9_lag_1', 'si_dist_17_lag_1', 'si_dist_18_lag_1', 'si_dist_20_lag_1', 'si_dist_58_lag_1', 'si_dist_56_lag_1', 'si_dist_53_lag_1', 'si_dist_34_lag_1', 'si_dist_37_lag_1', 'si_dist_42_lag_1', 'si_dist_45_lag_1', 'si_dist_8_lag_1', 'si_dist_61_lag_1', 'month', 'shop_1', 'si_min_item_100_lag_1', 'si_dist_93_lag_1', 'si_dist_92_lag_1', 'si_dist_81_lag_1', 'si_dist_80_lag_1', 'days', 'si_dist_76_lag_1', 'date_avg_item_cnt_lag_1', 'si_dist_62_lag_1', 'si_dist_64_lag_1', 'si_dist_65_lag_1', 'si_dist_13_lag_1', 'si_dist_75_lag_1', 'si_dist_15_lag_1', 'si_dist_16_lag_1', 'si_dist_67_lag_1', 'si_dist_31_lag_1', 'si_dist_5_lag_1', 'si_dist_39_lag_1', 'si_dist_88_lag_1', 'date_item_avg_item_cnt_lag_6', 'si_dist_79_lag_1', 'date_item_avg_item_cnt_lag_12', 'si_dist_55_lag_1', 'date_cat_avg_item_cnt_lag_12', 'si_dist_90_lag_1', 'si_dist_69_lag_1', 'si_dist_29_lag_1', 'date_cat_2_avg_item_cnt_lag_1', 'si_min_item_99_lag_1', 'si_dist_91_lag_1', 'si_dist_83_lag_1', 'delta_revenue_lag_1', 'si_dist_47_lag_1', 'si_dist_41_lag_1', 'si_dist_94_lag_1', 'si_dist_51_lag_1', 'si_dist_43_lag_1', 'si_dist_82_lag_1', 'si_dist_63_lag_1', 'si_dist_40_lag_1', 'cat_shop_avg_item_lag_12', 'si_dist_25_lag_1', 'item_avg_item_lag_12', 'si_dist_32_lag_1', 'si_dist_60_lag_1', 'cat_avg_item_lag_12', ]
# data.drop(drop_cols, axis=1, inplace=True)
data = matrix[matrix.date_block_num > 11].drop(drop_cols, axis=1)
data.info()

# data = matrix
data = data[data['date_block_num'] > 11]

for c in data.columns:
    if np.isnan(data[c]).any():
        print(c)
        data[c].fillna(0, inplace=True)

X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

del data
gc.collect()

X_train.info()

# [62]	validation_0-rmse:0.79811	validation_1-rmse:0.89230

ts = time.time()

model = XGBRegressor(
    max_depth=8,
    n_estimators=300,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8,
    eta=0.2,
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 5)

time.time() - ts

pd.Series({f: v for f, v in zip(X_train.columns, model.feature_importances_)}).sort_values().to_csv('../tmp/tmp3.csv')
base_score = mean_squared_error(Y_valid.values, model.predict(X_valid).clip(0, 20))
score_drop = {}
features = X_train.columns
for c in features:
    temp_df = X_valid.copy()
    
    if temp_df[c].dtypes.name != 'category':
        temp_df[c] = np.random.permutation(temp_df[c].values)
        val_pred = model.predict(temp_df[features])
        cur_score = mean_squared_error(Y_valid.values, val_pred.clip(0, 20))
        
        # If our current rmse score is less than base score
        # it means that feature most probably is a bad one
        # and our model is learning on noise
        score_drop[c] = np.round(cur_score - base_score, 4)
        print(c, score_drop[c])
sorted(score_drop.items(), key=lambda x: x[1], reverse=True)
pd.Series(score_drop).sort_values().to_csv('../tmp/tmp1.csv')


import pickle
pickle.dump(model, open('../data/model_xgb_2020-06-22_02.pkl', "wb"))

terr = mean_squared_error(Y_train.values.astype('float32'), model.predict(X_train).clip(0, 20))
verr = mean_squared_error(Y_valid.values.astype('float32'), model.predict(X_valid).clip(0, 20))
print('train-rmse=%.4f valid-rmse=%.4f' % (terr, verr))

Y_pred = model.predict(X_valid).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)

# from collections import Counter
# from sklearn.metrics import confusion_matrix
# Y_pred = model.predict(X_valid).clip(0, 20)
# sorted(Counter(Y_pred.round(0)).items())
# sorted(Counter(Y_valid).items())
# cm = confusion_matrix(Y_valid.values.astype('int8'), Y_pred.round(0).astype('int8'))
# pd.DataFrame(cm).to_csv('../data/errors.xgb.best.csv')

# Y_test = model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('../data/xgb_submission_2020-06-22_02.csv', index=False)
pd.Series(Y_pred).to_csv('../data/xgb_submission_2020-06-22_02_valid.csv', index=False)



model = pickle.load(open('../data/model_xgb_2020-06-22.pkl', "rb"))

base_score = mean_squared_error(Y_valid.values, model.predict(X_valid).clip(0, 20))

score_drop = {}
features = X_train.columns
for c in features:
    temp_df = X_valid.copy()
    
    if temp_df[c].dtypes.name != 'category':
        temp_df[c] = np.random.permutation(temp_df[c].values)
        val_pred = model.predict(temp_df[features])
        cur_score = mean_squared_error(Y_valid.values, val_pred.clip(0, 20))
        
        # If our current rmse score is less than base score
        # it means that feature most probably is a bad one
        # and our model is learning on noise
        score_drop[c] = np.round(cur_score - base_score, 4)
        print(c, score_drop[c])
sorted(score_drop.items(), key=lambda x: x[1], reverse=True)
pd.Series(score_drop).sort_values().to_csv('./tmp1.csv')


score_drop = {}
features = X_train.columns
for i1 in range(len(features) - 1):
    for i2 in range(i1 + 1, len(features)):
        c1 = features[i1]
        c2 = features[i2]
        
        temp_df = X_valid.copy()
        
        if temp_df[c1].dtypes.name != 'category' and temp_df[c2].dtypes.name != 'category':
            temp_df[[c1, c2]] = np.random.permutation(temp_df[[c1, c2]].values)
            val_pred = model.predict(temp_df[features])
            cur_score = mean_squared_error(Y_valid.values, val_pred.clip(0, 20))
            
            # If our current rmse score is less than base score
            # it means that feature most probably is a bad one
            # and our model is learning on noise
            score_drop[(c1, c2)] = np.round(cur_score - base_score, 4)
            print((c1, c2), score_drop[(c1, c2)])
            
            
sorted(score_drop.items(), key=lambda x: x[1], reverse=True)[:30]


model, err, importances, features, gain_counter, thresh_counter, thresh_counter2, best_splits, best_splits2 = utils_rf.feature_statistics(X_train, Y_train, X_valid, Y_valid, n_estimators=10, clip_low=0, clip_high=20)
pd.Series(importances).sort_values(ascending=False).round(3)


def knn_mean_enc(data, target='item_cnt_month', dist_cols=['item_0', 'item_1', 'item_2', 'item_3', 'item_4']):
    from sklearn.neighbors import NearestNeighbors
    
    # groupby date
    for date_block_num in data['date_block_num'].unique():
        m = data[data['date_block_num'] == date_block_num]
        n_items = m.item_id.unique().shape[0]
        n_shops = m.shop_id.unique().shape[0]
        n_rows = m.shape[0]
        nbrs = NearestNeighbors(n_neighbors=100 * n_shops, algorithm='auto').fit(m[dist_cols])
        mean_enc_name = '_'.join(['date', 'item_knn', 'avg', 'item'])
        encoded_feature = np.zeros(n_rows)
        for r in tqdm(range(m.shape[0])):
            ds, indices = nbrs.kneighbors(m[dist_cols].iloc[r:r+1].values)
            val = m[target].iloc[indices[0]].mean()
            encoded_feature[r] = val
        
        np.corrcoef(m[target], encoded_feature)[0, 1]
        

data = pd.read_pickle('../data/%s' % pkl_name)
sorted(data.columns)
drop_cols = ['ID', 'item_category_name', 'item_name', 'shop_name']
drop_cols = ['ID', 'shop_id', 'shop_name', 'item_name', 'item_id', 'item_category_name', 'item_category_id']
drop_cols += ['date_item_avg_item_cnt_lag_6', 'date_shop_avg_item_cnt_lag_6', 'date_shop_avg_item_cnt_lag_12', 'delta_revenue_lag_1', 'shop_avg_item_lag_12', 'item_cnt_month_lag_1_item_cnt_month_lag_2_avg_item_lag_12', 'item_avg_item_lag_12', 'date_cat_avg_item_cnt_lag_12', 'date_shop_1_avg_item_cnt_lag_1', 'date_item_avg_item_cnt_lag_3', 'date_item_avg_item_cnt_lag_12', 'date_shop_avg_item_cnt_lag_3', 'month', 'days', 'cat_avg_item_lag_12', 'date_shop_avg_item_cnt_lag_1', 'date_shop_avg_item_cnt_lag_2', 'date_shop_0_avg_item_cnt_lag_1', 'date_shop_2_avg_item_cnt_lag_1', 'item_avg_item_lag_1', ]
data.drop(drop_cols, axis=1, inplace=True)
data.info()

data = data[data['date_block_num'] > 11]

for c in data.columns:
    if np.isnan(data[c]).any():
        print(c)
        data[c].fillna(0, inplace=True)

X_train = data[data.date_block_num < 33].drop(['item_cnt_month', 'date_block_num', ], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month', 'date_block_num', ], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month', 'date_block_num', ], axis=1)

del data
gc.collect()


X_train.info()

# [62]	validation_0-rmse:0.79811	validation_1-rmse:0.89230

ts = time.time()
model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8,
    eta=0.02,
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 10)

time.time() - ts


terr = mean_squared_error(Y_train.values.astype('float32'), model.predict(X_train).clip(0, 20))
verr = mean_squared_error(Y_valid.values.astype('float32'), model.predict(X_valid).clip(0, 20))
print('train-rmse=%.4f valid-rmse=%.4f' % (terr, verr))

Y_pred = model.predict(X_valid).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)

# from collections import Counter
# from sklearn.metrics import confusion_matrix
# Y_pred = model.predict(X_valid).clip(0, 20)
# sorted(Counter(Y_pred.round(0)).items())
# sorted(Counter(Y_valid).items())
# cm = confusion_matrix(Y_valid.values.astype('int8'), Y_pred.round(0).astype('int8'))
# pd.DataFrame(cm).to_csv('../data/errors.xgb.best.csv')

# Y_test = model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('../data/xgb_submission_2020-06-19_04.csv', index=False)
pd.Series(Y_pred).to_csv('../data/xgb_submission_2020-06-19_04_valid.csv', index=False)


plot_features(model, (10,14))

# tffm sandbox

import importlib

import tffm
importlib.reload(tffm)

model = tffm.TFFMRegressor(
    order=5,
    rank=3,
    log_dir='../data/tffm_log',
    verbose=50,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.00003),
    n_epochs=3,
    batch_size=1000,
    init_std=0.0005,
    reg=0.01,
    input_type='dense'
)

model.fit(X_train.values.astype('float32'), Y_train.values.astype('float32'), show_progress=True)

terr = mean_squared_error(Y_train.values.astype('float32'), model.predict(X_train).clip(0, 20))
verr = mean_squared_error(Y_valid.values.astype('float32'), model.predict(X_valid).clip(0, 20))
print('train-rmse=%.4f valid-rmse=%.4f' % (terr, verr))

Y_pred = model.predict(X_valid).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.round(0)
submission.to_csv('../data/submission_2020-06-16_03_tffm.csv', index=False)

best = pd.read_csv('../data/xgb_submission_2020-06-14_01.csv')
best.round(1).to_csv('../data/xgb_submission_2020-06-14_01_rounded.csv', index=False)
((submission.set_index('ID') * 0.5 + best.set_index('ID') * 0.5).clip(0, 20).reset_index().sort_index()).to_csv('../data/submission_2020-06-16_02_tffm_plus_best_xgb.csv', index=False)







#fastFM sandbox
from fastFM import als
from sklearn.metrics import mean_squared_error
import scipy.sparse as sp

xtrain = sp.csc_matrix(X_train)
xvalid = sp.csc_matrix(X_valid)

n_iter = 50
rank = 4
seed = 42
step_size = 1
l2_reg_w = 0
l2_reg_V = 0

fm = als.FMRegression(n_iter=0, l2_reg_w=l2_reg_w,
        l2_reg_V=l2_reg_V, rank=rank, random_state=seed)
# initalize coefs
fm.fit(xtrain, Y_train.values)

rmse_train = []
rmse_test = []
for i in range(1, n_iter):
    fm.fit(xtrain, Y_train.values, n_more_iter=step_size)
    y_pred = fm.predict(xvalid)
    train_err = np.sqrt(mean_squared_error(fm.predict(xtrain), Y_train.values))
    valid_err = np.sqrt(mean_squared_error(fm.predict(xvalid), Y_valid.values))
    print("train-rmse=%.4f valid-rmse=%.4f" % (train_err, valid_err))
    rmse_train.append(train_err)
    rmse_test.append(valid_err)

train_err = np.sqrt(mean_squared_error(fm.predict(xtrain).clip(0, 20), Y_train.values))
valid_err = np.sqrt(mean_squared_error(fm.predict(xvalid).clip(0, 20), Y_valid.values))
print("train-rmse=%.4f valid-rmse=%.4f" % (train_err, valid_err))



# catboost sandbox
Y_train = Y_train.astype('float32')
Y_valid = Y_valid.astype('float32')
for c in X_train:
    if X_train[c].dtype == 'float16':
        X_train[c] = X_train[c].astype('float32')
        X_valid[c] = X_train[c].astype('float32')
X_train.info()

cb_model = CatBoostRegressor(n_estimators=1000,
                             learning_rate=0.02,
                             depth=8,
                             loss_function='RMSE',
                             eval_metric='RMSE',
                             min_data_in_leaf=300,
                             colsample_bylevel=0.8,
                             random_seed=42,
                             od_type='Iter',
                             od_wait=10)

cb_model.fit(X_train, Y_train,
             use_best_model=True,
             eval_set=(X_valid, Y_valid), silent=False, plot=True)

def naive_eda_and_level_plus_ternd_model():
    test.nunique()
    test_shop = test.shop_id.unique()
    test_item = test.item_id.unique()
    
    missing_items = np.setdiff1d(test_item, train.item_id.unique())
    len(missing_items)
    
    # train.date = pd.to_datetime(train.date.values, format='%d.%m.%Y').strftime('%Y%m%d')
    train['month'] = train.date.str.slice(0, 6)
    
    rev = train.groupby('month').apply(lambda df: df.item_cnt_day.dot(df.item_price))
    rev.sort_index().plot(marker='.')
    rev = rev.reset_index()
    rev['year'] = rev.month.str.slice(0, 4)
    rev['m'] = rev.month.str.slice(4, 6)
    rev.pivot_table(values=0, index='m', columns='year').plot(marker='.')
    
    # baseline: prev month and clipping to [0, 20]
    # "level + trend" model:
    # sales[m, y] = level[m - 1, y] + average_trend[m - 1]
    # average_trend[m - 1] = average_on_prev_years( level[m, prev_year] - level[m - 1, prev_year] )
    
    def baseline():
        s = train.merge(items, on='item_id', how='left')[['month', 'shop_id', 'item_id', 'item_category_id', 'item_cnt_day']]
        s['year'] = s.month.str.slice(0, 4)
        level = s[s.month.isin([prev_month(month)])].groupby(['year', 'item_category_id', 'shop_id', 'item_id']).item_cnt_day.sum().droplevel(0)
        mindex = pd.Index([tuple(x) for x in test[['shop_id', 'item_id']].merge(items)[['item_category_id', 'shop_id', 'item_id']].values])
        mindex.names = ['item_category_id', 'shop_id', 'item_id']
        return level.reindex(mindex).fillna(0).clip(0, 20)
    
    def level_plus_trend():
        s = train.merge(items, on='item_id', how='left')[['month', 'shop_id', 'item_id', 'item_category_id', 'item_cnt_day']]
        s['year'] = s.month.str.slice(0, 4)
        
        months = pd.Series(sorted(s.month.unique()))
        prev_month = lambda x: months[months.searchsorted(x) - 1]
        month = '201511'
        pm = month[-2:]
        pm_prev = prev_month(month)[-2:]
        prev_month(month)
        pyears = ['2013', '2014']
        
        pms = [y + pm for y in pyears]
        pms_prev = [y + pm_prev for y in pyears]
        
        # sales count per month
        # s1 = s[s.month.isin(pms)].groupby(['year', 'item_category_id', 'item_id']).item_cnt_day.sum()
        
        level = s[s.month.isin([prev_month(month)])].groupby(['year', 'item_category_id', 'shop_id', 'item_id']).item_cnt_day.sum().droplevel(0)
        
        t1 = s[s.month.isin(pms)].groupby(['year', 'item_category_id', 'shop_id', 'item_id']).item_cnt_day.sum()
        t0 = s[s.month.isin(pms_prev)].groupby(['year', 'item_category_id', 'shop_id', 'item_id']).item_cnt_day.sum()
        
        dt = t1.reindex(t1.index | t0.index).fillna(0.) - t0.reindex(t1.index | t0.index).fillna(0.)
        dt = dt.groupby(level=(1, 2, 3)).mean()
    
        x = (level + dt).reindex(mindex)
        return x.fillna(0).clip(0, 20)
        
    b0 = baseline()
    b1 = level_plus_trend()
    comp = b1.reset_index().merge(b0.reset_index(), on=['item_category_id', 'shop_id', 'item_id'])
    comp[comp.item_cnt_day_x != comp.item_cnt_day_y]
    
    s1 = b1.droplevel(0).reset_index().merge(test)[['ID', 'item_cnt_day']].set_index('ID').sort_index()
    s1.columns = ['item_cnt_month']
    s1.to_csv('../data/level_plus_trend.csv')
    s1.sum()
    b0.sum()



import pandas as pd
import numpy as np
from itertools import product

%cd "~/kaggle/competitive-data-science-predict-future-sales/src"
sales = pd.read_csv('../data/sales_train.csv')
index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from that month
grid = [] 
for block_num in sales['date_block_num'].unique():
    cur_shops = sales[sales['date_block_num']==block_num]['shop_id'].unique()
    cur_items = sales[sales['date_block_num']==block_num]['item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

#turn the grid into pandas dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

#get aggregated values for (shop_id, item_id, month)
gb = sales.groupby(index_cols)['item_cnt_day'].agg(target='sum').reset_index()

#fix column names
#join aggregated data to the grid
all_data = pd.merge(grid,gb,how='left',on=index_cols).fillna(0)
#sort the data
all_data.sort_values(['date_block_num','shop_id','item_id'],inplace=True)



encoded_feature = utils.kfold_mean_encoding(all_data, 'target', ['item_id'], k=5, global_mean=0.3343)
encoded_feature = utils.expanding_mean_encoding(all_data, 'target', ['item_id'], global_mean=0.3343)

%timeit -n1 -r3 utils.expanding_mean_encoding(all_data, 'target', ['item_id'], global_mean=0.3343)
%timeit -n1 -r3 utils.kfold_mean_encoding(all_data, 'target', ['item_id'], k=5, global_mean=0.3343)

    


# Differently from other homework we will not implement OOF predictions ourselves
# but use sklearn's `cross_val_predict`
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

# We will use two metrics for KNN
for metric in ['cosine']: # 'minkowski', 
    print (metric)
    
    # Set up splitting scheme, use StratifiedKFold
    # use skf_seed and n_splits defined above with shuffle=True
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=skf_seed)
    
    # Create instance of our KNN feature extractor
    # n_jobs can be larger than the number of cores
    NNF = NearestNeighborsFeats(n_jobs=4, k_list=k_list, metric=metric)
    
    # Get KNN features using OOF use cross_val_predict with right parameters
    preds = cross_val_predict(NNF, X, Y, cv=skf)# YOUR CODE GOES HERE
    
    # Save the features
    np.save('data/knn_feats_%s_train.npy' % metric, preds)
    

