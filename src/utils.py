# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:54:46 2020

@author: komarov
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
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
import text_features

def find_zero_var_features(data_train):
    s = data_train.var()
    return s[s == 0].index.tolist()


def recursive_features_eliminate(path_to_df='../data_processed/many_features_minus_correlated.snappy',
                                 target='item_cnt_month',
                                 max_iter=20,
                                 always_drop=['ID'],
                                 keep_cols=['date_block_num', 'item_id', 'shop_id', 'item_category_id', 'item_0', 'item_1', 'item_2', 'item_3', 'item_4', 'shop_0', 'shop_1', 'shop_2', 'cat_0', 'cat_1', 'cat_2'],
                                 tstat_thresh=-1, # tstat = (mean - base_model) / std threshold to drop features from the model
                                 max_drop_ratio=0.2, # fraction of features to drop from the non-eliminated features,
                                 min_drop = 0,
                                 train_index=None,
                                 valid_index=None,
                                 reload_data=False, # if memory is bottleneck deleting and re-reading data on each iteration
                                 model_type=None, # model to train and use for elimination
                                 clip_down=-np.inf,
                                 clip_up=np.inf,
                                 max_xgb_features=200, # memory issues if >200 columns
                                 num_permutations=5, # number of permutation for feature to get stats on baseline score drop
                                 n_estimators=10,
                                 max_depth=10,
                                 max_samples=0.8,
                                 seed=123,
                                 n_jobs=8,
                                 learning_rate=0.2,
                                 ):
    data = pd.read_parquet(path_to_df)

    random_string = ''.join(['0123456789abcdef'[np.random.choice(16)] for _ in range(8)])
    drops_cols_path = lambda iter_num: '../data_tmp/%s_%02d.pkl' % (random_string, iter_num)
    score_drop_path = lambda iter_num: '../data_tmp/%s_%02d_score_drop.csv' % (random_string, iter_num)
    importance_path = lambda iter_num: '../data_tmp/%s_%02d_importance.csv' % (random_string, iter_num)

    new_drop_cols = []

    # eliminate features with zero variance on the train data
    # new_drop_cols += [feat for feat in find_zero_var_features(data.loc[train_index]) if feat not in keep_cols]
    drop_cols = []
    iter_num = 0
    elimination_path = []
    drop_random_features = False
    all_cols = set(data.columns)
    while (iter_num == 0 or len(new_drop_cols) > 0) and iter_num < max_iter:
        drop_cols += new_drop_cols
        pd.Series(drop_cols).sort_values().to_pickle(drops_cols_path(iter_num))

        if reload_data and 'data' not in locals() or drop_random_features:
            data = pd.read_parquet(path_to_df, columns=list(all_cols - set(drop_cols)))

        data.drop([c for c in set(always_drop + drop_cols) if c in data.columns], axis=1, inplace=True)
        data.fillna(0, inplace=True)
        features = data.drop([target], axis=1).columns.tolist()

        drop_random_features = model_type == 'xgb' and len(features) > max_xgb_features
        if drop_random_features:
            num_drop = max(len(features) - max_xgb_features, 0)
            random_drop = np.random.choice([f for f in features if f not in keep_cols], size=num_drop, replace=False)
            features = [f for f in features if f not in set(random_drop)]
            data.drop(random_drop, axis=1, inplace=True)
            gc.collect()

        X_train = data.loc[train_index].drop([target], axis=1).values
        Y_train = data.loc[train_index][target].values
        X_valid = data.loc[valid_index].drop([target], axis=1).values
        Y_valid = data.loc[valid_index][target].values
        if reload_data or drop_random_features:
            del data; gc.collect()

        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=n_estimators,
                                          max_depth=max_depth,
                                          max_features='sqrt',
                                          max_samples=max_samples,
                                          random_state=seed,
                                          n_jobs=n_jobs) # all cores
            model.fit(X_train, Y_train)
        elif model_type == 'xgb':
            model = XGBRegressor(
                max_depth=max_depth,
                n_estimators=n_estimators,
                min_child_weight=300,
                colsample_bytree=0.5,
                subsample=max_samples,
                eta=learning_rate,
                seed=seed)

            model.fit(
                X_train,
                Y_train,
                eval_metric="rmse",
                eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
                verbose=True,
                early_stopping_rounds=-1)
        else:
            raise NotImplemented

        imps = pd.Series({f: v for f, v in zip(features, model.feature_importances_)}).sort_values(ascending=False)
        imps.to_csv(importance_path(iter_num))

        train_loss = mean_squared_error(Y_train, model.predict(X_train).clip(clip_down, clip_up))
        base_score = mean_squared_error(Y_valid, model.predict(X_valid).clip(clip_down, clip_up))
        print('iter=%02d train_loss=%.4f valid_loss=%.4f' % (iter_num, train_loss, base_score))
        elimination_path.append((base_score, drop_cols))
        score_drop = {}
        del X_train, Y_train
        gc.collect()

        # going only through worst 100 features
        cis = [features.index(x) for x in imps[-min(100, len(features)):].index.tolist()]
        for ci in tqdm(cis):
            temp_df = X_valid.copy()
            cur_score = []
            for _ in range(num_permutations):
                temp_df[:, ci] = np.random.permutation(temp_df[:, ci])
                val_pred = model.predict(temp_df)
                cur_score.append(mean_squared_error(Y_valid, val_pred.clip(clip_down, clip_up)))
            _m = np.array(cur_score).mean()
            _s = np.array(cur_score).std()
            score_drop[features[ci]] = (_m - base_score, (_m - base_score) / (_s + 1e-10))
        score_drop = pd.DataFrame({f: {'mean_drop': a, 'tstat_drop': b} for f, (a, b) in score_drop.items()}).T
        score_drop = score_drop.sort_values('tstat_drop')
        score_drop.to_csv(score_drop_path(iter_num))
        thresh = min(tstat_thresh, np.quantile(score_drop['tstat_drop'].values, max_drop_ratio))
        if min_drop == 0:
            new_drop_cols = [feat for feat in score_drop[score_drop['tstat_drop'] <= thresh].index if feat not in keep_cols \
                             and feat not in drop_cols]
        else:
            new_drop_cols = score_drop.index[:min_drop]
        print('tstat thresh=%.03f dropped %s features: %s' % (thresh, len(new_drop_cols), repr(sorted(new_drop_cols))))
        iter_num += 1
        del X_valid, Y_valid
        gc.collect()
    return drop_cols, elimination_path, model, random_string, score_drop


def df_info(matrix):
    from collections import Counter
    cnt = Counter(matrix.dtypes.values)
    size = sum(v * k.itemsize * matrix.shape[0] for k, v in cnt.items()) / 1024**3
    print(cnt)
    print('size=%.03f Gb' % size)


def gen_corr_pairs(data):
    for i in range(data.shape[1]):
        for j in range(i + 1, data.shape[1]):
            yield data.iloc[:, i].values.astype(np.float64), data.iloc[:, j].values.astype(np.float64)


def gen_corr_ids(data):
    for i in range(data.shape[1]):
        for j in range(i + 1, data.shape[1]):
            yield i, j


def correlate_pair(v1, v2):
    # %timeit -n10 -r10 f(np.random.choice(range(n_features), 2, False))
    # 50 / 1000 * 600 * 600 / 8
    mask = (~np.isnan(v1)) & (~np.isnan(v2))
    return np.corrcoef(v1[mask], v2[mask])[0, 1]


def find_duplicates_features(matrix, id_cols=['item_cnt_month', 'ID', 'shop_id', 'item_id', 'item_category_id'],
                             date_col='date_block_num',
                             date_thresh=11,
                             corr_thresh=0.97,
                             corr_sample_size=10000,
                             n_jobs=8,
                             ):
    index = np.random.choice(np.where(matrix[date_col] > date_thresh)[0], corr_sample_size, False)
    index.sort()
    data = matrix.iloc[index].drop(id_cols, axis=1)
    del matrix
    gc.collect()

    features = data.columns
    n_features = len(features)

    cp = correlate_pair
    with Pool(n_jobs) as p:
        cors = p.starmap(cp, gen_corr_pairs(data))

    cora = np.empty((n_features, n_features), dtype=np.float64)
    for cor_value, pair in zip(cors, gen_corr_ids(data)):
        cora[pair[0], pair[1]] = cor_value

    drop_cols = set()
    clusters = []
    for i in range(n_features):
        fi = features[i]
        for j in range(i + 1, n_features):
            fj = features[j]
            if abs(cora[i, j]) >= corr_thresh:
                drop_cols.add(fj)
                found = False
                for cl in clusters:
                    if fi in cl:
                        cl.add(fj)
                        found = True
                    elif fj in cl:
                        cl.add(fi)
                        found = True
                if not found:
                    clusters.append(set([fi, fj]))

    return drop_cols, clusters


def reduce_dimension(matrix,
                     id_cols=['item_cnt_month', 'ID', 'shop_id', 'item_id', 'item_category_id', 'date_block_num'],
                     sub_sample_row=100,
                     dim_size=50,
                     type='pca',  # 'nmf'
                     ):
    from sklearn.decomposition import PCA
    from sklearn.decomposition import NMF
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import pandas as pd

    scaler = MinMaxScaler()
    scaler.fit(matrix.drop(id_cols, axis=1).fillna(0))

    index = np.random.choice(matrix.index, min(matrix.shape[0], matrix.shape[1] * sub_sample_row), replace=False)
    data_train = scaler.transform(matrix.loc[index].drop(id_cols, axis=1).fillna(0))
    if type == 'pca':
        estimator = PCA(dim_size)
    elif type == 'nmf':
        estimator = NMF(dim_size)
    data_transformed = estimator.fit_transform(data_train)
    var_total = np.trace(np.cov(data_train.T))
    var_explained = np.trace(np.cov(data_transformed.T))
    print('variance explained dim=%s = %.3f%%' % (dim_size, var_explained / var_total * 100))

    data_test = scaler.transform(matrix.drop(id_cols, axis=1).fillna(0))
    res = []
    chunk = 100000
    n_chunks = int(np.ceil(data_test.shape[0] / chunk))
    for i in tqdm(range(n_chunks)):
        res.append(estimator.transform(data_test[chunk * i:chunk * (i + 1)]))
    res = np.vstack(res)
    res = pd.DataFrame(res, columns=['%s_%s' % (type, i) for i in range(50)], index=matrix.index)
    res = pd.concat([matrix[id_cols], res], axis=1)
    res.to_parquet('../data_processed/%s%s.snappy' % (type, dim_size), compression='snappy')
    return res


def stratified_data_generator(X, Y, Y_strat, cv):
    for _, index_patch in cv.split(X, Y_strat):
        yield X[index_patch], Y[index_patch]


def cross_val_predict1(estimator, X, Y, Y_strat, cv):
    """
    same as cross_val_predict but strats and predictions
    on different Y's: Y for pred and Y_stat for splits
    """
    pred_blocks = []
    for train_i, test_i in cv.split(X, Y_strat):
        X_train, Y_train, X_test = X[train_i], Y[train_i], X[test_i]
        estimator.fit(X_train, Y_train)
        pred_blocks.append((estimator.predict(X_test), test_i))

    test_indices = np.hstack(x for _, x in pred_blocks)
    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    return np.vstack(x for x, _ in pred_blocks)[inv_test_indices]


def cross_val_predict_approximate(estimator, X, Y, Y_strat, X_test, cv):
    """
    too large chunks dont fit into memory and we
    split train into n chunks and calculate prediction for test on each
    return average of all the splits
    """
    from sklearn.base import clone
    pred_blocks = []

    for X_tr, Y_tr in stratified_data_generator(X, Y, Y_strat, cv):
        e = clone(estimator)
        e.fit(X_tr, Y_tr)
        pred_blocks.append(e.predict(X_test))

    _ = np.stack(pred_blocks, axis=2)
    preds = _.mean(axis=2)
    preds_std = _.std(axis=2)

    return preds, preds_std


def embed_categorical_features():
    items = pd.read_csv('../data/items.csv')
    shops = pd.read_csv('../data/shops.csv')
    cats = pd.read_csv('../data/item_categories.csv')

    items.item_name = [text_features.norm_text(x) for x in items.item_name.values]
    shops.shop_name = [text_features.norm_text(x) for x in shops.shop_name.values]
    cats.item_category_name = [text_features.norm_text(x) for x in cats.item_category_name.values]

    # D = 4 seems mapping different items to same vector too often
    # D = 5 starts to distinguish
    item_features = text_features.text_feature_to_vec(items.item_name.values, D=5, epochs=3)
    text_features.show_neighbors(items.item_name.values, item_features)

    shop_features = text_features.text_feature_to_vec(shops.shop_name.values, D=3, epochs=1000)
    text_features.show_neighbors(shops.shop_name.values, shop_features)

    cat_features = text_features.text_feature_to_vec(cats.item_category_name.values, D=3, epochs=1000)
    text_features.show_neighbors(cats.item_category_name.values, cat_features)

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


# borrowed from here and modified https://www.kaggle.com/dlarionov/feature-engineering-xgboost
def plot_features(booster, figsize):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    return plot_importance(booster=booster, ax=ax)


def lag_feature(df, lags, col, id_cols=['date_block_num', 'shop_id', 'item_id']):
    tmp = df[id_cols + [col]].drop_duplicates()
    for i in tqdm(lags):
        shifted = tmp.copy()
        shifted.columns = id_cols + [col + '_lag_' + str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=id_cols, how='left')
    return df


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df


def lag1_expanding_mean_encoding(matrix, date_col='date_block_num', target='item_cnt_month', group_cols=[], global_mean=None):
    # groupby the and calculate the rolling mean on the previous values

    if global_mean is None:
        _s = matrix.groupby(date_col)[target].sum()
        _c = matrix.groupby(date_col)[target].count()
        global_mean = (_s.cumsum() - _s) / (1e-10 + _c.cumsum() - _c)
        global_mean = matrix[[date_col]].merge(global_mean, on=date_col)[target]

    _ = matrix.groupby([date_col] + group_cols).agg({target: ['sum', 'count']})
    _.columns = ['sum', 'count']
    _ = _.reset_index()
    encoded_feature = (_.groupby(group_cols)['sum'].cumsum() -
                       _['sum']) / (_.groupby(group_cols)['count'].cumsum() - _['count'])
    encoded_feature.name = 'enc_feature'
    encoded_feature = pd.merge(_[[date_col] + group_cols], encoded_feature, left_index=True, right_index=True, how='left')
    encoded_feature = matrix[[date_col] + group_cols].merge(encoded_feature, on=[date_col] + group_cols, how='left').loc[matrix.index, 'enc_feature']
    encoded_feature.fillna(global_mean, inplace=True)

    # You will need to compute correlation like that
    corr = np.corrcoef(matrix[target].values, encoded_feature.values)[0][1]
    print('corr with target=%.03f' % corr)
    return encoded_feature


def date_expanding_mean_encoding(matrix, date_col='date_block_num', target='item_cnt_month', group_cols=[], global_mean=None):
    """
    calculate per (date, group_cols) average
    group by group_cols and calculate expanding mean
    """

    if global_mean is None:
        _ = matrix.groupby(date_col)[target].sum()
        global_mean = _.cumsum() / matrix.groupby(date_col)[target].count().cumsum()
        global_mean = matrix[[date_col]].merge(global_mean, on=date_col)[target]

    cs = matrix.groupby([date_col] + group_cols, as_index=True)[[target]].sum().reset_index()
    cc = matrix.groupby([date_col] + group_cols, as_index=True)[[target]].count().reset_index()

    if len(group_cols) > 0:
        _ = cs.groupby(group_cols)[target].cumsum() / cc.groupby(group_cols)[target].cumsum()
    else:
        _ = cs[target].cumsum() / cc[target].cumsum()
    encoded_feature = matrix[[date_col] + group_cols].merge(
        cs[[date_col] + group_cols].merge(_, left_index=True, right_index=True, how='left'),
        on=[date_col] + group_cols, how='left')
    encoded_feature = encoded_feature.loc[matrix.index, target]

    # You will need to compute correlation like that
    corr = np.corrcoef(matrix[target].values, encoded_feature.values)[0][1]
    print('corr with target=%.03f' % corr)
    return encoded_feature, global_mean


def expanding_mean_encoding(matrix, target='item_cnt_month', group_cols=[], global_mean=None):
    # groupby the and calculate the rolling mean on the previous values

    if global_mean is None:
        global_mean = matrix[target].mean()

    encoded_feature = (matrix.groupby(group_cols)[target].cumsum() -
                       matrix[target]) / matrix.groupby(group_cols).cumcount()
    encoded_feature.fillna(global_mean, inplace=True)
    encoded_feature = encoded_feature.loc[matrix.index]

    # You will need to compute correlation like that
    corr = np.corrcoef(matrix[target].values, encoded_feature)[0][1]
    print(corr)
    return encoded_feature


def kfold_mean_encoding(all_data, target='target', group_cols=['item_id'], k=5, global_mean=None):
    # mean encoding:
    # K-fold validated
    from sklearn.model_selection import KFold

    if global_mean is None:
        global_mean = all_data[target].mean()

    encoded_feature = pd.Series(index=all_data.index, dtype=np.float64)

    kf = KFold(n_splits=k, shuffle=False)

    for train_ind, val_ind in kf.split(all_data):
        x_tr, x_val = all_data.iloc[train_ind], all_data.iloc[val_ind]
        map_fn = x_tr.groupby(group_cols)[target].mean().reset_index()
        encoded_feature.iloc[val_ind] = x_val[group_cols].reset_index(). \
            merge(map_fn, on=group_cols, how='left'). \
            drop(group_cols, axis=1).set_index('index')[target]

    encoded_feature.fillna(global_mean, inplace=True)
    # You will need to compute correlation like that
    corr = np.corrcoef(all_data[target].values, encoded_feature)[0][1]
    print(corr)
    return encoded_feature
