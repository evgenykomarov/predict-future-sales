# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 21:21:24 2020

@author: komarov
"""

%cd "~/kaggle/competitive-data-science-predict-future-sales/src"

import pandas as pd
from sklearn.neighbors import NearestNeighbors
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
from multiprocessing import Pool
from tensorflow.keras import layers
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
import tensorflow as tf
from xgboost import XGBRegressor
from xgboost import plot_importance
import seaborn as sns
import gc
import time
import sys
import pickle
import importlib

sys.path.append(".")
import utils_rf
import utils
import text_features
import knn_features
importlib.reload(utils_rf)
importlib.reload(utils)
importlib.reload(text_features)
importlib.reload(knn_features)

seed = 123

train = pd.read_csv('../data/sales_train1.csv')
train.date = train.date.astype('str')
# train.to_csv('../data/sales_train1.csv', index=False)
# sample_sub = pd.read_csv('../data/sample_submission.csv')
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
matrix = pd.merge(matrix, shops.drop('shop_name', axis=1), on=['shop_id'], how='left')
matrix = pd.merge(matrix, items.drop('item_name', axis=1), on=['item_id'], how='left')
matrix = pd.merge(matrix, cats.drop('item_category_name', axis=1), on=['item_category_id'], how='left')
# matrix['city_code'] = matrix['city_code'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
# matrix['type_code'] = matrix['type_code'].astype(np.int8)
# matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)
time.time() - ts

ts = time.time()
matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')
time.time() - ts

ts = time.time()
group = train.groupby(['item_id']).agg({'item_price': ['mean']})
group.columns = ['item_avg_item_price']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['item_id'], how='left')
matrix['item_avg_item_price'] = matrix['item_avg_item_price'].astype(np.float16)

group = train.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})
group.columns = ['date_item_avg_item_price']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_price'] = matrix['date_item_avg_item_price'].astype(np.float16)
time.time() - ts



cols = ['item_0', 'item_1', 'item_2', 'item_3', 'item_4']
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
skf2 = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)
target = 'item_cnt_month'
target_all = [target, 'item_avg_item_price', 'item_first_sale']

train_index = matrix[(matrix['date_block_num'] >= 11) & (matrix['date_block_num'] < 33)].index
valid_index = matrix[(matrix['date_block_num'] >= 11) & (matrix['date_block_num'] >= 33)].index

def cv_predict(estimator, X, Y, Y_strat, cv=skf):
    # same as cross_val_predict but strats and predictions 
    # on different Y's: Y for pred and Y_stat for splits
    pred_blocks = []
    for train_i, test_i in cv.split(X, Y_strat):
        X_train, Y_train, X_test = X[train_i], Y[train_i], X[test_i]
        estimator.fit(X_train, Y_train)
        pred_blocks.append((estimator.predict(X_test), test_i))
    
    test_indices = np.hstack(x for _, x in pred_blocks)
    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))
    
    return np.vstack(x for x, _ in pred_blocks)[inv_test_indices]

def predict_mean_of_cv(_estimator, X_train, Y_train, Y_train_strat, X_test, cv=skf):
    from sklearn.base import clone
    # too large chunks dont fit into memory and we
    # split train into n chunks and calculate prediction for test on each
    # return average of all the splits
    pred_blocks = []
    for _, train_i in cv.split(X_train, Y_train_strat):
        X_tr, Y_tr = X_train[train_i], Y_train[train_i]
        estimator = clone(_estimator)
        estimator.fit(X_tr, Y_tr)
        pred_blocks.append(estimator.predict(X_test))
    
    _ = np.stack(pred_blocks, axis=2)
    preds = _.mean(axis=2)
    preds_std  = _.std(axis=2)
    
    return preds, preds_std

ts = time.time()
k_list = [10, 100, 500, 1000]
nnrf = knn_features.NearestNeighborsRegFeats(n_jobs=8, k_list=k_list, metric=metric, eps=1e-6, seed=seed)
metric = 'minkowski'
preds = []
for _, idx in tqdm(skf2.split(matrix.loc[train_index, cols].values, matrix.loc[train_index, target].values)):
    X = matrix.loc[train_index[idx], cols].values
    Y = matrix.loc[train_index[idx], target_all].values
    Y_strat = Y[:, 0]
    _1 = cv_predict(nnrf, X, Y, Y_strat, cv=skf)
    preds.append((_1, train_index[idx]))
print(time.time() - ts)
feats = np.vstack(x for x, _ in preds)[np.argsort(np.hstack(x for _, x in preds))]
with open('../data_tmp/tmp1.npy', 'wb') as f:
    np.save(f, feats)

ts = time.time()
nnrf = knn_features.NearestNeighborsRegFeats(n_jobs=8, k_list=k_list, metric=metric)
X_train = matrix.loc[train_index, cols].values
Y_train = matrix.loc[train_index, target_all].values
X_test = matrix.loc[valid_index, cols].values
test_knn_feats, test_knn_feats_std = predict_mean_of_cv(nnrf, X_train, Y_train, Y_train[:, 0], X_test, skf2)
print(time.time() - ts)

feats = np.vstack([feats, test_knn_feats])
fs = ['item_knn_%s_%s_%s' % (k, typ, var) for k in k_list for typ in ['avg', 'med', 'max', 'std'] for var in ['item', 'price', 'sale']]

matrix = pd.concat([matrix, pd.DataFrame(feats, columns=fs, index=train_index.tolist() + valid_index.tolist())], axis=1)
matrix.info()
for c in matrix.columns:
    if matrix[c].dtype == 'float16':
        matrix[c] = matrix[c].astype(np.float32)


%timeit -n1 -r1 matrix.to_parquet('../data_tmp/tmp1.snappy', compression='snappy')
matrix = pd.read_parquet('../data_tmp/tmp1.snappy')

del X_train
del Y_train
del X_valid
del Y_valid
del m
del model, err, importances, features, gain_counter, thresh_counter, thresh_counter2, best_splits, best_splits2
gc.collect()

ts = time.time()
matrix = utils.lag_feature(matrix, [1, 2, 3, 6, 12], 'item_cnt_month')
time.time() - ts

%timeit -n1 -r1 matrix.to_parquet('../data_tmp/tmp2.snappy', compression='snappy')
# matrix = pd.read_pickle('../data/tmp2.pkl')

drop_cols = ['ID'] # 'shop_name', 'item_name', 'item_category_name'
drop_cols += [c for c in matrix.columns if "_knn_10_" in c]
drop_cols += ['item_knn_500_avg_sale', 'item_knn_100_med_price', 'item_knn_100_avg_sale', 'item_knn_1000_max_sale', 'item_knn_1000_max_price', 'item_knn_500_std_price', 'item_knn_100_std_price', 'item_knn_500_med_sale', 'item_knn_100_max_sale', 'item_knn_100_avg_price', 'item_knn_1000_std_sale', 'item_knn_1000_avg_sale', 'item_knn_1000_avg_item', 'item_knn_100_max_price', 'item_knn_1000_std_item', 'item_knn_500_std_sale', 'item_knn_1000_med_item', 'item_knn_1000_med_sale', 'item_knn_1000_med_price', 'date_item_avg_item_price', 'item_knn_1000_std_price', 'item_knn_500_std_item', 'item_knn_1000_max_item', 'item_knn_100_std_sale', ]
target = 'item_cnt_month'
data = matrix.drop(drop_cols, axis=1)
data = data[data.date_block_num > 11]
data.fillna(0, inplace=True)

X_train = data[data.date_block_num < 33].drop([target], axis=1)
Y_train = data[data.date_block_num < 33][target]
X_valid = data[data.date_block_num == 33].drop([target], axis=1)
Y_valid = data[data.date_block_num == 33][target]
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
gc.collect()
gc.collect()

X_train.info()
model = XGBRegressor(
    max_depth=8,
    n_estimators=40,
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



model, err, importances, features, gain_counter, thresh_counter, thresh_counter2, best_splits, best_splits2 = utils_rf.feature_statistics(X_train, Y_train, X_valid, Y_valid, n_estimators=10, clip_low=0, clip_high=20)
pd.Series(importances).sort_values(ascending=False).to_csv('../data_tmp/tmp.csv')

base_score = mean_squared_error(Y_valid.values, model.predict(X_valid).clip(0, 20))
score_drop = {}
features = X_train.columns
for c in features:
    temp_df = X_valid.copy()

    if temp_df[c].dtypes.name != 'category':
        cur_score = []
        for _ in range(5):
            temp_df[c] = np.random.permutation(temp_df[c].values)
            val_pred = model.predict(temp_df[features])
            cur_score.append( mean_squared_error(Y_valid.values, val_pred.clip(0, 20)) )
        cur_score = np.array(cur_score).mean()
        # If our current rmse score is less than base score
        # it means that feature most probably is a bad one
        # and our model is learning on noise
        score_drop[c] = np.round(cur_score - base_score, 4)
        print(c, score_drop[c])
sorted(score_drop.items(), key=lambda x: x[1], reverse=True)
pd.Series(score_drop).sort_values(ascending=False).to_csv('../data_tmp/tmp1.csv')
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
lags = [1, 2, 3, 4, 5, 6]
matrix = utils.lag_feature(matrix, lags, 'date_item_avg_item_price')

for i in tqdm(lags):
    matrix['delta_price_lag_' + str(i)] = \
        matrix['date_item_avg_item_price_lag_' + str(i)] / matrix['item_avg_item_price'] - 1.0

gc.collect()

matrix['delta_price_lag'] = matrix[['delta_price_lag_' + str(lag) for lag in lags]].astype('float32').fillna(method='bfill', axis=1).iloc[:, 0]
matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)
matrix['delta_price_lag'].fillna(0, inplace=True)

# https://stackoverflow.com/questions/31828240/first-non-null-value-per-row-from-a-list-of-pandas-columns/31828559
# matrix['price_trend'] = matrix[['delta_price_lag_1','delta_price_lag_2','delta_price_lag_3']].bfill(axis=1).iloc[:, 0]
# Invalid dtype for backfill_2d [float16]

fetures_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
for i in lags:
    fetures_to_drop += ['date_item_avg_item_price_lag_' + str(i)]
    fetures_to_drop += ['delta_price_lag_' + str(i)]

matrix.drop(fetures_to_drop, axis=1, inplace=True)

time.time() - ts


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
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product, chain
from sklearn.preprocessing import LabelEncoder
from sklearn import feature_extraction
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from xgboost import XGBRegressor
from xgboost import plot_importance
# from catboost import CatBoostRegressor
import seaborn as sns
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
import scipy.sparse as sp

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
    

