# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 21:21:24 2020

@author: komarov
"""

# %cd "~/kaggle/competitive-data-science-predict-future-sales/src"
# !ls -lah 

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
from tensorflow.keras import layers
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
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

cats = pd.read_csv('../data/item_categories.csv')
items = pd.read_csv('../data/items.csv')
train = pd.read_csv('../data/sales_train1.csv')
train.date = train.date.astype('str')
# train.to_csv('../data/sales_train1.csv', index=False)
sample_sub = pd.read_csv('../data/sample_submission.csv')
shops = pd.read_csv('../data/shops.csv')
test = pd.read_csv('../data/test.csv')

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

# borrowed from here and modified https://www.kaggle.com/dlarionov/feature-engineering-xgboost
def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

plt.figure(figsize=(10, 4))
plt.xlim(-100, 3000)
sns.boxplot(x=train.item_cnt_day)

plt.figure(figsize=(10, 4))
plt.xlim(train.item_price.min(), train.item_price.max()*1.1)
sns.boxplot(x=train.item_price)

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

# train['revenue'] = train['item_price'] *  train['item_cnt_day']

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
shops = pd.merge(
      pd.DataFrame(shop_features, index=pd.Index(shops.shop_name.values, name='shop_name'), columns=["shop_%s" % i for i in range(shop_features.shape[1])]).reset_index(),
      shops,
      on='shop_name').drop_duplicates()
items = pd.merge(
      pd.DataFrame(item_features, index=pd.Index(items.item_name.values, name='item_name'), columns=["item_%s" % i for i in range(item_features.shape[1])]).reset_index(),
      items,
      on='item_name').drop_duplicates()
cats = pd.merge(
      pd.DataFrame(cat_features, index=pd.Index(cats.item_category_name.values, name='item_category_name'), columns=["cat_%s" % i for i in range(cat_features.shape[1])]).reset_index(),
      cats,
      on='item_category_name').drop_duplicates()
matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
matrix = pd.merge(matrix, items, on=['item_id'], how='left')
matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')
# matrix['city_code'] = matrix['city_code'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
# matrix['type_code'] = matrix['type_code'].astype(np.int8)
# matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)
time.time() - ts

train_ind = matrix[(matrix['date_block_num'] >= 11) & (matrix['date_block_num'] < 33)].index
val_ind  = matrix[(matrix['date_block_num'] >= 11) & (matrix['date_block_num'] == 33)].index

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

cols = ['item_0', 'item_1', 'item_2', 'item_3',  'item_4']
for train_idx, test_idx in skf.split(matrix.loc[train_ind[:100000]], matrix.loc[train_ind[:100000], 'item_cnt_month']):
    print(test_idx)
    n_jobs = 8
    target = 'item_cnt_month'
    metric='minkowski'
    X = matrix.loc[train_idx, cols].values
    Y = matrix.loc[train_idx, target].values
    X_test = matrix.loc[test_idx, cols].values
    NN = NearestNeighbors(n_neighbors=100,
                          metric='minkowski', 
                          n_jobs=8, 
                          algorithm='brute' if metric=='cosine' else 'auto')
    %timeit -n1 -r1 NN.fit(X)
    %timeit -n1 -r1 NN.kneighbors(X_test)
    x = X[i:i+1, :]
    def get_features_for_one(x):
        distances, neighs = NN.kneighbors(x)
        return Y_train[neighs].mean()
    
    with Pool(n_jobs) as p:
        gen = (X[i:i + 1] for i in range(X.shape[0]))
        test_feats = p.map(get_features_for_one, gen)
        
    res = np.vstack(test_feats)
    

ts = time.time()
matrix = lag_feature(matrix, [1, 2, 3, 6, 12], 'item_cnt_month')
time.time() - ts

matrix.to_pickle('../data/tmp.pkl')
matrix = pd.read_pickle('../data/tmp.pkl')

drop_cols = ['ID', 'shop_name', 'item_name', 'item_category_name', 'shop_0', 'shop_1', 'shop_2', 'cat_0', 'cat_1', 'cat_2', 'item_0', 'item_1', 'item_2', 'item_3', 'item_4']
target = 'item_cnt_month'
m = matrix.drop(drop_cols, axis=1)
m = m[m.date_block_num > 11]
m.fillna(0, inplace=True)

X_train = m[m.date_block_num < 33].drop([target], axis=1)
Y_train = m[m.date_block_num < 33][target]
X_valid = m[m.date_block_num == 33].drop([target], axis=1)
Y_valid = m[m.date_block_num == 33][target]

model, err, importances, features, gain_counter, thresh_counter, thresh_counter2, best_splits, best_splits2 = utils_rf.feature_statistics(X_train, Y_train, X_valid, Y_valid, n_estimators=10, clip_low=0, clip_high=20)


#          tr       val
# 0  0.849660  0.967019
# 1  0.832691  0.939528
# 2  0.821814  0.926490
# 3  0.818825  0.924790
# 4  0.819366  0.922294
# 5  0.819724  0.922039
# 6  0.818608  0.922749
# 7  0.818735  0.922561
# 8  0.817863  0.922561
# 9  0.817901  0.922768

# category is in top features, mean encode it and RF again!
mean_enc_name = '_'.join(['cat', 'avg', 'item'])
encoded_feature = utils.expanding_mean_encoding(matrix, 'item_cnt_month', ['item_category_id'])
matrix[mean_enc_name] = encoded_feature
matrix = lag_feature(matrix, [1, 12], mean_enc_name)
matrix.drop([mean_enc_name], axis=1, inplace=True)

drop_cols = ['ID', 'shop_name', 'item_name', 'item_category_name', 'shop_0', 'shop_1', 'shop_2', 'cat_0', 'cat_1', 'cat_2', 'item_0', 'item_1', 'item_2', 'item_3', 'item_4']
target = 'item_cnt_month'
m = matrix.drop(drop_cols, axis=1)
m = m[m.date_block_num > 11]
m.fillna(0, inplace=True)

X_train = m[m.date_block_num < 33].drop([target], axis=1)
Y_train = m[m.date_block_num < 33][target]
X_valid = m[m.date_block_num == 33].drop([target], axis=1)
Y_valid = m[m.date_block_num == 33][target]

model, err, importances, features, gain_counter, thresh_counter, thresh_counter2, best_splits, best_splits2 = utils_rf.feature_statistics(X_train, Y_train, X_valid, Y_valid, n_estimators=10, clip_low=0, clip_high=20)

#          tr       val
# 0  0.848029  0.940065
# 1  0.820561  0.927835
# 2  0.808258  0.920968
# 3  0.807389  0.915521
# 4  0.804251  0.915241
# 5  0.804919  0.915275
# 6  0.804362  0.916752
# 7  0.804204  0.915285
# 8  0.802487  0.914483
# 9  0.802092  0.913683


# try shop mean encode, drop and RF again!
mean_enc_name = '_'.join(['shop', 'avg', 'item'])
encoded_feature = utils.expanding_mean_encoding(matrix, 'item_cnt_month', ['shop_id'])
matrix[mean_enc_name] = encoded_feature
matrix = lag_feature(matrix, [1, 12], mean_enc_name)
matrix.drop([mean_enc_name], axis=1, inplace=True)

drop_cols = ['ID', 'shop_name', 'item_name', 'item_category_name', 'shop_0', 'shop_1', 'shop_2', 'cat_0', 'cat_1', 'cat_2', 'item_0', 'item_1', 'item_2', 'item_3', 'item_4']
target = 'item_cnt_month'
m = matrix.drop(drop_cols, axis=1)
m = m[m.date_block_num > 11]
m.fillna(0, inplace=True)

X_train = m[m.date_block_num < 33].drop([target], axis=1)
Y_train = m[m.date_block_num < 33][target]
X_valid = m[m.date_block_num == 33].drop([target], axis=1)
Y_valid = m[m.date_block_num == 33][target]

model, err, importances, features, gain_counter, thresh_counter, thresh_counter2, best_splits, best_splits2 = utils_rf.feature_statistics(X_train, Y_train, X_valid, Y_valid, n_estimators=10, clip_low=0, clip_high=20)

#          tr       val
# 0  0.866199  0.979238
# 1  0.834564  0.944609
# 2  0.826763  0.934637
# 3  0.819277  0.925920
# 4  0.813710  0.920112
# 5  0.814046  0.916089
# 6  0.807369  0.911445
# 7  0.804545  0.910938
# 8  0.804665  0.905007
# 9  0.803864  0.906997

matrix.columns.tolist()

# try shop x cat mean encode, drop and RF again!
mean_enc_name = '_'.join(['cat', 'shop', 'avg', 'item'])
encoded_feature = utils.expanding_mean_encoding(matrix, 'item_cnt_month', ['item_category_id', 'shop_id'])
matrix[mean_enc_name] = encoded_feature
matrix = lag_feature(matrix, [1, 12], mean_enc_name)
matrix.drop([mean_enc_name], axis=1, inplace=True)

drop_cols = ['ID', 'shop_name', 'item_name', 'item_category_name', 'shop_0', 'shop_1', 'shop_2', 'cat_0', 'cat_1', 'cat_2', 'item_0', 'item_1', 'item_2', 'item_3', 'item_4']
target = 'item_cnt_month'
m = matrix.drop(drop_cols, axis=1)
m = m[m.date_block_num > 11]
m.fillna(0, inplace=True)

X_train = m[m.date_block_num < 33].drop([target], axis=1)
Y_train = m[m.date_block_num < 33][target]
X_valid = m[m.date_block_num == 33].drop([target], axis=1)
Y_valid = m[m.date_block_num == 33][target]

model, err, importances, features, gain_counter, thresh_counter, thresh_counter2, best_splits, best_splits2 = utils_rf.feature_statistics(X_train, Y_train, X_valid, Y_valid, n_estimators=10, clip_low=0, clip_high=20)

#          tr       val
# 0  0.900956  0.988979
# 1  0.834003  0.912848
# 2  0.814602  0.893347
# 3  0.805974  0.893060
# 4  0.803847  0.894871
# 5  0.807421  0.895015
# 6  0.805097  0.895340
# 7  0.803103  0.895724
# 8  0.801858  0.895754
# 9  0.801433  0.894465

matrix.to_pickle('../data/tmp2.pkl')
matrix = pd.read_pickle('../data/tmp2.pkl')

best_splits = dict(best_splits)
# mean encode lag interactions:

for f0, f1 in [('item_cnt_month_lag_1', 'item_cnt_month_lag_2')]:
    mean_enc_name = '_'.join([f0, f1, 'avg', 'item'])
    bins0 = [-np.inf] + sorted(best_splits[f0]) + [np.inf]
    bins1 = [-np.inf] + sorted(best_splits[f1]) + [np.inf]
    matrix['f0'] = pd.cut(matrix[f0], bins0)
    matrix['f1'] = pd.cut(matrix[f1], bins1)
    group_cols = ['f0', 'f1']
    encoded_feature = utils.expanding_mean_encoding(matrix, 'item_cnt_month', group_cols)
    matrix.drop(group_cols, axis=1, inplace=True)
    matrix[mean_enc_name] = encoded_feature
    matrix = lag_feature(matrix, [1, 12], mean_enc_name)
    matrix.drop(mean_enc_name, axis=1, inplace=True)

drop_cols = ['ID', 'shop_name', 'item_name', 'item_category_name', 'shop_0', 'shop_1', 'shop_2', 'cat_0', 'cat_1', 'cat_2', 'item_0', 'item_1', 'item_2', 'item_3', 'item_4']
target = 'item_cnt_month'
m = matrix.drop(drop_cols, axis=1)
m = m[m.date_block_num > 11]
m.fillna(0, inplace=True)

X_train = m[m.date_block_num < 33].drop([target], axis=1)
Y_train = m[m.date_block_num < 33][target]
X_valid = m[m.date_block_num == 33].drop([target], axis=1)
Y_valid = m[m.date_block_num == 33][target]

model, err, importances, features, gain_counter, thresh_counter, thresh_counter2, best_splits, best_splits2 = utils_rf.feature_statistics(X_train, Y_train, X_valid, Y_valid, n_estimators=10, clip_low=0, clip_high=20)
#          tr       val
# 0  0.832448  0.935837
# 1  0.812377  0.917962
# 2  0.805472  0.911502
# 3  0.802480  0.906920
# 4  0.800837  0.905397
# 5  0.799597  0.905255
# 6  0.799978  0.904288
# 7  0.800281  0.904410
# 8  0.799312  0.904073
# 9  0.798535  0.902952


matrix.to_pickle('../data/tmp3.pkl')
matrix = pd.read_pickle('../data/tmp3.pkl')


mean_enc_name = '_'.join(['item', 'avg', 'item'])
encoded_feature = utils.expanding_mean_encoding(matrix, 'item_cnt_month', ['item_id'])
matrix[mean_enc_name] = encoded_feature
matrix = lag_feature(matrix, [1, 12], mean_enc_name)
matrix.drop([mean_enc_name], axis=1, inplace=True)

drop_cols = ['ID', 'shop_name', 'item_name', 'item_category_name', 'shop_0', 'shop_1', 'shop_2', 'cat_0', 'cat_1', 'cat_2', 'item_0', 'item_1', 'item_2', 'item_3', 'item_4']
target = 'item_cnt_month'
m = matrix.drop(drop_cols, axis=1)
m = m[m.date_block_num > 11]
m.fillna(0, inplace=True)

X_train = m[m.date_block_num < 33].drop([target], axis=1)
Y_train = m[m.date_block_num < 33][target]
X_valid = m[m.date_block_num == 33].drop([target], axis=1)
Y_valid = m[m.date_block_num == 33][target]

model, err, importances, features, gain_counter, thresh_counter, thresh_counter2, best_splits, best_splits2 = utils_rf.feature_statistics(X_train, Y_train, X_valid, Y_valid, n_estimators=10, clip_low=0, clip_high=20)
#          tr       val
# 0  0.848578  0.946070
# 1  0.816975  0.919288
# 2  0.804606  0.909504
# 3  0.797678  0.903073
# 4  0.797902  0.904079
# 5  0.791984  0.893393
# 6  0.792939  0.893319
# 7  0.791924  0.892510
# 8  0.790734  0.892875
# 9  0.791638  0.893164


matrix.to_pickle('../data/tmp4.pkl')
matrix = pd.read_pickle('../data/tmp4.pkl')



for agg_cols, agg_names, lags in tqdm(
        [([], [], [1]),
         (['item_id'], ['item'], [1, 2, 3, 6, 12]),
         (['shop_id'], ['shop'], [1, 2, 3, 6, 12]),
         (['item_category_id'], ['cat'], [1, 2, 3, 12]),
         (['shop_id', 'item_category_id'], ['shop', 'cat'], [1, 2, 3, 12]),
         (['shop_0'], ['shop_0'], [1]),
         (['shop_1'], ['shop_1'], [1]),
         (['shop_2'], ['shop_2'], [1]),
         (['cat_0'], ['cat_0'], [1]),
         (['cat_1'], ['cat_1'], [1]),
         (['cat_2'], ['cat_2'], [1]),]):
    ts = time.time()
    mean_enc_name = '_'.join(['date'] + agg_names + ['avg', 'item_cnt'])


    group = matrix.groupby(['date_block_num'] + agg_cols).agg({'item_cnt_month': ['mean']})
    group.columns = [mean_enc_name]
    group.reset_index(inplace=True)

    if mean_enc_name in matrix.columns:
        raise Exception

    matrix = pd.merge(matrix, group, on=['date_block_num'] + agg_cols, how='left')
        
    matrix[mean_enc_name] = matrix[mean_enc_name].astype(np.float16)
    matrix = lag_feature(matrix, lags, mean_enc_name)
    matrix.drop([mean_enc_name], axis=1, inplace=True)
    if any(c.endswith('_x') for c in matrix.columns):
        raise Exception
    print(time.time() - ts)


matrix.to_pickle('../data/tmp5.pkl')
matrix = pd.read_pickle('../data/tmp5.pkl')

drop_cols = ['ID', 'shop_name', 'item_name', 'item_category_name', 'shop_0', 'shop_1', 'shop_2', 'cat_0', 'cat_1', 'cat_2', 'item_0', 'item_1', 'item_2', 'item_3', 'item_4']
target = 'item_cnt_month'
m = matrix.drop(drop_cols, axis=1)
m = m[m.date_block_num > 11]
m.fillna(0, inplace=True)

X_train = m[m.date_block_num < 33].drop([target], axis=1)
Y_train = m[m.date_block_num < 33][target]
X_valid = m[m.date_block_num == 33].drop([target], axis=1)
Y_valid = m[m.date_block_num == 33][target]

model, err, importances, features, gain_counter, thresh_counter, thresh_counter2, best_splits, best_splits2 = utils_rf.feature_statistics(X_train, Y_train, X_valid, Y_valid, n_estimators=10, clip_low=0, clip_high=20)

#          tr       val
# 0  0.806813  0.926937
# 1  0.780090  0.906789
# 2  0.771492  0.900014
# 3  0.767331  0.899756
# 4  0.765485  0.893207
# 5  0.763429  0.891750
# 6  0.760633  0.886710
# 7  0.761498  0.884854
# 8  0.760110  0.884409
# 9  0.760527  0.885065


for c in matrix.columns:
    if np.issubdtype(matrix[c], np.number):
        s = (~np.isfinite(matrix[c]) & ~np.isnan(matrix[c])).sum()
        if s > 0:
            print(c)
            # matrix.loc[np.where(~np.isfinite(matrix[c]))[0], c] = np.nan


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

lags = [1, 2, 3, 4, 5, 6]
matrix = lag_feature(matrix, lags, 'date_item_avg_item_price')

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


matrix = lag_feature(matrix, [1], 'delta_revenue')
matrix['delta_revenue_lag_1'].fillna(0.0, inplace=True)
matrix.drop(['date_shop_revenue','shop_avg_revenue','delta_revenue'], axis=1, inplace=True)
time.time() - ts


matrix['month'] = matrix['date_block_num'] % 12
days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
matrix['days'] = matrix['month'].map(days).astype(np.int8)


ts = time.time()
matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')
time.time() - ts

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

matrix.to_pickle('../data/tmp6.pkl')
matrix = pd.read_pickle('../data/tmp6.pkl')


matrix.info()
for c in matrix.columns:
    if matrix[c].dtype == 'float64':
        print(c)
        matrix[c] = matrix[c].astype('float32')

for c in matrix.columns:
    if matrix[c].isna().any():
        print(c)
        matrix[c].fillna(0, inplace=True)

matrix.to_pickle('../data/tmp7.pkl')
matrix = pd.read_pickle('../data/tmp7.pkl')


drop_cols = ['ID', 'shop_id', 'shop_name', 'item_name', 'item_id', 'item_category_name', 'item_category_id']
target = 'item_cnt_month'
m = matrix.drop(drop_cols, axis=1)
m = m[m.date_block_num > 11]
m.fillna(0, inplace=True)

X_train = m[m.date_block_num < 33].drop([target], axis=1).astype(np.float32)
Y_train = m[m.date_block_num < 33][target].astype(np.float32)
X_valid = m[m.date_block_num == 33].drop([target], axis=1).astype(np.float32)
Y_valid = m[m.date_block_num == 33][target].astype(np.float32)

model, err, importances, features, gain_counter, thresh_counter, thresh_counter2, best_splits, best_splits2 = utils_rf.feature_statistics(X_train, Y_train, X_valid, Y_valid, n_estimators=10, clip_low=0, clip_high=20)
pd.Series(importances).sort_values(ascending=False).round(3)
#          tr       val
# 0  0.826983  0.969705
# 1  0.778036  0.914604
# 2  0.737648  0.885147
# 3  0.741158  0.880086
# 4  0.738577  0.865459
# 5  0.733905  0.859138
# 6  0.728956  0.853122
# 7  0.723238  0.850856
# 8  0.720369  0.845485
# 9  0.720264  0.847227


pkl_name = 'data_2020-06-19_02.pkl'
matrix.to_pickle('../data/%s' % pkl_name)


matrix = pd.read_pickle('../data/%s' % pkl_name)

for c in tqdm(['shop_0', 'shop_1', 'shop_2']):
    mean_enc_name = c + '_avg_item'
    matrix[mean_enc_name] = utils.expanding_mean_encoding(matrix, target='item_cnt_month', group_cols=[c])
    # matrix = utils.lag_feature(matrix, [1], mean_enc_name)
    # matrix.drop(mean_enc_name, axis=1, inplace=True)
    
for c in tqdm(['item_0', 'item_1', 'item_2', 'item_3', 'item_4']):
    mean_enc_name = c + '_avg_item'
    matrix[mean_enc_name] = utils.expanding_mean_encoding(matrix, target='item_cnt_month', group_cols=[c])
    # matrix = utils.lag_feature(matrix, [1], mean_enc_name)
    # matrix.drop(mean_enc_name, axis=1, inplace=True)

# generating mean encoded features:
def gen_knn_features(matrix, k_list=[5, 10, 20, 50], n_jobs=8, n_splits=5):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_predict

    
    for metric in ['braycurtis']: # , 'cosine', 'minkowski'
        print(metric)
        metric = 'braycurtis'
        # Create instance of our KNN feature extractor
        
        le = LabelEncoder()
        matrix['class'] = le.fit_transform(pd.cut(matrix.item_cnt_month, range(-1, 22, 2)).values)
        
        cols = ['shop_0_avg_item', 'shop_1_avg_item', 'shop_2_avg_item'] + \
               ['item_0_avg_item', 'item_1_avg_item', 'item_2_avg_item', 'item_3_avg_item', 'item_4_avg_item']
        
        # skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # for i, (train_ind, test_ind) in tqdm(enumerate(skf.split(matrix.loc[matrix.date_block_num > 11, cols], matrix['class'].values))):
        #     NNF = knn_features.NearestNeighborsFeats(n_jobs=8, k_list=k_list, metric=metric)
        #     X = matrix[cols].iloc[train_ind].values
        #     Y = matrix['class'].iloc[train_ind].values
        #     Y_reg = matrix['item_cnt_month'].iloc[train_ind].values
        #     NNF.fit(X, Y, Y_reg)
        
        #     # Get features for test
        #     test_knn_feats = NNF.predict(matrix[cols].iloc[test_ind].values)
            
        #     # Dump the features to disk
        #     np.save('../matrix/knn_feats_date_shopitem_meanenc_%s_%s_test.npy' % (metric, i), test_knn_feats)
        #     np.save('../matrix/knn_feats_date_shopitem_meanenc_%s_%s_index.npy' % (metric, i), test_ind)
        

        for date_block_num in tqdm(range(11, 34)):
            NNF = knn_features.NearestNeighborsFeats(n_jobs=n_jobs, k_list=k_list, metric=metric)
            X = matrix.loc[matrix.date_block_num == date_block_num, cols].values
            Y = matrix.loc[matrix.date_block_num == date_block_num, 'class'].values
            Y_reg = matrix.loc[matrix.date_block_num == date_block_num, 'item_cnt_month'].values
            NNF.fit(X, Y, Y_reg)
        
            # Get features for test
            test_knn_feats = NNF.predict(X)
            
            # Dump the features to disk
            np.save('../data/knn_feats_date_shopitem_me_%s_%s_test.npy' % (metric, date_block_num), test_knn_feats)
        
        matrix.drop('class', axis=1, inplace=True)
        

import os
from glob import iglob
knn_feats = []
k_list = [5, 10, 20, 50]

for date_block_num in tqdm(range(11, 34)):
    file = '../data/knn_feats_date_shopitem_me_braycurtis_%s_test.npy' % date_block_num
    # date_block_num
    # print("%s: %s" % (file, os.path.isfile(file)))
    
    d = np.load(file)
    f_names = ['si_dist_' + str(i) for i in range(d.shape[1] - len(k_list) * 4)] + \
              ['si_avg_item_' + str(i) for i in range(d.shape[1] - len(k_list) * 4, d.shape[1] - len(k_list) * 3)] + \
              ['si_min_item_' + str(i) for i in range(d.shape[1] - len(k_list) * 3, d.shape[1] - len(k_list) * 2)] + \
              ['si_max_item_' + str(i) for i in range(d.shape[1] - len(k_list) * 2, d.shape[1] - len(k_list) * 1)] + \
              ['si_std_item_' + str(i) for i in range(d.shape[1] - len(k_list) * 1, d.shape[1] - len(k_list) * 0)]
    d = pd.DataFrame(d, columns=f_names)
    d['date_block_num'] = date_block_num
    d['shop_id'] = matrix.loc[matrix.date_block_num == date_block_num, 'shop_id'].values
    d['item_id'] = matrix.loc[matrix.date_block_num == date_block_num, 'item_id'].values
    knn_feats.append(d)
cols = ['date_block_num', 'shop_id', 'item_id']

knn_feats = utils.reduce_mem_usage(pd.concat(knn_feats, axis=0)) #keys=cols, ignore_index=True
cols_to_lag = knn_feats.columns.difference(cols)
knn_feats['date_block_num'] += 1
knn_feats.columns = [c + '_lag_1' if c in cols_to_lag else c for c in knn_feats.columns]

matrix = pd.merge(matrix, knn_feats, on=cols, how='left')
del knn_feats
del d
del test_knn_feats
matrix = utils.reduce_mem_usage(matrix)
matrix.to_pickle('../data/tmp8.pkl')
matrix = pd.read_pickle('../data/tmp8.pkl')
matrix.drop(['shop_0_avg_item', 'shop_1_avg_item', 'shop_2_avg_item',
       'item_0_avg_item', 'item_1_avg_item', 'item_2_avg_item',
       'item_3_avg_item', 'item_4_avg_item'], axis=1, inplace=True)
matrix.info()


# drop_cols = ['ID', 'shop_id', 'shop_name', 'item_name', 'item_id', 'item_category_name', 'item_category_id']


drop_cols = ['ID', 'shop_name', 'item_name', 'item_category_name']
drop_cols += ['si_avg_item_119_lag_1', 'si_dist_109_lag_1', 'si_dist_32_lag_1', 'si_dist_86_lag_1', 'si_dist_111_lag_1', 'si_dist_11_lag_1', 'si_dist_43_lag_1', 'si_dist_22_lag_1', 'si_dist_87_lag_1', 'si_dist_38_lag_1', 'si_dist_98_lag_1', 'si_dist_106_lag_1', 'si_dist_13_lag_1', 'si_dist_21_lag_1', 'si_dist_110_lag_1', 'si_dist_42_lag_1', 'si_dist_39_lag_1', 'si_dist_35_lag_1', 'si_dist_17_lag_1', 'si_dist_16_lag_1', 'si_dist_26_lag_1', 'si_dist_18_lag_1', 'si_std_item_134_lag_1', 'si_dist_41_lag_1', 'si_dist_80_lag_1', 'si_dist_37_lag_1', ]
drop_cols += ['si_dist_10_lag_1', 'si_dist_63_lag_1', 'si_dist_97_lag_1', 'si_dist_25_lag_1', 'si_dist_89_lag_1', 'si_dist_95_lag_1', 'si_dist_40_lag_1', 'si_min_item_125_lag_1', ]
drop_cols += ['item_cnt_month_lag_1_item_cnt_month_lag_2_avg_item_lag_12', 'si_dist_29_lag_1', 'si_dist_20_lag_1', 'si_dist_14_lag_1', 'si_min_item_126_lag_1', 'si_dist_105_lag_1', 'si_dist_79_lag_1', 'si_dist_31_lag_1', 'si_dist_30_lag_1', 'si_dist_28_lag_1', 'si_dist_27_lag_1', 'si_dist_19_lag_1', 'si_dist_7_lag_1', 'si_dist_6_lag_1', 'si_dist_5_lag_1', 'si_dist_4_lag_1', 'si_dist_1_lag_1', 'si_dist_8_lag_1', 'si_dist_9_lag_1', 'si_dist_94_lag_1', 'si_dist_15_lag_1']
drop_cols += ['si_min_item_124_lag_1', 'si_dist_104_lag_1', 'si_dist_64_lag_1', 'si_dist_36_lag_1', 'si_dist_2_lag_1', 'si_dist_3_lag_1', 'si_dist_24_lag_1', 'si_dist_65_lag_1']
drop_cols += ['date_item_avg_item_cnt_lag_12', 'si_dist_23_lag_1', 'si_dist_113_lag_1', 'days']
drop_cols += ['cat_avg_item_lag_12', 'si_dist_93_lag_1', 'si_dist_12_lag_1']
drop_cols += ['si_dist_62_lag_1', 'si_dist_83_lag_1', 'si_dist_81_lag_1']
drop_cols += ['si_dist_102_lag_1', 'date_shop_1_avg_item_cnt_lag_1', 'si_min_item_123_lag_1']
drop_cols += ['item_avg_item_lag_12', 'si_dist_91_lag_1', 'si_dist_96_lag_1']
drop_cols += ['si_dist_82_lag_1']
drop_cols += ['si_dist_72_lag_1', 'si_dist_108_lag_1']
drop_cols += ['si_dist_60_lag_1', 'si_dist_69_lag_1']
drop_cols += ['si_dist_50_lag_1', 'si_dist_101_lag_1', 'date_cat_avg_item_cnt_lag_12']
drop_cols += ['si_dist_70_lag_1']
target = 'item_cnt_month'
m = matrix.drop(drop_cols, axis=1)
m = m[m.date_block_num > 11]
m.fillna(0, inplace=True)

X_train = m[m.date_block_num < 33].drop([target], axis=1) #.astype(np.float32)
Y_train = m[m.date_block_num < 33][target].astype(np.float32)
X_valid = m[m.date_block_num == 33].drop([target], axis=1)#.astype(np.float32)
Y_valid = m[m.date_block_num == 33][target].astype(np.float32)

model, err, importances, features, gain_counter, thresh_counter, thresh_counter2, best_splits, best_splits2 = utils_rf.feature_statistics(X_train, Y_train, X_valid, Y_valid, n_estimators=10, clip_low=0, clip_high=20)
pd.Series(importances).sort_values(ascending=False).round(4).to_csv('./tmp.csv')

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

i = pd.Series(importances).sort_values(ascending=False)
bs= pd.Series(score_drop).sort_values(ascending=False)

ir = pd.Series({f: i / len(features) for i, f in enumerate(i.index)}).reindex(bs.index[bs <= 0])
bsr = pd.Series({f: i / len(features) for i, f in enumerate(bs.index)}).reindex(bs.index[bs <= 0])
d = pd.concat([ir, bsr], axis=1)
print(d.loc[d[0] > bsr.min()])
print(d.loc[d[0] > bsr.min()].index)
print(bs[d.loc[d[0] > bsr.min()].index])

def get_splits(thresh_counter, f0, n=4):
    bins0, edges0 = np.histogram(thresh_counter[f0])
    idx = np.argsort(bins0)[-n:]
    res = np.array([edges0[i:i+1].mean() for i in idx])
    res.sort()
    return res

# ('date_item_avg_item_cnt_lag_1', 'si_std_item_131_lag_1'),
#  ('date_item_avg_item_cnt_lag_1', 'si_avg_item_121_lag_1'),
#  ('item_cnt_month_lag_1', 'si_avg_item_122_lag_1')

# [('item_cnt_month_lag_2', 'si_dist_55_lag_1'),
#  ('si_dist_56_lag_1', 'si_max_item_127_lag_1'),
#  ('item_cnt_month_lag_2', 'si_dist_45_lag_1')]

# [('item_cnt_month_lag_1', 'si_dist_56_lag_1'),
#  ('si_dist_45_lag_1', 'si_std_item_131_lag_1')]
matrix['item_cnt_month'] = matrix['item_cnt_month'].astype(np.float32)
for (f0, f1), _ in tqdm(best_splits2[:2]):
    print(f0, f1)
    mean_enc_name = '_'.join([f0, f1, 'avg', 'item'])
    bins0 = [-np.inf] + list(get_splits(thresh_counter, f0)) + [np.inf]
    bins1 = [-np.inf] + list(get_splits(thresh_counter, f1)) + [np.inf]
    matrix['f0'] = pd.cut(matrix[f0], bins0)
    matrix['f1'] = pd.cut(matrix[f1], bins1)
    group_cols = ['f0', 'f1']
    encoded_feature = utils.expanding_mean_encoding(matrix, 'item_cnt_month', group_cols)
    matrix.drop(group_cols, axis=1, inplace=True)
    matrix[mean_enc_name] = encoded_feature
    matrix = utils.lag_feature(matrix, [1], mean_enc_name)
    matrix.drop(mean_enc_name, axis=1, inplace=True)

matrix.to_pickle('../data/tmp9.pkl')

#          tr       val
# 0  0.796383  0.944961
# 1  0.757731  0.898915
# 2  0.744215  0.871710
# 3  0.740088  0.863658
# 4  0.738156  0.864555
# 5  0.737994  0.864851
# 6  0.732803  0.860529
# 7  0.731292  0.858797
# 8  0.727715  0.850789
# 9  0.729209  0.853500
#          tr       val
# 0  0.769977  0.881551
# 1  0.755038  0.864187
# 2  0.733014  0.847705
# 3  0.725533  0.840980
# 4  0.722480  0.837231
# 5  0.726901  0.842928
# 6  0.726254  0.841193
# 7  0.729352  0.843993
# 8  0.726612  0.841903
# 9  0.725546  0.841934
#          tr       val
# 0  0.773610  0.910777
# 1  0.746056  0.867055
# 2  0.735736  0.861412
# 3  0.727129  0.848411
# 4  0.723513  0.848254
# 5  0.719967  0.843480
# 6  0.717665  0.840188
# 7  0.722007  0.842438
# 8  0.725094  0.845591
# 9  0.723701  0.842954
#         tr       val
# 0  0.830903  0.936611
# 1  0.778993  0.893162
# 2  0.749806  0.861103
# 3  0.749833  0.861822
# 4  0.745060  0.857636
# 5  0.740251  0.856785
# 6  0.739904  0.853018
# 7  0.734573  0.850511
# 8  0.733597  0.851151
# 9  0.734809  0.852695
#          tr       val
# 0  0.817292  0.960985
# 1  0.778535  0.917822
# 2  0.750385  0.884202
# 3  0.736813  0.868320
# 4  0.738127  0.867078
# 5  0.741515  0.868060
# 6  0.736981  0.865310
# 7  0.738017  0.866742
# 8  0.737219  0.863972
# 9  0.734140  0.859083
#          tr       val
# 0  0.786021  0.964622
# 1  0.749125  0.923703
# 2  0.741474  0.888429
# 3  0.737339  0.869374
# 4  0.730340  0.859589
# 5  0.730619  0.861659
# 6  0.722235  0.849430
# 7  0.720989  0.847964
# 8  0.718977  0.843580
# 9  0.721521  0.844809
#          tr       val
# 0  0.800063  0.912185
# 1  0.767611  0.889948
# 2  0.753351  0.873780
# 3  0.740731  0.865787
# 4  0.742910  0.867276
# 5  0.733577  0.854114
# 6  0.729546  0.853658
# 7  0.730772  0.856088
# 8  0.730440  0.858762
# 9  0.728235  0.856137
#          tr       val
# 0  0.814549  1.072330
# 1  0.788626  0.963885
# 2  0.773217  0.923042
# 3  0.769386  0.908030
# 4  0.765758  0.896930
# 5  0.758945  0.889068
# 6  0.754874  0.882988
# 7  0.750391  0.876366
# 8  0.745795  0.866530
# 9  0.740447  0.862951
#          tr       val
# 0  0.771048  0.949909
# 1  0.750845  0.898636
# 2  0.724798  0.868284
# 3  0.729631  0.867237
# 4  0.721034  0.852414
# 5  0.718699  0.845058
# 6  0.722388  0.847395
# 7  0.724067  0.850001
# 8  0.724046  0.849073
# 9  0.723519  0.850462
#          tr       val
# 0  0.825703  0.938051
# 1  0.756497  0.877114
# 2  0.740692  0.865294
# 3  0.741675  0.867636
# 4  0.743392  0.870329
# 5  0.741710  0.868180
# 6  0.735516  0.856362
# 7  0.737907  0.858529
# 8  0.738820  0.859046
# 9  0.735237  0.855032
#          tr       val
# 0  0.822025  0.938730
# 1  0.779221  0.900359
# 2  0.755290  0.874408
# 3  0.750252  0.869480
# 4  0.743618  0.862883
# 5  0.741403  0.864679
# 6  0.739013  0.862107
# 7  0.738923  0.862940
# 8  0.735731  0.860968
# 9  0.737070  0.862263
#          tr       val
# 0  0.812326  0.936501
# 1  0.767968  0.890555
# 2  0.735866  0.856987
# 3  0.738337  0.857931
# 4  0.738516  0.858394
# 5  0.734936  0.856411
# 6  0.730585  0.857196
# 7  0.730226  0.855592
# 8  0.728735  0.853518
# 9  0.727150  0.853076
#          tr       val
# 0  0.802481  0.934797
# 1  0.771351  0.905830
# 2  0.760862  0.891543
# 3  0.750068  0.876964
# 4  0.751975  0.880001
# 5  0.740751  0.865931
# 6  0.741602  0.868424
# 7  0.736521  0.865943
# 8  0.734110  0.862143
# 9  0.733059  0.859066
#          tr       val
# 0  0.796739  0.939213
# 1  0.764756  0.908864
# 2  0.766424  0.892994
# 3  0.762298  0.888259
# 4  0.755962  0.876816
# 5  0.756913  0.876159
# 6  0.756892  0.877304
# 7  0.754463  0.873252
# 8  0.752288  0.869281
# 9  0.749994  0.866930
#          tr       val
# 0  0.790994  0.937722
# 1  0.769116  0.897050
# 2  0.749487  0.864139
# 3  0.744877  0.856296
# 4  0.744864  0.859200
# 5  0.743044  0.858358
# 6  0.743523  0.858751
# 7  0.741489  0.850456
# 8  0.741345  0.852217
# 9  0.740450  0.852075
#          tr       val
# 0  0.804933  0.931076
# 1  0.772319  0.907505
# 2  0.766800  0.893068
# 3  0.751605  0.877660
# 4  0.748445  0.872896
# 5  0.741421  0.863391
# 6  0.737062  0.860781
# 7  0.737219  0.861256
# 8  0.736966  0.863616
# 9  0.739140  0.864862
#          tr       val
# 0  0.806915  0.978442
# 1  0.750461  0.900852
# 2  0.739126  0.870220
# 3  0.742194  0.867775
# 4  0.742536  0.866510
# 5  0.737743  0.866852
# 6  0.737843  0.861926
# 7  0.739351  0.864201
# 8  0.738280  0.864117
# 9  0.735511  0.862084

pkl_name = 'data_2020-06-23_01.pkl'
# matrix = utils.reduce_mem_usage(matrix)
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
drop_cols += ['si_avg_item_119_lag_1', 'si_dist_109_lag_1', 'si_dist_32_lag_1', 'si_dist_86_lag_1', 'si_dist_111_lag_1', 'si_dist_11_lag_1', 'si_dist_43_lag_1', 'si_dist_22_lag_1', 'si_dist_87_lag_1', 'si_dist_38_lag_1', 'si_dist_98_lag_1', 'si_dist_106_lag_1', 'si_dist_13_lag_1', 'si_dist_21_lag_1', 'si_dist_110_lag_1', 'si_dist_42_lag_1', 'si_dist_39_lag_1', 'si_dist_35_lag_1', 'si_dist_17_lag_1', 'si_dist_16_lag_1', 'si_dist_26_lag_1', 'si_dist_18_lag_1', 'si_std_item_134_lag_1', 'si_dist_41_lag_1', 'si_dist_80_lag_1', 'si_dist_37_lag_1', ]
drop_cols += ['si_dist_10_lag_1', 'si_dist_63_lag_1', 'si_dist_97_lag_1', 'si_dist_25_lag_1', 'si_dist_89_lag_1', 'si_dist_95_lag_1', 'si_dist_40_lag_1', 'si_min_item_125_lag_1', ]
drop_cols += ['item_cnt_month_lag_1_item_cnt_month_lag_2_avg_item_lag_12', 'si_dist_29_lag_1', 'si_dist_20_lag_1', 'si_dist_14_lag_1', 'si_min_item_126_lag_1', 'si_dist_105_lag_1', 'si_dist_79_lag_1', 'si_dist_31_lag_1', 'si_dist_30_lag_1', 'si_dist_28_lag_1', 'si_dist_27_lag_1', 'si_dist_19_lag_1', 'si_dist_7_lag_1', 'si_dist_6_lag_1', 'si_dist_5_lag_1', 'si_dist_4_lag_1', 'si_dist_1_lag_1', 'si_dist_8_lag_1', 'si_dist_9_lag_1', 'si_dist_94_lag_1', 'si_dist_15_lag_1']
drop_cols += ['si_min_item_124_lag_1', 'si_dist_104_lag_1', 'si_dist_64_lag_1', 'si_dist_36_lag_1', 'si_dist_2_lag_1', 'si_dist_3_lag_1', 'si_dist_24_lag_1', 'si_dist_65_lag_1']
drop_cols += ['date_item_avg_item_cnt_lag_12', 'si_dist_23_lag_1', 'si_dist_113_lag_1', 'days']
drop_cols += ['cat_avg_item_lag_12', 'si_dist_93_lag_1', 'si_dist_12_lag_1']
drop_cols += ['si_dist_62_lag_1', 'si_dist_83_lag_1', 'si_dist_81_lag_1']
drop_cols += ['si_dist_102_lag_1', 'date_shop_1_avg_item_cnt_lag_1', 'si_min_item_123_lag_1']
drop_cols += ['item_avg_item_lag_12', 'si_dist_91_lag_1', 'si_dist_96_lag_1']
drop_cols += ['si_dist_82_lag_1']
drop_cols += ['si_dist_72_lag_1', 'si_dist_108_lag_1']
drop_cols += ['si_dist_60_lag_1', 'si_dist_69_lag_1']
drop_cols += ['si_dist_50_lag_1', 'si_dist_101_lag_1', 'date_cat_avg_item_cnt_lag_12']
drop_cols += ['si_dist_70_lag_1']
drop_cols += ['si_dist_45_lag_1', 'si_avg_item_121_lag_1', 'si_dist_56_lag_1', 'si_max_item_129_lag_1', 'item_avg_item_lag_1', 'si_dist_75_lag_1', 'si_avg_item_120_lag_1', 'si_std_item_132_lag_1', 'item_cnt_month_lag_2_si_dist_55_lag_1_avg_item_lag_1', 'date_shop_2_avg_item_cnt_lag_1', 'si_dist_57_lag_1', 'si_dist_78_lag_1', 'si_max_item_128_lag_1', 'date_shop_0_avg_item_cnt_lag_1', 'si_dist_49_lag_1', 'item_cnt_month_lag_2_si_dist_45_lag_1_avg_item_lag_1', 'si_dist_54_lag_1', 'si_dist_76_lag_1', 'si_dist_77_lag_1', 'si_dist_74_lag_1', 'si_dist_55_lag_1', 'si_dist_71_lag_1', 'si_dist_68_lag_1', 'si_dist_59_lag_1', 'si_dist_51_lag_1', 'si_dist_47_lag_1', 'si_dist_73_lag_1', 'si_dist_84_lag_1', 'si_dist_92_lag_1', 'si_dist_88_lag_1', 'si_dist_56_lag_1_si_max_item_127_lag_1_avg_item_lag_1', 'date_item_avg_item_cnt_lag_1_si_avg_item_121_lag_1_avg_item_lag_1', 'date_item_avg_item_cnt_lag_1_si_std_item_131_lag_1_avg_item_lag_1', 'si_std_item_133_lag_1', 'si_std_item_131_lag_1', 'si_avg_item_122_lag_1', 'si_dist_118_lag_1', 'si_dist_117_lag_1', 'si_dist_116_lag_1', 'si_dist_115_lag_1', 'si_dist_114_lag_1', 'si_dist_112_lag_1', 'si_dist_107_lag_1', 'si_dist_103_lag_1', 'si_dist_100_lag_1', 'si_dist_99_lag_1', 'si_dist_90_lag_1', 'si_dist_85_lag_1', 'month', 'si_dist_45_lag_1_si_std_item_131_lag_1_avg_item_lag_1', 'date_shop_avg_item_cnt_lag_12', 'date_shop_avg_item_cnt_lag_1', 'delta_revenue_lag_1', 'cat_shop_avg_item_lag_12', 'date_shop_avg_item_cnt_lag_2', 'shop_avg_item_lag_12', 'date_shop_avg_item_cnt_lag_3', 'si_dist_53_lag_1', 'si_dist_52_lag_1', 'date_item_avg_item_cnt_lag_3', 'item_cnt_month_lag_1_si_avg_item_122_lag_1_avg_item_lag_1', ]
data.drop(drop_cols, axis=1, inplace=True)
data.info()

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
    early_stopping_rounds = 10)

time.time() - ts

pd.Series({f: v for f, v in zip(X_train.columns, model.feature_importances_)}).sort_values().to_csv('./tmp3.csv')


# for max_tree in range(20, 41, 2):
#     y_train = model.predict(X_train, ntree_limit=max_tree)
#     y_valid = model.predict(X_valid, ntree_limit=max_tree)
#     terr = mean_squared_error(Y_train.values, y_train.clip(0, 20))
#     verr = mean_squared_error(Y_valid.values, y_valid.clip(0, 20))
#     print('%s: train-rmse=%.4f valid-rmse=%.4f' % (max_tree, terr, verr))

#     Y_test = model.predict(X_test, ntree_limit=max_tree).clip(0, 20)
#     submission = pd.DataFrame({
#         "ID": test.index, 
#         "item_cnt_month": Y_test
#     })
#     submission.to_csv('../data/xgb_submission_2020-06-19_03_ntree=%s.csv' % max_tree, index=False)

import pickle
pickle.dump(model, open('../data/model_xgb_2020-06-23_02.pkl', "wb"))

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
submission.to_csv('../data/xgb_submission_2020-06-23_02.csv', index=False)
pd.Series(Y_pred).to_csv('../data/xgb_submission_2020-06-23_02_valid.csv', index=False)



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
    

