# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 21:21:24 2020

@author: komarov
"""

# %cd "~/kaggle/competitive-data-science-predict-future-sales/src"
# !ls -lah 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product, chain
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
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


cats = pd.read_csv('../data/item_categories.csv')
items = pd.read_csv('../data/items.csv')
train = pd.read_csv('../data/sales_train1.csv')
train.date = train.date.astype('str')
# train.to_csv('../data/sales_train1.csv', index=False)
sample_sub = pd.read_csv('../data/sample_submission.csv')
shops = pd.read_csv('../data/shops.csv')
test = pd.read_csv('../data/test.csv')

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
    
    return extractor(x).numpy().round(2)


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


def rolling_mean_encoding(all_data, target, group_cols, global_mean=None):
    # groupby the and calculate the rolling mean on the previous values
    
    if global_mean is None:
        global_mean = all_data[label_col].mean()
    
    encoded_feature = (all_data.groupby(group_cols)[target].cumsum() - 
                       all_data[target]) / all_data.groupby(group_cols).cumcount()
    encoded_feature.fillna(global_mean, inplace=True)
    encoded_feature = encoded_feature.loc[all_data.index]
    
    # You will need to compute correlation like that
    corr = np.corrcoef(all_data[target].values, encoded_feature)[0][1]
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
        encoded_feature.iloc[val_ind] = x_val[group_cols].reset_index().\
                                                          merge(map_fn, on=group_cols, how='left').\
                                                          drop(group_cols, axis=1).set_index('index')[target]
    
    encoded_feature.fillna(global_mean, inplace=True)
    # You will need to compute correlation like that
    corr = np.corrcoef(all_data[target].values, encoded_feature)[0][1]
    print(corr)
    return encoded_feature


items.item_name = [norm_text(x) for x in items.item_name.values]
shops.shop_name = [norm_text(x) for x in shops.shop_name.values]
cats.item_category_name = [norm_text(x) for x in cats.item_category_name.values]

# D = 4 seems mapping different items to same vector too often
# D = 5 starts to distinguish
item_features = text_feature_to_vec(items.item_name.values, D=5, epochs=3)
show_neighbors(items.item_name.values, item_features)

shop_features = text_feature_to_vec(shops.shop_name.values, D=3, epochs=1000)
show_neighbors(shops.shop_name.values, shop_features)

cat_features = text_feature_to_vec(cats.item_category_name.values, D=3, epochs=1000)
show_neighbors(cats.item_category_name.values, cat_features)

# borrowed from here and modified https://www.kaggle.com/dlarionov/feature-engineering-xgboost
def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
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

train['revenue'] = train['item_price'] *  train['item_cnt_day']

# calculating target for training: monthly sales per (shop_id, item_id)
ts = time.time()
group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=cols, how='left')
matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .fillna(0)
                                .clip(0,20) # NB clip target here
                                .astype(np.float16))
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

def lag_feature(df, lags, col):
    tmp = df[['date_block_num', 'shop_id', 'item_id', col]]
    for i in tqdm(lags):
        shifted = tmp.copy()
        shifted.columns = ['date_block_num', 'shop_id', 'item_id', col + '_lag_' + str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return df

ts = time.time()
matrix = lag_feature(matrix, [1, 2, 3, 6, 12], 'item_cnt_month')
time.time() - ts

matrix.to_pickle('../data/tmp.pkl')
matrix = pd.read_pickle('../data/tmp.pkl')

from sklearn.ensemble import RandomForestRegressor

# for c in matrix.columns:
#     if isinstance(matrix[c], np.floating):
#         matrix[c] = matrix[c].astype(np.float32)

n_estimators = 50
model = RandomForestRegressor(max_depth=10, random_state=123, n_estimators=n_estimators, n_jobs=-1)
m = matrix.drop(['ID', 'shop_id', 'shop_name', 'item_name', 'item_id', 'item_category_name', 'item_category_id'], axis=1)
m = m[m.date_block_num > 11]
m.fillna(0, inplace=True)
X_train = m[m.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = m[m.date_block_num < 33]['item_cnt_month']
X_valid = m[m.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = m[m.date_block_num == 33]['item_cnt_month']
model.fit(X_train, Y_train)

err_val = {}
err_tr = {}
pred_val = None
pred_tr = None
for i in tqdm(range(n_estimators)):
    if pred_tr is None:
        pred_val =  model.estimators_[i].predict(X_valid)
        pred_tr  =  model.estimators_[i].predict(X_train)
    else:
        pred_val = (pred_val * i +  model.estimators_[i].predict(X_valid)) / (i + 1)
        pred_tr =  (pred_tr  * i +  model.estimators_[i].predict(X_train)) / (i + 1)
    err_val[i] = mean_squared_error(Y_valid.values.astype('float32'), pred_val.clip(0, 20))
    err_tr[i]  = mean_squared_error(Y_train.values.astype('float32'), pred_tr.clip(0, 20))

pd.DataFrame({'tr': err_tr, 'val': err_val}).plot()

print('train-rmse=%.4f valid-rmse=%.4f' % (terr, verr))


for agg_cols, agg_names, lags in tqdm(
        [([], [], [1]),
         (['item_id'], ['item'], [1, 2, 3, 6, 12]),
         (['shop_id'], ['shop'], [1, 2, 3, 6, 12]),
         (['item_category_id'], ['cat'], [1, 12]),
         (['shop_id', 'item_category_id'], ['shop', 'cat'], [1, 12]),
         (['shop_0'], ['shop_0'], [1]),
         (['shop_1'], ['shop_1'], [1]),
         (['shop_2'], ['shop_2'], [1]),
         (['cat_0'], ['cat_0'], [1]),
         (['cat_1'], ['cat_1'], [1]),
         (['cat_2'], ['cat_2'], [1]),
         (['shop_id', 'cat_0'], ['shop', 'cat_0'], [1]),
         (['shop_id', 'cat_1'], ['shop', 'cat_1'], [1]),
         (['shop_id', 'cat_2'], ['shop', 'cat_2'], [1]),
         (['item_id', 'shop_0'], ['item', 'shop_0'], [1]),
         (['item_id', 'shop_1'], ['item', 'shop_1'], [1]),
         (['item_id', 'shop_2'], ['item', 'shop_2'], [1]),]):
    ts = time.time()
    mean_enc_name = '_'.join(['date'] + agg_names + ['avg', 'item_cnt'])
    group = matrix.groupby(['date_block_num'] + agg_cols).agg({'item_cnt_month': ['mean']})
    group.columns = [mean_enc_name]
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num'] + agg_cols, how='left')
    matrix[mean_enc_name] = matrix[mean_enc_name].astype(np.float16)
    matrix = lag_feature(matrix, lags, mean_enc_name)
    matrix.drop([mean_enc_name], axis=1, inplace=True)
    if any(c.endswith('_x') for c in matrix.columns):
        break
    print(time.time() - ts)


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


matrix.info()
for c in matrix.columns:
    if matrix[c].dtype == 'float32':
        print(c)
        matrix[c] = matrix[c].astype('float16')

for c in matrix.columns:
    if matrix[c].isna().any():
        print(c)

pkl_name = 'data_2020-06-16_01.pkl'
# matrix.to_pickle('../data/%s' % pkl_name)

del matrix
del group
del items
del shops
del cats
del train
# leave test for submission
gc.collect()


%cd "~/kaggle/competitive-data-science-predict-future-sales/src"

import pandas as pd
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
data.drop(['ID', 'item_category_name', 'item_name', 'shop_name'], axis=1, inplace=True)
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

# [62]	validation_0-rmse:0.79811	validation_1-rmse:0.89230

ts = time.time()

model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8,
    eta=0.01,
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 10)

time.time() - ts

from collections import Counter
from sklearn.metrics import confusion_matrix
Y_pred = model.predict(X_valid).clip(0, 20)
sorted(Counter(Y_pred.round(0)).items())
sorted(Counter(Y_valid).items())
cm = confusion_matrix(Y_valid.values.astype('int8'), Y_pred.round(0).astype('int8'))
pd.DataFrame(cm).to_csv('../data/errors.xgb.best.csv')

Y_test = model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('../data/xgb_submission_2020-06-16_01.csv', index=False)

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



encoded_feature = kfold_mean_encoding(all_data, 'target', ['item_id'], k=5, global_mean=0.3343)
encoded_feature = rolling_mean_encoding(all_data, 'target', ['item_id'], global_mean=0.3343)

%timeit -n1 -r3 rolling_mean_encoding(all_data, 'target', ['item_id'], global_mean=0.3343)
%timeit -n1 -r3 kfold_mean_encoding(all_data, 'target', ['item_id'], k=5, global_mean=0.3343)


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool

import numpy as np


class NearestNeighborsFeats(BaseEstimator, ClassifierMixin):
    '''
        This class should implement KNN features extraction 
    '''
    def __init__(self, n_jobs, k_list, metric, n_classes=None, n_neighbors=None, eps=1e-6):
        self.n_jobs = n_jobs
        self.k_list = k_list
        self.metric = metric
        
        if n_neighbors is None:
            self.n_neighbors = max(k_list) 
        else:
            self.n_neighbors = n_neighbors
            
        self.eps = eps        
        self.n_classes_ = n_classes
    
    def fit(self, X, y):
        '''
            Set's up the train set and self.NN object
        '''
        # Create a NearestNeighbors (NN) object. We will use it in `predict` function 
        self.NN = NearestNeighbors(n_neighbors=max(self.k_list), 
                                      metric=self.metric, 
                                      n_jobs=1, 
                                      algorithm='brute' if self.metric=='cosine' else 'auto')
        self.NN.fit(X)
        
        # Store labels 
        self.y_train = y
        
        # Save how many classes we have
        if self.n_classes_ is not None:
            self.n_classes = self.n_classes_
        else:
            self.n_classes = np.unique(y).shape[0]
        
    def predict(self, X):       
        '''
            Produces KNN features for every object of a dataset X
        '''
        if self.n_jobs == 1:
            test_feats = []
            for i in range(X.shape[0]):
                test_feats.append(self.get_features_for_one(X[i:i+1]))
        else:
            '''
                 *Make it parallel*
                     Number of threads should be controlled by `self.n_jobs`  
                     
                     
                     You can use whatever you want to do it
                     For Python 3 the simplest option would be to use 
                     `multiprocessing.Pool` (but don't use `multiprocessing.dummy.Pool` here)
                     You may try use `joblib` but you will most likely encounter an error, 
                     that you will need to google up (and eventually it will work slowly)
                     
                     For Python 2 I also suggest using `multiprocessing.Pool` 
                     You will need to use a hint from this blog 
                     http://qingkaikong.blogspot.ru/2016/12/python-parallel-method-in-class.html
                     I could not get `joblib` working at all for this code 
                     (but in general `joblib` is very convenient)
                     
            '''
            
            # YOUR CODE GOES HERE
            # test_feats =  # YOUR CODE GOES HERE
            # YOUR CODE GOES HERE
            
            # Comment out this line once you implement the code
            # assert False, 'You need to implement it for n_jobs > 1'
            with Pool(self.n_jobs) as p:
                gen = (X[i:i + 1] for i in range(X.shape[0]))
                test_feats = p.map(self.get_features_for_one, gen)
            
        return np.vstack(test_feats)
        
        
    def get_features_for_one(self, x):
        '''
            Computes KNN features for a single object `x`
        '''

        NN_output = self.NN.kneighbors(x)
        
        # Vector of size `n_neighbors`
        # Stores indices of the neighbors
        neighs = NN_output[1][0]
        
        # Vector of size `n_neighbors`
        # Stores distances to corresponding neighbors
        neighs_dist = NN_output[0][0] 

        # Vector of size `n_neighbors`
        # Stores labels of corresponding neighbors
        neighs_y = self.y_train[neighs] 
        
        ## ========================================== ##
        ##              YOUR CODE BELOW
        ## ========================================== ##
        
        # We will accumulate the computed features here
        # Eventually it will be a list of lists or np.arrays
        # and we will use np.hstack to concatenate those
        return_list = [] 
        
        
        ''' 
            1. Fraction of objects of every class.
               It is basically a KNNСlassifiers predictions.

               Take a look at `np.bincount` function, it can be very helpful
               Note that the values should sum up to one
        '''
        for k in self.k_list:
            classes_in_neighs = np.bincount(neighs_y[:k], minlength=self.n_classes)
            feats = classes_in_neighs / classes_in_neighs.sum()
            assert len(feats) == self.n_classes
            return_list += [feats]
        
        '''
            2. Same label streak: the largest number N, 
               such that N nearest neighbors have the same label.
               
               What can help you: `np.where`
        '''
        feats = np.array([len(neighs)])
        non_class_locs = np.where(neighs_y != neighs_y[0])[0]
        if len(non_class_locs) > 0:
            feats[0] = non_class_locs[0]
        
        assert len(feats) == 1
        return_list += [feats]
        
        '''
            3. Minimum distance to objects of each class
               Find the first instance of a class and take its distance as features.
               
               If there are no neighboring objects of some classes, 
               Then set distance to that class to be 999.

               `np.where` might be helpful
        '''
        feats = []
        for c in range(self.n_classes):
            class_locs = np.where(neighs_y == c)[0]
            if len(class_locs) > 0:
                feats.append( neighs_dist[class_locs[0]] )
            else:
                feats.append(999)
        
        assert len(feats) == self.n_classes
        return_list += [feats]
        
        '''
            4. Minimum *normalized* distance to objects of each class
               As 3. but we normalize (divide) the distances
               by the distance to the closest neighbor.
               
               If there are no neighboring objects of some classes, 
               Then set distance to that class to be 999.
               
               Do not forget to add self.eps to denominator.
        '''
        feats = []
        for c in range(self.n_classes):
            class_locs = np.where(neighs_y == c)[0]
            if len(class_locs) > 0:
                feats.append( neighs_dist[class_locs[0]] / (neighs_dist[0] + self.eps) )
            else:
                feats.append(999)
        
        assert len(feats) == self.n_classes
        return_list += [feats]
        
        '''
            5. 
               5.1 Distance to Kth neighbor
                   Think of this as of quantiles of a distribution
               5.2 Distance to Kth neighbor normalized by 
                   distance to the first neighbor
               
               feat_51, feat_52 are answers to 5.1. and 5.2.
               should be scalars
               
               Do not forget to add self.eps to denominator.
        '''
        for k in self.k_list:
            
            feat_51 = neighs_dist[k - 1]
            feat_52 = neighs_dist[k - 1] / (neighs_dist[0] + self.eps)
            
            return_list += [[feat_51, feat_52]]
        
        '''
            6. Mean distance to neighbors of each class for each K from `k_list` 
                   For each class select the neighbors of that class among K nearest neighbors 
                   and compute the average distance to those objects
                   
                   If there are no objects of a certain class among K neighbors, set mean distance to 999
                   
               You can use `np.bincount` with appropriate weights
               Don't forget, that if you divide by something, 
               You need to add `self.eps` to denominator.
        '''
        for k in self.k_list:
            feats = []
            for c in range(self.n_classes):
                class_locs = np.where(neighs_y[:k] == c)[0]
                if len(class_locs) > 0:
                    feats.append( neighs_dist[class_locs].mean() )
                else:
                    feats.append(999)
            
            assert len(feats) == self.n_classes
            return_list += [feats]
        
        
        # merge
        knn_feats = np.hstack(return_list)
        
        assert knn_feats.shape == (239,) or knn_feats.shape == (239, 1)
        return knn_feats
    
    
for metric in ['minkowski', 'cosine']:
    print (metric)
    
    # Create instance of our KNN feature extractor
    NNF = NearestNeighborsFeats(n_jobs=4, k_list=k_list, metric=metric)
    
    # Fit on train set
    NNF.fit(X, Y)

    # Get features for test
    test_knn_feats = NNF.predict(X_test)
    
    # Dump the features to disk
    np.save('data/knn_feats_%s_test.npy' % metric , test_knn_feats)
    

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
    
