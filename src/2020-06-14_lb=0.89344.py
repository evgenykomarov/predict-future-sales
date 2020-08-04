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

cats = pd.read_csv('../data/item_categories.csv')
items = pd.read_csv('../data/items.csv')
# train = pd.read_csv('../data/sales_train1.csv')
train = pd.read_csv('../data/sales_train.csv')
train.date = train.date.astype('str')
# train.to_csv('../data/sales_train1.csv', index=False)
# sample_sub = pd.read_csv('../data/sample_submission.csv')
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
    nei_items = item_names[neigh.kneighbors(item_features[[i], :])[1]]
    nei_dists = neigh.kneighbors(item_features[[i], :])[0]
    print(item_names[i])
    print(nei_items)
    print(nei_dists)

def norm_text(x):
    # removes non-alpha numeric
    # lower case
    # strip
    # replace many spaces with 1
    import re
    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x.lower())
    x = re.sub('\s+', ' ', x)
    return x.strip()

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

import time
import sys
import gc
import pickle
sys.version_info

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

# for c in matrix.columns:
#     if matrix[c].dtype == 'float16':
#         matrix[c] = matrix[c].astype(np.float32)
# matrix.drop(['item_name', 'item_category_name', 'shop_name'], axis=1).to_parquet('../data_tmp/tmp.snappy')

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

for agg_cols, agg_names, lags in tqdm(
        [([], [], [1]),
         (['item_id'], ['item'], [1, 2, 3, 6, 12]),
         (['shop_id'], ['shop'], [1, 2, 3, 6, 12]),
         (['item_category_id'], ['cat'], [1]),
         (['shop_id', 'item_category_id'], ['shop', 'cat'], [1]),
         (['shop_0'], ['shop_0'], [1]),
         (['shop_1'], ['shop_1'], [1]),
         (['shop_2'], ['shop_2'], [1]),
         (['cat_0'], ['cat_0'], [1]),
         (['cat_1'], ['cat_1'], [1]),
         (['cat_2'], ['cat_2'], [1]),
         # (['shop_id'], ['cat_0'], [1]),
         # (['shop_id'], ['cat_1'], [1]),
         # (['shop_id'], ['cat_2'], [1]),
         # (['item_id'], ['shop_0'], [1]),
         # (['item_id'], ['shop_1'], [1]),
         # (['item_id'], ['shop_2'], [1]),
         ]):
    ts = time.time()
    mean_enc_name = '_'.join(['date'] + agg_names + ['avg', 'item_cnt'])
    group = matrix.groupby(['date_block_num'] + agg_cols).agg({'item_cnt_month': ['mean']})
    group.columns = [mean_enc_name]
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num'] + agg_cols, how='left')
    matrix[mean_enc_name] = matrix[mean_enc_name].astype(np.float16)
    matrix = lag_feature(matrix, lags, mean_enc_name)
    matrix.drop([mean_enc_name], axis=1, inplace=True)
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
        matrix[c] = matrix[c].astype('float16')
# matrix.to_pickle('../data/data_2020-06-14_01_replica.pkl')
for c in matrix.columns:
    if matrix[c].dtype == 'float16' or matrix[c].dtype == 'float64':
        matrix[c] = matrix[c].astype('float32')
matrix.drop(['item_name', 'shop_name', 'item_category_name'], axis=1).to_parquet('../data_tmp/data_2020-06-14_01_.snappy')
matrix = pd.read_parquet('../data_tmp/data_2020-06-14_01_.snappy')

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
import seaborn as sns
import gc
import time


test = pd.read_csv('../data/test.csv')

data = pd.read_pickle('../data/data.pkl')
sorted(data.columns)
data.drop(['ID', 'item_category_name', 'item_name', 'shop_name'], axis=1, inplace=True)
data.info()

X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

del data
gc.collect()

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
pickle.dump(model, open('../data/model_xgb_submission_2020-06-14_01.pkl', "wb"))

time.time() - ts

Y_pred = model.predict(X_valid).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('../submission/xgb_submission_2020-06-14_01_replica.csv', index=False)
pd.Series(Y_pred).to_csv('../submission/xgb_submission_2020-06-14_01_replica_valid.csv', index=False)

plot_features(model, (10,14))

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

