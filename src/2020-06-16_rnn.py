

pkl_name = 'data_2020-06-15_01.pkl'

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
# from xgboost import XGBRegressor
# from xgboost import plot_importance
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

X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

del data
gc.collect()



X_train = X_train.values
Y_train = Y_train.values

XY_train = np.hstack([X_train.values, Y_train.values[:, np.newaxis]])

shop_item = X_train[['shop_id', 'item_id']].drop_duplicates()

# Create training examples / targets
si = tf.data.Dataset.from_tensor_slices(shop_item.values)

seq_length = 21
batch_size = 10
sis = si.batch(batch_size, drop_remainder=True)

def get_shop_item_seq(shop_item_pairs):
    masks = []
    for s, i in shop_item_pairs:
        mask.append((X_train.shop_id == s) & (X_train.item_id == i))
    
    xs = X_train[]
    ys = xyseq[:, -1]
    return xs, ys
    
dataset = sis.map(get_shop_item_seq)


for inp, out in dataset.take(1):
    print(repr(inp))
for inp, out in dataset.take(1):
    print(''.join([idx2char[i] for i in out]))


# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)






