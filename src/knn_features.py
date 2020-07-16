# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:45:48 2020

@author: komarov
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
from collections import defaultdict
import os
import gc
import numpy as np
from tqdm import tqdm
import pandas as pd

def rolling_window_data_generator(t, dates, win_size, date_col, cols, lag, X, Y, test=False):
    """
    return X and Y windowed to win_size and lagged lag from t

    X, Y = dataframes
    date_col = name of col with date
    dates = list of unique dates
    for lag=lag and t returns window
    [t - lag - win_size + 1, t - lag + 1)
    """
    if test:
        test_index = X[date_col].isin(dates[t: t + 1])
        x_test = X.loc[test_index, cols].values
        return x_test, X.index[np.where(test_index)[0]]
    else:
        win_dates = dates[t - win_size - lag + 1: t - lag + 1]
        train_index = X[date_col].isin(win_dates)
        x_train = X.loc[train_index, cols].values
        y_train = Y.loc[train_index].values
        return x_train, y_train


def rolling_window_range(t0, dates, win_size, lag):
    t_range = list(range(max(t0, win_size + lag - 1), len(dates)))
    return t_range


def fit_rolling_estimators_parallel(estimator, data_getter, t_range, **estimator_kwargs):
    """
    estimator.fit(X, y, i) method must return self and i. i used to identify the async parallel result
    data_generator_partial(t, test=False) should return (x_train, y_train) for each value t
    """
    estimators_fitted = {}

    def on_ret(retval):
        obj, obj_id = retval
        estimators_fitted[obj_id] = obj

    est_gen = (estimator(**estimator_kwargs) for t in t_range)
    data_gen = (data_getter(t, test=False) for t in t_range)

    p = Pool(estimator_kwargs['n_jobs'])
    for i, (est, (x_train, y_train)) in tqdm(enumerate(zip(est_gen, data_gen))):
        p.apply_async(est.fit, args=(x_train, y_train, i), callback=on_ret)
    p.close(); p.join()
    return [estimators_fitted[i] for i in range(len(estimators_fitted))]


def rolling_knn(matrix, t0=0, date_col='date_block_num',
                knn_prefix = 'item',
                knn_cols=['item_0', 'item_1', 'item_2', 'item_3', 'item_4'], # columns to find nearest neighbors for
                target_cols_short = ['item', 'date_num', 'price'], # short names for generating features names from target_cols
                target_cols=['item_cnt_month', 'item_first_sale', 'date_item_avg_item_price'], # columns to calculate knn averages
                win_size=3, lag=1,
                n_jobs=8, k_list=[10, 50, 100, 500], metric='minkowski', eps=1e-6, seed=123):

    estimator_kwargs = dict(n_jobs=n_jobs, k_list=k_list, metric=metric, eps=eps, seed=seed)
    estimator = NearestNeighborsRegFeats
    X = matrix[list( set([date_col]) | set(knn_cols) )]
    Y = matrix[target_cols]
    dates = sorted(matrix[date_col].unique())

    while os.path.isfile('../data_processed/knn_temp/knn_%s_win%s_lag%s_t_%02d.snappy' % (knn_prefix, win_size, lag, t0)):
        t0 += 1

    t_range = rolling_window_range(t0, dates, win_size, lag)
    data_gen = lambda t, test: rolling_window_data_generator(t, dates, win_size, date_col, knn_cols, lag, X, Y, test=test)
    estimators = fit_rolling_estimators_parallel(estimator, data_gen, t_range, **estimator_kwargs)

    if len(t_range) == 0:
        return
    columns = estimators[0].get_feature_names(knn_prefix, target_cols_short)
    encoded_feature = []
    for i, t in tqdm(enumerate(t_range)):
        fname = os.path.abspath('../data_processed/knn_temp/knn_%s_win%s_lag%s_t_%02d.snappy' % (knn_prefix, win_size, lag, t))
        print('saving results for %s win=%s lag=%s t=%s to %s' % (knn_prefix, win_size, lag, t, fname))
        x_test, test_index = data_gen(t, test=True)
        feats = estimators[i].predict(x_test)
        pd.DataFrame(feats, index=test_index, columns=columns).to_parquet(fname, compression='snappy')
        encoded_feature.append((feats, test_index))
    # start: 20:37
    feats = np.vstack([x for x, _ in encoded_feature])
    index = np.hstack([x for _, x in encoded_feature])
    res = pd.DataFrame(feats, index=index, columns=columns)
    return res


class NearestNeighborsRegFeatsByGroups(BaseEstimator, ClassifierMixin):
    """
    Approach:
    knn analog of group by averaging:
    for given X we find its unique rows X_unique
    and find k-neighbors on X_unique and then calculate
    knn averages on entries in X corresponding to neighbors
    from X_unique
    """
    def __init__(self, n_jobs, k_list, metric, eps=1e-6, seed=123):
        self.n_jobs = n_jobs
        self.k_list = k_list
        self.metric = metric
        self.n_neighbors = max(k_list)
        self.eps = eps
        self.seed = seed

    def fit(self, X, y, i=None):
        """
        Set's up the train set and self.NN object
        y_reg = regression values
        y = labels bins
        """
        # Create a NearestNeighbors (NN) object. We will use it in `predict` function
        # NN = NearestNeighbors(n_neighbors=7,
        #                               metric='minkowski',
        #                               n_jobs=8,
        #                               algorithm='brute')
        # knn_cols = ['item_0', 'item_1', 'item_2', 'item_3', 'item_4']
        # t = 22
        # date_col = 'date_block_num'
        # target = 'item_cnt_month'
        # X = matrix.loc[matrix[date_col] == t, knn_cols].values
        # Y = matrix.loc[matrix[date_col] == t, target].values

        # X_unique, loc_in_unique = np.unique(X, axis=0, return_inverse=True)
        # uniquei_to_all = defaultdict(list)
        # [uniquei_to_all[loc].append(index) for index, loc in enumerate(loc_in_unique)]
        # uniquei_to_all = {k: np.array(vs) for k, vs in uniquei_to_all.items()}
        # NN.fit(X_unique)
        # neighs_to_unique = NN.kneighbors(X_unique)[1]

        # unique_i = 0
        # neighs_in_unique = neighs_to_unique[unique_i]
        # indices_to_average = np.hstack(uniquei_to_all[nei] for nei in neighs_in_unique)
        # Y[uniquei_to_all[unique_i]].max()

        self.NN = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      metric=self.metric,
                                      n_jobs=1,
                                      algorithm='brute' if self.metric=='cosine' else 'auto')


        print('fitting nn for i=%s...' % i)

        self.X_unique, loc_in_unique = np.unique(X, axis=0, return_inverse=True)
        self.uniquei_to_all = defaultdict(list)
        [self.uniquei_to_all[loc].append(index) for index, loc in enumerate(loc_in_unique)]
        self.uniquei_to_all = {k: np.array(vs) for k, vs in self.uniquei_to_all.items()}

        self.NN.fit(self.X_unique)
        # self.neighs_to_unique = self.NN.kneighbors(self.X_unique)[1]
        self.y_train = y
        self.fns = ['mean', 'min', 'max', 'std']
        return self, i

    def predict(self, X=None, parallel_job_id=None):
        """
        KNN features for every object of a dataset X
        """
        X_unique, loc_in_unique = np.unique(X, axis=0, return_inverse=True)
        uniquei_to_all = defaultdict(list)
        [uniquei_to_all[loc].append(index) for index, loc in enumerate(loc_in_unique)]
        uniquei_to_all = {k: np.array(vs) for k, vs in uniquei_to_all.items()}
        # self.neighs_to_unique = self.NN.kneighbors(X_unique)[1]

        if self.n_jobs == 1:
            test_feats = []
            for i in range(X_unique.shape[0]):
                test_feats.append(self.get_features_for_one(X_unique[i:i+1]))
        else:
            with Pool(self.n_jobs) as p:
                gen = (X_unique[i:i+1] for i in range(X_unique.shape[0]))
                test_feats = p.map(self.get_features_for_one, gen)
        features_for_unique = np.vstack(test_feats)
        return_features = np.empty((X.shape[0], features_for_unique.shape[1]), dtype=np.float32)
        for i in range(features_for_unique.shape[0]):
            return_features[uniquei_to_all[i], :] = features_for_unique[i, :]

        return return_features, parallel_job_id

    def get_feature_names(self, knn_prefix, y_names):
        if len(y_names) > 1:
            assert len(y_names) == self.y_train.shape[1]
        else:
            assert len(y_names) == 1
            assert self.y_train.ndim == 1 or self.y_train.shape[1] == 1
        return ['%s_knn_%s_%s_%s' % (knn_prefix, k, fn, y_name) for y_name in y_names for k in self.k_list for fn in self.fns]

    def get_features_for_one(self, x_unique):
        """
        Computes KNN features for a single object x
        unique_i = index of row from self.X_unique
        """
        neighs_to_unique = self.NN.kneighbors(x_unique)[1][0]

        Y = self.y_train
        res = []
        for k in self.k_list:
            ns_indices = np.hstack([self.uniquei_to_all[nei] for nei in neighs_to_unique[:k]])
            res += [Y[ns_indices].mean(axis=0),
                    Y[ns_indices].min(axis=0),
                    Y[ns_indices].max(axis=0),
                    Y[ns_indices].std(axis=0)]

        if Y.ndim > 1:
            knn_feats = np.hstack(res).reshape(-1, Y.shape[1]).T.reshape(-1)  # first features for 1st col of Y then for 2nd etc
        else:
            knn_feats= np.hstack(res)

        return knn_feats


class NearestNeighborsRegFeats(BaseEstimator, ClassifierMixin):
    """
    This class should implement KNN features extraction
    """
    def __init__(self, n_jobs, k_list, metric, eps=1e-6, seed=123):
        self.n_jobs = n_jobs
        self.k_list = k_list
        self.metric = metric
        self.n_neighbors = max(k_list)
        self.eps = eps        
        self.seed = seed
        
    def fit(self, X, y, i=None):
        """
        Set's up the train set and self.NN object
        y_reg = regression values
        y = labels bins
        """
        # Create a NearestNeighbors (NN) object. We will use it in `predict` function 
        self.NN = NearestNeighbors(n_neighbors=self.n_neighbors, 
                                      metric=self.metric, 
                                      n_jobs=1, 
                                      algorithm='brute' if self.metric=='cosine' else 'auto')
        print('fitting nn for i=%s...' % i)
        self.NN.fit(X)
        self.y_train = y
        self.fns = ['mean', 'median', 'min', 'max']
        return self, i
        
    def predict(self, X):
        """
        KNN features for every object of a dataset X
        """
        if self.n_jobs == 1:
            test_feats = []
            for i in range(X.shape[0]):
                test_feats.append(self.get_features_for_one(X[i:i+1]))
        else:
            with Pool(self.n_jobs) as p:
                gen = (X[i:i + 1] for i in range(X.shape[0]))
                test_feats = p.map(self.get_features_for_one, gen)
            
        return np.vstack(test_feats)

    def get_feature_names(self, knn_prefix, y_names):
        if len(y_names) > 1:
            assert len(y_names) == self.y_train.shape[1]
        else:
            assert len(y_names) == 1
            assert self.y_train.ndim == 1 or self.y_train.shape[1] == 1
        return ['%s_knn_%s_%s_%s' % (knn_prefix, k, fn, y_name) for y_name in y_names for k in self.k_list for fn in self.fns]


    def get_features_for_one(self, x):
        """
        Computes KNN features for a single object x
        """
        NN_output = self.NN.kneighbors(x)
        neighs = NN_output[1][0]
        # neighs_dist = NN_output[0][0]
        Y = self.y_train
        res = []
        for k in self.k_list:
            ns = neighs[:k]
            res += [Y[ns].mean(axis=0), np.median(Y[ns], axis=0), Y[ns].min(axis=0), Y[ns].max(axis=0)]
        
        if Y.ndim > 1:
            knn_feats = np.hstack(res).reshape(-1, Y.shape[1]).T.reshape(-1) # first features for 1st col of Y then for 2nd etc
        else:
            knn_feats= np.hstack(res)        
        
        return knn_feats


class NearestNeighborsFeatsByGroups(BaseEstimator, ClassifierMixin):
    """
    This class should implement KNN features extraction
    """
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

    def fit(self, X, y_bins, y_regr):
        """
        Set's up the train set and self.NN object
        y_reg = regression values
        y = labels bins
        """
        # Create a NearestNeighbors (NN) object. We will use it in `predict` function
        self.NN = NearestNeighbors(n_neighbors=max(self.k_list),
                                      metric=self.metric,
                                      n_jobs=1,
                                      algorithm='brute' if self.metric=='cosine' else 'auto')
        self.X_unique, loc_in_unique = np.unique(X, axis=0, return_inverse=True)
        self.uniquei_to_all = defaultdict(list)
        [self.uniquei_to_all[loc].append(index) for index, loc in enumerate(loc_in_unique)]
        self.uniquei_to_all = {k: np.array(vs) for k, vs in self.uniquei_to_all.items()}

        self.NN.fit(self.X_unique)
        self.y_train = y_bins
        self.y_train_reg = y_regr

        # Save how many classes we have
        if self.n_classes_ is not None:
            self.n_classes = self.n_classes_
        else:
            self.n_classes = np.unique(y_bins).shape[0]

    def predict(self, X, parallel_job_id=None):
        """
        KNN features for every object of a dataset X
        """
        X_unique, loc_in_unique = np.unique(X, axis=0, return_inverse=True)
        uniquei_to_all = defaultdict(list)
        [uniquei_to_all[loc].append(index) for index, loc in enumerate(loc_in_unique)]
        uniquei_to_all = {k: np.array(vs) for k, vs in uniquei_to_all.items()}
        # self.neighs_to_unique = self.NN.kneighbors(X_unique)[1]

        if self.n_jobs == 1:
            test_feats = []
            for i in range(X_unique.shape[0]):
                test_feats.append(self.get_features_for_one(X_unique[i:i+1]))
        else:
            with Pool(self.n_jobs) as p:
                gen = (X_unique[i:i+1] for i in range(X_unique.shape[0]))
                test_feats = p.map(self.get_features_for_one, gen)
        features_for_unique = np.vstack(test_feats)
        return_features = np.empty((X.shape[0], features_for_unique.shape[1]), dtype=np.float32)
        for i in range(features_for_unique.shape[0]):
            return_features[uniquei_to_all[i], :] = features_for_unique[i, :]

        return return_features, parallel_job_id

    def get_features_for_one(self, x_unique):
        """
        Computes KNN features for a single object x
        """
        nn_output = self.NN.kneighbors(x_unique)
        # Vector of size `n_neighbors`
        # Stores indices of the neighbors
        neighs_to_unique = nn_output[1][0]
        # Vector of size `n_neighbors`
        # Stores distances to corresponding neighbors
        dists_to_unique = nn_output[0][0]
        # neighs_dist = NN_output[0][0]

        neighs_y = {}
        for k in self.k_list:
            ns_indices = np.hstack([self.uniquei_to_all[nei] for nei in neighs_to_unique[:k]])
            neighs_y[k] = self.y_train[ns_indices]
        neighs_dist = np.hstack([np.repeat(dist_to_nei, len(self.uniquei_to_all[nei]))
                                 for nei, dist_to_nei in zip(neighs_to_unique, dists_to_unique)])


        # Vector of size `n_neighbors`
        # Stores labels of corresponding neighbors
        # neighs_y = self.y_train[neighs]
        # neighs_y_reg = self.y_train_reg[neighs]

        return_list = []

        ''' 
        Fraction of objects of every class.
        '''
        for k in self.k_list:
            classes_in_neighs = np.bincount(neighs_y[k], minlength=self.n_classes)
            feats = classes_in_neighs / classes_in_neighs.sum()
            assert len(feats) == self.n_classes
            return_list += [feats]

        '''
        Same label streak: the largest number N, 
        such that N nearest neighbors have the same label.
        '''
        n_neighbors = len(neighs_to_unique)
        feats = n_neighbors
        non_class_locs = np.where(neighs_y[n_neighbors - 1] != neighs_y[n_neighbors - 1][0])[0]
        if len(non_class_locs) > 0:
            feats = non_class_locs[0]

        return_list += [feats]

        '''
        Minimum distance to objects of each class
        the first instance of a class and take its distance as features.
               
        If there are no neighboring objects of some classes, 
        Then set distance to that class to be 999.
        '''
        feats = []
        for c in range(self.n_classes):
            class_locs = np.where(neighs_y[n_neighbors - 1] == c)[0]
            if len(class_locs) > 0:
                feats.append(neighs_dist[class_locs[0]])
            else:
                feats.append(999)

        assert len(feats) == self.n_classes
        return_list += [feats]

        '''
        Minimum *normalized* distance to objects of each class
        divide the distances by the distance to the 2nd closest neighbor.
               
        If there are no neighboring objects of some classes, 
        Then set distance to that class to be 999.
        '''
        feats = []
        for c in range(self.n_classes):
            class_locs = np.where(neighs_y[n_neighbors - 1] == c)[0]
            if len(class_locs) > 0:
                feats.append(neighs_dist[class_locs[0]] / (neighs_dist[1] + self.eps))
            else:
                feats.append(999)

        assert len(feats) == self.n_classes
        return_list += [feats]

        '''
        Distance to Kth neighbor ("quantiles" of a distribution)
        Distance to Kth neighbor normalized by distance to the second neighbor
        
        '''
        for k in self.k_list:

            feat_51 = neighs_dist[k - 1]
            feat_52 = neighs_dist[k - 1] / (neighs_dist[1] + self.eps)

            return_list += [[feat_51, feat_52]]

        '''
        Mean distance to neighbors of each class for each K from `k_list` 
        For each class select the neighbors of that class among K nearest neighbors 
        and compute the average distance to those objects
        
        If there are no objects of a certain class among K neighbors, set mean distance to 999
        '''
        for k in self.k_list:
            feats = []
            for c in range(self.n_classes):
                class_locs = np.where(neighs_y[n_neighbors - 1][k] == c)[0]
                if len(class_locs) > 0:
                    feats.append(neighs_dist[class_locs].mean())
                else:
                    feats.append(999)

            assert len(feats) == self.n_classes
            return_list += [feats]


        '''
        Mean target on the neighbours
        '''
        del neighs_y; gc.collect()
        neighs_y_reg = {}
        for k in self.k_list:
            ns_indices = np.hstack([self.uniquei_to_all[nei] for nei in neighs_to_unique[:k]])
            neighs_y_reg[k] = self.y_train_reg[ns_indices]
        return_list += [neighs_y_reg[k].mean() for k in self.k_list]
        return_list += [neighs_y_reg[k].min() for k in self.k_list]
        return_list += [neighs_y_reg[k].max() for k in self.k_list]
        return_list += [neighs_y_reg[k].std() for k in self.k_list]

        knn_feats = np.hstack(return_list)

        # assert knn_feats.shape == (239,) or knn_feats.shape == (239, 1)
        return knn_feats


class NearestNeighborsFeats(BaseEstimator, ClassifierMixin):
    """
    This class should implement KNN features extraction
    """
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
    
    def fit(self, X, y_bins, y_regr):
        """
        Set's up the train set and self.NN object
        y_reg = regression values
        y = labels bins
        """
        # Create a NearestNeighbors (NN) object. We will use it in `predict` function 
        self.NN = NearestNeighbors(n_neighbors=max(self.k_list), 
                                      metric=self.metric, 
                                      n_jobs=1, 
                                      algorithm='brute' if self.metric=='cosine' else 'auto')
        self.NN.fit(X)

        # Store labels
        self.y_train = y_bins
        self.y_train_reg = y_regr
        
        # Save how many classes we have
        if self.n_classes_ is not None:
            self.n_classes = self.n_classes_
        else:
            self.n_classes = np.unique(y_bins).shape[0]
        
    def predict(self, X):
        """
        KNN features for every object of a dataset X
        """
        if self.n_jobs == 1:
            test_feats = []
            for i in range(X.shape[0]):
                test_feats.append(self.get_features_for_one(X[i:i+1]))
        else:
            with Pool(self.n_jobs) as p:
                gen = (X[i:i + 1] for i in range(X.shape[0]))
                test_feats = p.map(self.get_features_for_one, gen)

        return np.vstack(test_feats)

    def get_features_for_one(self, x):
        """
        Computes KNN features for a single object x
        """

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
        neighs_y_reg = self.y_train_reg[neighs]

        return_list = [] 
        
        ''' 
        Fraction of objects of every class.
        '''
        for k in self.k_list:
            classes_in_neighs = np.bincount(neighs_y[:k], minlength=self.n_classes)
            feats = classes_in_neighs / classes_in_neighs.sum()
            assert len(feats) == self.n_classes
            return_list += [feats]
        
        '''
        Same label streak: the largest number N, 
        such that N nearest neighbors have the same label.
        '''
        feats = np.array([len(neighs)])
        non_class_locs = np.where(neighs_y != neighs_y[0])[0]
        if len(non_class_locs) > 0:
            feats[0] = non_class_locs[0]
        
        assert len(feats) == 1
        return_list += [feats]
        
        '''
        Minimum distance to objects of each class
        the first instance of a class and take its distance as features.
               
        If there are no neighboring objects of some classes, 
        Then set distance to that class to be 999.
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
        Minimum *normalized* distance to objects of each class
        divide the distances by the distance to the closest neighbor.
               
        If there are no neighboring objects of some classes, 
        Then set distance to that class to be 999.
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
        Distance to Kth neighbor ("quantiles" of a distribution)
        Distance to Kth neighbor normalized by distance to the first neighbor
        
        '''
        for k in self.k_list:
            
            feat_51 = neighs_dist[k - 1]
            feat_52 = neighs_dist[k - 1] / (neighs_dist[0] + self.eps)
            
            return_list += [[feat_51, feat_52]]
        
        '''
        Mean distance to neighbors of each class for each K from `k_list` 
        For each class select the neighbors of that class among K nearest neighbors 
        and compute the average distance to those objects
        
        If there are no objects of a certain class among K neighbors, set mean distance to 999
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
        
        
        '''
        Mean target on the neighbours
        '''
        return_list += [neighs_y_reg[:k].mean() for k in self.k_list]
        return_list += [neighs_y_reg[:k].min() for k in self.k_list]
        return_list += [neighs_y_reg[:k].max() for k in self.k_list]
        return_list += [neighs_y_reg[:k].std() for k in self.k_list]
        
        knn_feats = np.hstack(return_list)
        
        # assert knn_feats.shape == (239,) or knn_feats.shape == (239, 1)
        return knn_feats
