# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 11:29:45 2020

@author: komarov
"""

import pandas as pd
import numpy as np
from numba import jit
import matplotlib.pyplot as plt


# defining constants
W = 45

dates = pd.date_range('2010-01-01', '2015-12-31', freq='1b')
T = len(dates)
W = 45
N_STOCK = 5

names = ['spy'] + ['stock_%s' % i for i in range(N_STOCK)]
sigmas = np.array([0.1] + [0.2] * N_STOCK) / np.sqrt(252)
data = pd.DataFrame(np.random.randn(len(dates), N_STOCK + 1) * sigmas, index=dates, columns=names)


@jit("f8[:](f8[:,:], i8)")
def fast_rolling_beta(xy, W):
    """
    xy = np.arryay of shape (T, 2) with 1st col being spy and 2nd being the stock of interest
    
    y_t = alpha + beta * (x_t - r_f) + e_t
    beta = cov(x, y) / var(x)
    
    
    res[t] = beta for time window [t - 45, t - 1]
    plenty of room to optimize, but not just yet...
    """
    T = int(xy.shape[0])
    res = np.empty(T, dtype=np.float64)
    res[:W] = np.nan
    for t in range(W, T):
        res[t] = np.cov(xy[t-W:t,:], rowvar=False)[0, 1] / np.var(xy[t-W:t,0])
    return res

# getting time series of beta for each stock with spy
betas = []
for i in range(1, data.shape[1]):
    b = pd.Series(fast_rolling_beta(data.iloc[:, [0, i]].values, W), index=data.index, name='beta_%s' % (i - 1))
    betas.append(b)
    

betas = pd.concat(betas, axis=1)
betas_left_align = betas.shift(-W)


betas_left_align.iloc[:, 4].hist(bins=40)


z01 = pd.DataFrame(np.random.randn(T, 2), index=data.index, columns=['z0', 'z1'])

def get_rho_distribution(z01, rho):
    x = z01.iloc[:, 0].values * rho + z01.iloc[:, 1].values * np.sqrt(1 - rho * rho)
    print(x.mean(), x.std())
    xy = np.stack([z01.iloc[:, 0].values, x], axis=1)
    rolling_beta = fast_rolling_beta(xy, W)
    plt.hist(rolling_beta, bins=40)
    plt.title('rho=%.03f, corr=%.03f' % (rho, np.corrcoef(xy, rowvar=False)[0, 1]))
    
get_rho_distribution(z01, 0.6)

