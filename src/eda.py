# -*- coding: utf-8 -*-
"""
Created on Mon May  7 20:18:22 2018

@author: komarov
"""

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import os
import zipfile
import gzip

import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#zf = zipfile.ZipFile("../data/sales_train.csv.gz")
#with gzip.open("../data/sales_train.csv.gz") as fp:
#    df = pd.read_csv(fp, parse_dates=['date'], dayfirst=True, infer_datetime_format=True)
#    df = pd.read_csv(fp, parse_dates=["FL_DATE"]).rename(columns=str.lower)

df = pd.read_csv(
        "../data/sales_train.csv.gz", 
        parse_dates=['date'], 
        dayfirst=True, 
        infer_datetime_format=True
        )

