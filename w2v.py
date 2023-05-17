import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import re
import math
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cab

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import warnings
import json 
import pickle
warnings.filterwarnings('ignore')

import torch
import pickle
import random

import sentence_transformers 
from sklearn.preprocessing import KBinsDiscretizer
from sentence_transformers import SentenceTransformer

from gensim.models import Word2Vec

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset

from concurrent.futures import ThreadPoolExecutor, as_completed

w2v_window = 3
w2v_min_count = 1
w2v_epochs = 500
w2v_vector_size = 128
n = 20

# id_seqs = np.load('./data/id_seqs.npy', allow_pickle=True)
# id_word2vec = Word2Vec(sentences=id_seqs, window=w2v_window, min_count=w2v_min_count, workers=40, vector_size=w2v_vector_size, epochs=w2v_epochs)
# id_word2vec.save('./data/word2vec_{}_{}_{}_{}.pt'.format(w2v_vector_size, w2v_epochs, w2v_window, w2v_min_count))

id_word2vec = Word2Vec.load('./data/word2vec_{}_{}_{}_{}.pt'.format(w2v_vector_size, w2v_epochs, w2v_window, w2v_min_count))

product2id = json.load(open('data/product2id.json', 'r'))
id2product = json.load(open('data/id2product.json', 'r'))
id2product = {int(k): v for k, v in id2product.items()}

w2v_map = defaultdict(list)
print('start')
# for id_ in tqdm(id2product.keys(), total=len(id2product)):  # id2product.keys()
#     try:
#         w2v_sim = id_word2vec.wv.most_similar(id_)
#         w2v_sim = [id2product[x[0]] for x in w2v_sim]
#         w2v_map[id2product[id_]] = w2v_sim
#     except:
#         print(id_)
#         w2v_map[id2product[id_]] = []

def process_id(id_):
    try:
        w2v_sim = id_word2vec.wv.most_similar(id_, topn=n)
        w2v_sim = [id2product[x[0]] for x in w2v_sim]
        w2v_map[id2product[id_]] = w2v_sim
    except:
        print(id_)
        w2v_map[id2product[id_]] = []


# 创建线程池
with ThreadPoolExecutor(max_workers=40) as executor:
    # 提交任务，并使用tqdm显示进度条
    futures = [executor.submit(process_id, id_) for id_ in id2product.keys()]
    progress_bar = tqdm(total=len(futures))
    
    # 获取已完成的任务结果
    for future in as_completed(futures):
        # 处理任务结果（如果有需要的话）
        progress_bar.update(1)

    progress_bar.close()

# 在这里，w2v_map 已经被填充好了，可以使用它进行后续操作
pickle.dump(w2v_map, open('./data/word2vec_map_top{}.pkl'.format(n), 'wb'))
json.dump(w2v_map, open('./data/word2vec_map_top{}.json'.format(n), 'w'))
