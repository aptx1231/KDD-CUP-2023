import time
import numpy as np 
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import re
# import math
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# from lightgbm import LGBMRegressor, LGBMClassifier
# from xgboost import XGBRegressor, XGBClassifier
# from catboost import CatBoostRegressor, CatBoostClassifier
# import lightgbm as lgb
# import xgboost as xgb
# import catboost as cab

# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, GridSearchCV
# from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, StackingRegressor
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier
# from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
# from sklearn import metrics
# from sklearn.svm import SVC
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
# from collections import defaultdict, Counter
import warnings
import json 
import pickle
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import random
from tqdm import tqdm
# import sentence_transformers 
# from sklearn.preprocessing import KBinsDiscretizer
# from sentence_transformers import SentenceTransformer

# from gensim.models import Word2Vec

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset
import argparse
from utils import set_random_seed, get_logger, ensure_dir, str2bool, str2float
from data import NNDataset, NNDatasetV2, NNDatasetV3
from model import MatchModel, BaseModel, MatchModelV2, BinaryModel
from sklearn import metrics
"""
相比于trainDeeep.py，加入一些手动聚合的序列特征，例如历史序列的平均价格，历史序列的不同类别数之类的
SeqFeatureEmbedding现在使用2层全连接，可以换多层！
# TODO: product-id作为特征之一
# TODO: emb_dim变大 不同特征用不同emb_dim等
# TODO: DIN等序列模型
"""

parser = argparse.ArgumentParser()
# 增加指定的参数
# parser.add_argument('--model', type=str,
#                         default='xDeepFM', help='the name of model')
parser.add_argument('--exp_id', type=str, default=None, help='id of experiment')
parser.add_argument('--seed', type=int, default=2023, help='random seed')
parser.add_argument('--Fold', type=int, default=0, help='Fold')
parser.add_argument('--device', type=int, default=0, help='device')

parser.add_argument('--len_candidate_set', type=int, default=10, help='len_candidate_set')
parser.add_argument('--emb_dim', type=int, default=16, help='emb_dim')
parser.add_argument('--hid_dim', type=int, default=256, help='hid_dim')
parser.add_argument('--layers', type=int, default=4, help='layers')
parser.add_argument('--heads', type=int, default=8, help='heads')
parser.add_argument('--bidirectional', type=str2bool, default=False, help='bidirectional')
parser.add_argument('--seq_emb_factor', type=int, default=4, help='seq_emb_factor')
parser.add_argument('--intra', type=str, default='intra', help='intra attn: intra / intra2')
parser.add_argument('--seq_model', type=str, default='LSTM', help='seq_model LSTM/transformer')

parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--batch_size', type=int, default=1024, help='batch_size')
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--lr', type=float, default=0.002, help='lr')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight_decay')
parser.add_argument('--lr_patience', type=int, default=5, help='lr_patience')
parser.add_argument('--dense_norm', type=str2bool, default=True, help='dense_norm')
parser.add_argument('--attn_match', type=str2bool, default=True, help='attn_match')

parser.add_argument('--train', type=str2bool, default=True, help='train')
parser.add_argument('--load_init', type=str2bool, default=False, help='load_init')
parser.add_argument('--load_epoch', type=int, default=None, help='load_epoch')
parser.add_argument('--load_exp_id', type=int, default=None, help='load_exp_id')
parser.add_argument('--log_train_mrr', type=str2bool, default=True, help='log_train_mrr')

parser.add_argument('--add_title', type=str2bool, default=True, help='add_title')
parser.add_argument('--add_desc', type=str2bool, default=True, help='add_desc')
parser.add_argument('--add_w2v', type=str2bool, default=True, help='add_w2v')
args = parser.parse_args()


log_train_mrr = args.log_train_mrr
emb_dim = args.emb_dim
dense_bins = 10
hid_dim = args.hid_dim
dropout = args.dropout
layers = args.layers
heads = args.heads
bidirectional = args.bidirectional
seq_emb_factor = args.seq_emb_factor  # 人工序列特征的嵌入是emb_dim的几倍
seq_model = args.seq_model

batch_size = args.batch_size
epochs = args.epochs
len_candidate_set = args.len_candidate_set
Fold = args.Fold
device = torch.device('cuda:{}'.format(args.device))
dense_norm = args.dense_norm
num_workers = 0
intra = args.intra

train = args.train
load_init = args.load_init
load_epoch = args.load_epoch
load_exp_id = args.load_exp_id

add_title = args.add_title
add_desc = args.add_desc
add_w2v = args.add_w2v

# TODO: 调整batch-size， 调整hidden-size
# TODO: 跑Fold 4
learning_rate = args.lr
weight_decay = args.weight_decay
early_stop_lr = 1e-6
lr_patience = args.lr_patience
lr_decay_ratio = 0.1
clip = 5
log_every = 100
early_stop = True
patience = 10
kfold = 5
attn_match = args.attn_match 

w2v_window = 3
w2v_min_count = 1
w2v_epochs = 500
w2v_vector_size = 128

seed = args.seed
set_random_seed(seed)

model_name = 'BinaryModelFold{}'.format(Fold)
loc2id = {'DE': 0, 'JP': 1, 'UK': 2, 'ES': 3, 'FR': 4, 'IT': 5}

config = locals()

# 加载必要的数据

exp_id = args.exp_id
if exp_id is None:
    exp_id = int(random.SystemRandom().random() * 100000)
    config['exp_id'] = exp_id

logger = get_logger(config)
logger.info('Exp_id {}'.format(exp_id))
logger.info(config)

logger.info('read data')

titles_embedding = np.load('./data/titles_embedding.npy')
descs_embedding = np.load('./data/descs_embedding.npy')
logger.info('titles_embedding: {}'.format(titles_embedding.shape))
logger.info('descs_embedding: {}'.format(descs_embedding.shape))

product2id = json.load(open('data/product2id.json', 'r'))
id2product = json.load(open('data/id2product.json', 'r'))
id2product = {int(k): v for k, v in id2product.items()}
logger.info('product2id: {}'.format(len(product2id)))
logger.info('id2product: {}'.format(len(id2product)))

word2vec_embedding = np.load('./data/word2vec_embedding.npy')
logger.info('word2vec_embedding: {}'.format(word2vec_embedding.shape))

# top200 = pickle.load(open('data/top200_new.pkl', 'rb'))

df_train_encoded = pd.read_csv('data/df_train_encoded.csv')
df_test_encoded = pd.read_csv('data/df_test_encoded_phase2.csv')
products_encoded = pd.read_csv('./data/products_encoded.csv')
logger.info('df_train_encoded: {}'.format(df_train_encoded.shape))
logger.info('df_test_encoded: {}'.format(df_test_encoded.shape))
logger.info('products_encoded: {}'.format(products_encoded.shape))

num_features = ['price', 'len_title', 'len_desc']
if dense_norm:
    logger.info('MinMaxScaler Norm products_num_feas')
    mms = MinMaxScaler(feature_range=(0,1))
    products_encoded[num_features] = mms.fit_transform(products_encoded[num_features])
for fe in num_features:
    products_encoded[fe] = products_encoded[fe].astype('float32')
    assert products_encoded[fe].dtypes == 'float32'

id_count = products_encoded.shape[0]

train_preds_encoded = pickle.load(open('./data/train_preds_all_encoded_new_phase2.pkl', 'rb'))  # (len_train, 100)
test_preds_encoded = pickle.load(open('./data/test_preds_all_encoded_new_phase2.pkl', 'rb'))  # (len_test, 100)
test_preds = pickle.load(open('./data/test_preds_phase2.pkl', 'rb'))
logger.info('train_preds_encoded: {}'.format(len(train_preds_encoded)))
logger.info('test_preds_encoded: {}'.format(len(test_preds_encoded)))
logger.info('test_preds: {}'.format(len(test_preds)))

logger.info('Cutting the candidate_set to {}'.format(len_candidate_set))
# TODO: 可以改成保障一个正样本，补充9个负样本
cut_train_preds_encoded = [lst[:len_candidate_set] for lst in tqdm(train_preds_encoded, total=len(train_preds_encoded))]
df_train_encoded['recall'] = cut_train_preds_encoded
cut_test_preds_encoded = [lst[:len_candidate_set] for lst in tqdm(test_preds_encoded, total=len(test_preds_encoded))]
df_test_encoded['recall'] = cut_test_preds_encoded

logger.info('Eval the prev_items')
df_train_encoded['prev_items'] = df_train_encoded['prev_items'].apply(eval)
df_test_encoded['prev_items'] = df_test_encoded['prev_items'].apply(eval)

logger.info('Load Hand-made Seq Features')
df_train_seqs_feas_all = pd.read_csv('data/df_train_seqs_feas_all.csv')  # 29维特征
df_test_seqs_feas_all = pd.read_csv('data/df_test_seqs_feas_all_phase2.csv')
logger.info('df_train_seqs_feas_all: {}'.format(df_train_seqs_feas_all.shape))
logger.info('df_test_seqs_feas_all: {}'.format(df_test_seqs_feas_all.shape))
seqs_cat_feas = [f for f in df_train_seqs_feas_all.columns if 'NUNIQUE' in f or 'COUNT' in f or 'encode_' in f]
seqs_num_feas = [f for f in df_train_seqs_feas_all.columns if f not in seqs_cat_feas]
logger.info('seqs_cat_feas: {}'.format(seqs_cat_feas))
logger.info('seqs_num_feas: {}'.format(seqs_num_feas))

if dense_norm:
    logger.info('MinMaxScaler Norm seqs_num_feas')
    mms = MinMaxScaler(feature_range=(0,1))
    df_train_seqs_feas_all[seqs_num_feas] = mms.fit_transform(df_train_seqs_feas_all[seqs_num_feas])
    df_test_seqs_feas_all[seqs_num_feas] = mms.fit_transform(df_test_seqs_feas_all[seqs_num_feas])

for fe in seqs_num_feas:
    df_train_seqs_feas_all[fe] = df_train_seqs_feas_all[fe].astype('float32')
    df_test_seqs_feas_all[fe] = df_test_seqs_feas_all[fe].astype('float32')

df_train_all = pd.concat([df_train_encoded, df_train_seqs_feas_all], axis=1)
df_test_all = pd.concat([df_test_encoded, df_test_seqs_feas_all], axis=1)
logger.info('df_train_all: {}'.format(df_train_all.shape))
logger.info('df_test_all: {}'.format(df_test_all.shape))

df_train_all_exploded = df_train_all.explode('recall')
df_test_all_exploded = df_test_all.explode('recall')

df_train_all_exploded['label'] = df_train_all_exploded['next_item'] == df_train_all_exploded['recall']
df_test_all_exploded['label'] = df_test_all_exploded['next_item'] == df_test_all_exploded['recall']

df_train_all_exploded['index'] = df_train_all_exploded.index 
df_test_all_exploded['index'] = df_test_all_exploded.index 

logger.info('df_train_all_exploded: {}'.format(df_train_all_exploded.shape))
logger.info('df_test_all_exploded: {}'.format(df_test_all_exploded.shape))

logger.info('df_train_all_exploded.label.sum: {}'.format(df_train_all_exploded['label'].sum()))
logger.info('df_test_all_exploded.label.sum: {}'.format(df_test_all_exploded['label'].sum()))

df_train_encoded_exploded = df_train_all_exploded[['prev_items', 'next_item', 'locale', 'recall', 'label', 'index']]
df_train_seqs_cat_feas = df_train_all_exploded[seqs_cat_feas]
df_train_seqs_num_feas = df_train_all_exploded[seqs_num_feas]
df_test_encoded_exploded = df_test_all_exploded[['prev_items', 'next_item', 'locale', 'recall', 'label', 'index']]
df_test_seqs_cat_feas = df_test_all_exploded[seqs_cat_feas]
df_test_seqs_num_feas = df_test_all_exploded[seqs_num_feas]

del df_train_all_exploded
del df_test_all_exploded
del df_train_all
del df_test_all
del train_preds_encoded

logger.info('df_train_encoded_exploded: {}'.format(df_train_encoded_exploded.shape))
logger.info('df_train_seqs_cat_feas: {}'.format(df_train_seqs_cat_feas.shape))
logger.info('df_train_seqs_num_feas: {}'.format(df_train_seqs_num_feas.shape))
logger.info('df_test_encoded_exploded: {}'.format(df_test_encoded_exploded.shape))
logger.info('df_test_seqs_cat_feas: {}'.format(df_test_seqs_cat_feas.shape))
logger.info('df_test_seqs_num_feas: {}'.format(df_test_seqs_num_feas.shape))


df_test = pd.read_csv('data/phase2/sessions_test_task1.csv')
logger.info('df_test: {}'.format(df_test.shape))

tmp = pd.concat([df_train_seqs_feas_all[seqs_cat_feas], df_test_seqs_feas_all[seqs_cat_feas]])
tmp_nunique = (tmp.max() + 1).to_dict()  # 不是nunique，因为这个是计数特征，不是连续的0~n-1

data_feature = {}
data_feature['len_encode_brand'] = products_encoded['encode_brand'].nunique()
data_feature['len_encode_color'] = products_encoded['encode_color'].nunique()
data_feature['len_encode_size'] = products_encoded['encode_size'].nunique()
data_feature['len_encode_model'] = products_encoded['encode_model'].nunique()
data_feature['len_encode_material'] = products_encoded['encode_material'].nunique()
data_feature['len_encode_author'] = products_encoded['encode_author'].nunique()
data_feature['len_locale'] = len(loc2id)
data_feature['dense_bins'] = dense_bins
data_feature['id_count'] = id_count
data_feature['len_features'] = products_encoded.shape[1] - 1
data_feature['len_emb_features'] = int(add_desc) + int(add_title) + int(add_w2v)
data_feature['len_candidate_set'] = len_candidate_set
data_feature['w2v_vector_size'] = w2v_vector_size
data_feature['sentence_vector_size'] = 384
data_feature['len_seqs_cat_feas'] = len(seqs_cat_feas)
data_feature['len_seqs_num_feas'] = len(seqs_num_feas)
data_feature['seq_emb_factor'] = seq_emb_factor
data_feature.update(tmp_nunique)
logger.info('data_feature:')
logger.info(data_feature)

del tmp

fold_size = int(df_train_encoded_exploded.shape[0] / len_candidate_set // kfold) * len_candidate_set  # 7212490
index_list = list(range(0, len(df_train_encoded_exploded)))
val_indexes = []
start = 0
for i in range(kfold):
    end = min(start + fold_size, len(index_list))
    print(start, end)
    val_indexes.append(index_list[start:end])
    # print(np.min(val_indexes[-1]), np.max(val_indexes[-1]), len(val_indexes[-1]))
    start = end
train_indexes = []
for i, fold in enumerate(val_indexes):
    print(f"Fold {i+1}:")
    print("Validation set:", len(fold))
    train_indexes.append([index for sublist in val_indexes[:i] + val_indexes[i+1:] for index in sublist])
    print("Training set:", len(train_indexes[-1]))

# np.save('data/5fold_trn_idx_list_binary_phase2_{}.npy'.format(len_candidate_set), train_indexes)
# np.save('data/5fold_val_idx_list_binary_phase2_{}.npy'.format(len_candidate_set), val_indexes)


# 加载模型等

logger.info('create model')

products_input = {name: torch.tensor(products_encoded[name].values).to(device) for name in products_encoded.columns}

model = BinaryModel(config, data_feature, products_input, word2vec_embedding, titles_embedding, descs_embedding).to(device)
logger.info(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=lr_patience, factor=lr_decay_ratio)

for name, param in model.named_parameters():
    logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))
total_num = sum([param.nelement() for param in model.parameters()])
logger.info('Total parameter numbers: {}'.format(total_num))


# 数据集DataLoader

trn_idx_list = train_indexes
val_idx_list = val_indexes
logger.info('Fold {}: trn_idx {}'.format(Fold, len(trn_idx_list[Fold])))
logger.info('Fold {}: val_idx {}'.format(Fold, len(val_idx_list[Fold])))

train_set = NNDatasetV3(df_train_encoded_exploded.iloc[trn_idx_list[Fold]], 
                        df_train_seqs_cat_feas.iloc[trn_idx_list[Fold]], 
                        df_train_seqs_num_feas.iloc[trn_idx_list[Fold]])
val_set = NNDatasetV3(df_train_encoded_exploded.iloc[val_idx_list[Fold]], 
                      df_train_seqs_cat_feas.iloc[val_idx_list[Fold]], 
                      df_train_seqs_num_feas.iloc[val_idx_list[Fold]])
test_set = NNDatasetV3(df_test_encoded_exploded, df_test_seqs_cat_feas, 
                       df_test_seqs_num_feas)
logger.info('train_set: {}'.format(len(train_set)))
logger.info('val_set: {}'.format(len(val_set)))
logger.info('test_set: {}'.format(len(test_set)))


def collate_fn(indices):
    batch_prev_items = []
    batch_locale = []
    batch_len = []
    batch_mask = []
    batch_candidate = []
    batch_origin_label = []
    batch_label = []
    batch_seq_cat = []
    batch_seq_num = []
    batch_index = []
    for item in indices:
        batch_len.append(len(item[0]))  # prev_items
    max_len = max(batch_len)
    for item in indices:
        l = len(item[0])
        batch_mask.append([1] * (l) + [0] * (max_len - l))  # 0代表padding的位置，需要mask
    for item in indices:
        # ['prev_items', 'locale', 'recall', 'next_item', label', 'index', 'seqs_cat_feas', 'seqs_num_feas']
        prev_items = item[0].copy()
        while (len(prev_items) < max_len):
            prev_items.append(id_count)  # embdding的时候id_count+1，把id_count作为padding了
        batch_prev_items.append(prev_items)
        batch_locale.append(item[1])
        batch_candidate.append(item[2])
        batch_origin_label.append(item[3])
        batch_label.append(item[4])
        batch_index.append(item[5])
        batch_seq_cat.append(item[6])
        batch_seq_num.append(item[7])
    return [torch.LongTensor(batch_prev_items).to(device), torch.LongTensor(batch_locale).to(device), 
            torch.LongTensor(batch_len).to(device), torch.LongTensor(batch_mask).to(device), 
            torch.LongTensor(batch_candidate).to(device), torch.FloatTensor(batch_label).to(device), 
            torch.LongTensor(batch_origin_label).to(device), torch.LongTensor(batch_index).to(device), 
            torch.LongTensor(batch_seq_cat).to(device), torch.FloatTensor(batch_seq_num).to(device)]


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
logger.info('train_loader: {}'.format(len(train_loader)))
logger.info('val_loader: {}'.format(len(val_loader)))
logger.info('test_loader: {}'.format(len(test_loader)))


output_dir = 'ckpt/{}'.format(exp_id)
ensure_dir(output_dir)

if load_init:
    load_dir = 'ckpt/{}'.format(load_exp_id)
    load_path = '{}/{}_{}_{}.pt'.format(load_dir, load_exp_id, model_name, load_epoch)
    logger.info('Load Init model from {}'.format(load_path))
    model.load_state_dict(torch.load(load_path, map_location='cpu'))
    # print(model.device)


def get_train_label_index(group):
    label = group['train_origin_label'].iloc[0]
    candidate = group['train_candidate'].tolist()
    try:
        index = candidate.index(label)
        rrank = 1 / (index + 1)
    except:
        rrank = 0
    return pd.Series({'label_rrank': rrank})

def get_val_label_index(group):
    label = group['val_origin_label'].iloc[0]
    candidate = group['val_candidate'].tolist()
    try:
        index = candidate.index(label)
        rrank = 1 / (index + 1)
    except:
        rrank = 0
    return pd.Series({'label_rrank': rrank})

def get_test_label_index(group):
    label = group['test_origin_label'].iloc[0]
    candidate = group['test_candidate'].tolist()
    try:
        index = candidate.index(label)
        rrank = 1 / (index + 1)
    except:
        rrank = 0
    return pd.Series({'label_rrank': rrank})


best_epoch = -1
# 训练
if train:
    logger.info('Training...')
    output_dir = 'ckpt/{}'.format(exp_id)
    ensure_dir(output_dir)
    auc_all = []
    mrr_all = []
    min_val_loss = float('inf')
    max_val_auc = 0.0
    max_val_mrr = 0.0
    # best_epoch = -1
    for epoch in range(epochs):
        # train
        logger.info('start train epoch {}'.format(epoch))
        model.train()
        train_loss_list = []
        train_candidate_list = []
        train_y_pred_list = []
        train_index_list = []
        train_label_list = []
        train_origin_label_list = []
        for batch_prev_items, batch_locale, batch_len, batch_mask, batch_candidate, \
                batch_label, batch_origin_label, batch_index, batch_seq_cat, batch_seq_num \
                in tqdm(train_loader, desc='train model {}'.format(exp_id), total=len(train_loader)):
            optimizer.zero_grad()
            # y_pred (B, 1), batch_candidate / batch_origin_label / batch_label / batch_index (B, )
            y_pred, loss = model.predict(batch_prev_items=batch_prev_items, batch_locale=batch_locale, 
                                        batch_candidate_set=batch_candidate, batch_len=batch_len, 
                                        batch_label=batch_label.unsqueeze(-1), batch_mask=batch_mask,
                                        batch_seq_cat=batch_seq_cat, batch_seq_num=batch_seq_num)
            # TODO: 去掉？
            loss.backward(retain_graph=True)
            train_loss_list.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            with torch.no_grad():
                train_candidate_list += batch_candidate.tolist()
                train_y_pred_list += y_pred.squeeze(-1).tolist()
                train_index_list += batch_index.tolist()
                train_label_list += batch_label.tolist()
                train_origin_label_list += batch_origin_label.tolist()
            # break
        # binary-metric
        train_logloss = metrics.log_loss(train_label_list, train_y_pred_list)
        train_auc = metrics.roc_auc_score(train_label_list, train_y_pred_list)
        train_precision = metrics.precision_score(train_label_list, [1 if i >= 0.5 else 0 for i in train_y_pred_list])
        train_recall = metrics.recall_score(train_label_list, [1 if i >= 0.5 else 0 for i in train_y_pred_list])
        train_f1 = metrics.f1_score(train_label_list, [1 if i >= 0.5 else 0 for i in train_y_pred_list])
        logger.info("Train Epoch {}, Logloss: {:.4f}, AUC: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}".format(
            epoch, train_logloss, train_auc, train_precision, train_recall, train_f1))
        # MRR
        if log_train_mrr:
            t1 = time.time()
            df = pd.DataFrame({
                'train_candidate': train_candidate_list,
                'train_index': train_index_list,
                'train_origin_label': train_origin_label_list,
                'train_y_pred': train_y_pred_list
            })
            df_sorted = df.sort_values(['train_index', 'train_y_pred'], ascending=[True, False]).reset_index(drop=True)
            df_grouped = df_sorted.groupby('train_index').apply(get_train_label_index)
            train_mrr = np.mean(df_grouped['label_rrank'].values)
            t2 = time.time()
            logger.info(f"Train Epoch {epoch}, MRR: {train_mrr:.4f}, Time: {t2-t1:.4f}")
        else:
            train_mrr = 0
        train_loss = np.mean(train_loss_list)
        # val
        val_loss_list= []
        val_candidate_list = []
        val_y_pred_list = []
        val_index_list = []
        val_label_list = []
        val_origin_label_list = []
        with torch.no_grad():
            model.eval()
            for batch_prev_items, batch_locale, batch_len, batch_mask, batch_candidate, \
                batch_label, batch_origin_label, batch_index, batch_seq_cat, batch_seq_num \
                in tqdm(val_loader, desc='val model {}'.format(exp_id), total=len(val_loader)):
                # y_pred (B, 1), batch_candidate / batch_origin_label / batch_label / batch_index (B, )
                y_pred, loss = model.predict(batch_prev_items=batch_prev_items, batch_locale=batch_locale, 
                                            batch_candidate_set=batch_candidate, batch_len=batch_len, 
                                            batch_label=batch_label.unsqueeze(-1), batch_mask=batch_mask,
                                            batch_seq_cat=batch_seq_cat, batch_seq_num=batch_seq_num)
                val_loss_list.append(loss.item())
                with torch.no_grad():
                    val_candidate_list += batch_candidate.tolist()
                    val_y_pred_list += y_pred.squeeze(-1).tolist()
                    val_index_list += batch_index.tolist()
                    val_label_list += batch_label.tolist()
                    val_origin_label_list += batch_origin_label.tolist()
                # break
        # binary-metric
        val_logloss = metrics.log_loss(val_label_list, val_y_pred_list)
        val_auc = metrics.roc_auc_score(val_label_list, val_y_pred_list)
        val_precision = metrics.precision_score(val_label_list, [1 if i >= 0.5 else 0 for i in val_y_pred_list])
        val_recall = metrics.recall_score(val_label_list, [1 if i >= 0.5 else 0 for i in val_y_pred_list])
        val_f1 = metrics.f1_score(val_label_list, [1 if i >= 0.5 else 0 for i in val_y_pred_list])
        logger.info("Val Epoch {}, Logloss: {:.4f}, AUC: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}".format(
            epoch, val_logloss, val_auc, val_precision, val_recall, val_f1))
        # MRR
        if log_train_mrr:
            t1 = time.time()
            df = pd.DataFrame({
                'val_candidate': val_candidate_list,
                'val_index': val_index_list,
                'val_origin_label': val_origin_label_list,
                'val_y_pred': val_y_pred_list
            })
            df_sorted = df.sort_values(['val_index', 'val_y_pred'], ascending=[True, False]).reset_index(drop=True)
            df_grouped = df_sorted.groupby('val_index').apply(get_val_label_index)
            val_mrr = np.mean(df_grouped['label_rrank'].values)
            t2 = time.time()
            logger.info(f"Val Epoch {epoch}, MRR: {val_mrr:.4f}, Time: {t2-t1:.4f}")
        else:
            val_mrr = 0
        val_loss = np.mean(val_loss_list)
        
        mrr_all.append(val_mrr)
        auc_all.append(val_auc)
        lr_scheduler.step(val_mrr)
        lr = optimizer.param_groups[0]['lr']
        if log_train_mrr:
            logger.info('Epoch {}/{} Complete: Train Loss {:.6f}, Train MRR {:.6f}, Val Loss {:.6f}, Val MRR {:.6f}, lr {}'.format(
                epoch, epochs, train_loss, train_mrr, val_loss, val_mrr, lr))
        else:
            logger.info('Epoch {}/{} Complete: Train Loss {:.6f}, Val Loss {:.6f}, lr {}'.format(epoch, epochs, train_loss, val_loss, lr))
        if val_mrr > max_val_mrr:
            min_val_loss = val_loss
            max_val_mrr = val_mrr
            max_val_auc = val_auc
            best_epoch = epoch
            save_path = '{}/{}_{}.pt'.format(output_dir, exp_id, model_name)
            logger.info('Save model to {}'.format(save_path))
            torch.save(model.state_dict(), save_path)
        if lr < early_stop_lr:
            logger.info('early stop')
            break

    # load best epoch
    assert best_epoch == np.argmax(mrr_all)
    logger.info('best_epoch {}, MRR = {}'.format(best_epoch, max_val_mrr))
    load_path = '{}/{}_{}.pt'.format(output_dir, exp_id, model_name)
    logger.info('Load model from {}'.format(load_path))
    model.load_state_dict(torch.load(load_path))
    save_path = '{}/{}_{}_{}.pt'.format(output_dir, exp_id, model_name, best_epoch)
    logger.info('Save model to {}'.format(save_path))
    torch.save(model.state_dict(), save_path)


# 开始评估
logger.info('Testing...')
test_candidate_list = []
test_y_pred_list = []
test_index_list = []
test_label_list = []
test_origin_label_list = []
model.eval()

for batch_prev_items, batch_locale, batch_len, batch_mask, batch_candidate, \
    batch_label, batch_origin_label, batch_index, batch_seq_cat, batch_seq_num \
        in tqdm(test_loader, desc='test model {}'.format(exp_id), total=len(test_loader)):
    # y_pred (B, 1), batch_candidate / batch_origin_label / batch_label / batch_index (B, )
    y_pred, loss = model.predict(batch_prev_items=batch_prev_items, batch_locale=batch_locale, 
                                batch_candidate_set=batch_candidate, batch_len=batch_len, 
                                batch_label=batch_label.unsqueeze(-1), batch_mask=batch_mask,
                                batch_seq_cat=batch_seq_cat, batch_seq_num=batch_seq_num)
    with torch.no_grad():
        test_candidate_list += batch_candidate.tolist()
        test_y_pred_list += y_pred.squeeze(-1).tolist()
        test_index_list += batch_index.tolist()
        test_label_list += batch_label.tolist()
        test_origin_label_list += batch_origin_label.tolist()

# MRR
df = pd.DataFrame({
    'test_candidate': test_candidate_list,
    'test_index': test_index_list,
    'test_y_pred': test_y_pred_list
})
df_sorted = df.sort_values(['test_index', 'test_y_pred'], ascending=[True, False]).reset_index(drop=True)
df_test_preds = df_sorted.groupby('test_index').apply(lambda group : group['test_candidate'].tolist())


logger.info('Decoding the results...')
test_res_unencoded = []
for ind, x in tqdm(enumerate(range(len(test_preds_encoded))), total=len(test_preds_encoded)):
    x = df_test_preds.iloc[ind] + test_preds_encoded[ind][len_candidate_set:]
    assert len(x) == 100
    x_unencoded = [id2product[id_] for id_ in x]
    test_res_unencoded.append(x_unencoded)


logger.info('Saving...')
df_test['next_item_prediction'] = test_res_unencoded
logger.info(df_test['next_item_prediction'].apply(len).describe())
logger.info('Save Fold {} res to {}'.format(Fold, 'output/{}_{}_{}_{}.parquet'.format(seed, exp_id, model_name, best_epoch)))
df_test[['locale', 'next_item_prediction']].to_parquet('output/{}_{}_{}_{}.parquet'.format(seed, exp_id, model_name, best_epoch), engine='pyarrow')
np.save('output/preds_{}_{}_{}_{}_Fold{}.npy'.format(seed, exp_id, model_name, best_epoch, Fold), df_test_preds.values)
