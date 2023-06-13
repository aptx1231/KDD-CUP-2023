import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
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
from data import NNDataset, NNDatasetV2, Data
from model import MatchModel, BaseModel, MatchModelV2, SessionGraph

parser = argparse.ArgumentParser()
# 增加指定的参数
# parser.add_argument('--model', type=str,
#                         default='xDeepFM', help='the name of model')
parser.add_argument('--exp_id', type=str, default=None, help='id of experiment')
parser.add_argument('--seed', type=int, default=2023, help='random seed')
parser.add_argument('--Fold', type=int, default=0, help='Fold')
parser.add_argument('--device', type=int, default=0, help='device')
parser.add_argument('--task', type=str, default='task1', help='task1 / task2')

parser.add_argument('--len_candidate_set', type=int, default=10, help='len_candidate_set')
parser.add_argument('--emb_dim', type=int, default=16, help='emb_dim')
parser.add_argument('--hid_dim', type=int, default=256, help='hid_dim')
parser.add_argument('--layers', type=int, default=4, help='layers')
parser.add_argument('--heads', type=int, default=8, help='heads')
parser.add_argument('--bidirectional', type=str2bool, default=False, help='bidirectional')
parser.add_argument('--seq_emb_factor', type=int, default=4, help='seq_emb_factor')
parser.add_argument('--pro_emb_factor', type=int, default=2, help='pro_emb_factor')
parser.add_argument('--intra', type=str, default='intra', help='intra attn: intra / intra2 / intra3')
parser.add_argument('--seq_model', type=str, default='LSTM', help='seq_model LSTM/transformer')
parser.add_argument('--recall', type=str, default='window', help='next / window')
parser.add_argument('--div', type=int, default=1, help='div of ProductV2')

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

parser.add_argument('--add_title', type=str2bool, default=True, help='add_title')
parser.add_argument('--add_desc', type=str2bool, default=True, help='add_desc')
parser.add_argument('--add_w2v', type=str2bool, default=True, help='add_w2v')
parser.add_argument('--add_pemb', type=str2bool, default=False, help='add_pemb')
parser.add_argument('--grad_title', type=str2bool, default=False, help='grad_title')
parser.add_argument('--grad_desc', type=str2bool, default=False, help='grad_desc')
parser.add_argument('--grad_w2v', type=str2bool, default=False, help='grad_w2v')
parser.add_argument('--grad_pemb', type=str2bool, default=False, help='grad_pemb')
parser.add_argument('--pca', type=str2bool, default=False, help='pca')
parser.add_argument('--pca_dim', type=int, default=64, help='pca_dim')
parser.add_argument('--feature', type=str, default='v1', help='feature v1 v2 v3')
parser.add_argument('--add_mode', type=str2bool, default=True, help='add_mode in feature v3')
args = parser.parse_args()


emb_dim = args.emb_dim
dense_bins = 10
hid_dim = args.hid_dim
dropout = args.dropout
layers = args.layers
heads = args.heads
bidirectional = args.bidirectional
seq_emb_factor = args.seq_emb_factor  # 人工序列特征的嵌入是emb_dim的几倍
pro_emb_factor = args.pro_emb_factor
seq_model = args.seq_model
task = args.task
recall = args.recall
div = args.div
add_mode = args.add_mode

batch_size = args.batch_size
epochs = args.epochs
len_candidate_set = args.len_candidate_set
Fold = args.Fold
device = torch.device('cuda:{}'.format(args.device))
dense_norm = args.dense_norm
num_workers = 0
intra = args.intra

feature = args.feature
train = args.train
load_init = args.load_init
load_epoch = args.load_epoch
load_exp_id = args.load_exp_id

add_title = args.add_title
add_desc = args.add_desc
add_w2v = args.add_w2v
add_pemb = args.add_pemb
grad_title = args.grad_title
grad_desc = args.grad_desc
grad_w2v = args.grad_w2v
grad_pemb = args.grad_pemb
pca = args.pca
pca_dim = args.pca_dim

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

model_name = 'MatchModelV2withATTMatchFold{}'.format(Fold)
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

if not pca:
    titles_embedding = np.load('./data/titles_embedding.npy')
    descs_embedding = np.load('./data/descs_embedding.npy')
else:
    titles_embedding = np.load('./data/titles_embedding_reduced_{}.npy'.format(pca_dim))
    descs_embedding = np.load('./data/descs_embedding_reduced_{}.npy'.format(pca_dim))
logger.info('titles_embedding: {}'.format(titles_embedding.shape))
logger.info('descs_embedding: {}'.format(descs_embedding.shape))

product2id = json.load(open('data/product2id.json', 'r'))
id2product = json.load(open('data/id2product.json', 'r'))
id2product = {int(k): v for k, v in id2product.items()}
logger.info('product2id: {}'.format(len(product2id)))
logger.info('id2product: {}'.format(len(id2product)))

word2vec_embedding = np.load('./data/word2vec_embedding.npy')
logger.info('word2vec_embedding: {}'.format(word2vec_embedding.shape))

df_train_encoded = pd.read_csv('data/df_train_encoded.csv')
if task == 'task1':
    df_test_encoded = pd.read_csv('data/df_test_encoded_phase2.csv')
elif task == 'task2':
    df_test_encoded = pd.read_csv('data/df_test_encoded_phase2_onlytask2.csv')

if feature.lower() == 'v1':
    products_encoded = pd.read_csv('./data/products_encoded.csv')
    num_features = ['price', 'len_title', 'len_desc']
elif feature.lower() == 'v2':
    products_encoded = pd.read_csv('./data/products_encoded_newfeature.csv')
    num_features = ['price']
elif feature.lower() == 'v3':
    products_encoded = pd.read_csv('./data/products_encoded_phase2_V3.csv')
    num_features = ['price']

logger.info('df_train_encoded: {}'.format(df_train_encoded.shape))
logger.info('df_test_encoded: {}'.format(df_test_encoded.shape))
logger.info('products_encoded: {}'.format(products_encoded.shape))

if dense_norm:
    logger.info('MinMaxScaler Norm products_num_feas')
    mms = MinMaxScaler(feature_range=(0,1))
    products_encoded[num_features] = mms.fit_transform(products_encoded[num_features])
for fe in num_features:
    products_encoded[fe] = products_encoded[fe].astype('float32')
    assert products_encoded[fe].dtypes == 'float32'

logger.info('Load Hand-made Seq Features')

if feature.lower() == 'v1':
    df_train_seqs_feas_all = pd.read_csv('data/df_train_seqs_feas_all.csv')  # 29维特征
    if task == 'task1':
        df_test_seqs_feas_all = pd.read_csv('data/df_test_seqs_feas_all_phase2.csv')
    elif task == 'task2':
        df_test_seqs_feas_all = pd.read_csv('data/df_test_seqs_feas_all_phase2_onlytask2.csv')
    dense_bins = 10
elif feature.lower() == 'v2':
    df_train_seqs_feas_all = pd.read_csv('data/df_train_seqs_feas_all_newfeature.csv')  # 24维特征
    if task == 'task1':
        df_test_seqs_feas_all = pd.read_csv('data/df_test_seqs_feas_all_newfeature_phase2.csv')
    elif task == 'task2':
        df_test_seqs_feas_all = pd.read_csv('data/df_test_seqs_feas_all_newfeature_phase2_onlytask2.csv')
    dense_bins = 100
elif feature.lower() == 'v3':
    df_train_seqs_feas_all = pd.read_csv('data/df_train_seqs_feas_all_phase2_V3.csv')  # 36维特征
    if task == 'task1':
        df_test_seqs_feas_all = pd.read_csv('data/df_test_seqs_feas_all_phase2_V3.csv')
    elif task == 'task2':
        df_test_seqs_feas_all = pd.read_csv('data/df_test_seqs_feas_all_phase2_V3_onlytask2.csv')
    dense_bins = 100

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

df_train_seqs_cat_feas = df_train_seqs_feas_all[seqs_cat_feas]
df_train_seqs_num_feas = df_train_seqs_feas_all[seqs_num_feas]
df_test_seqs_cat_feas = df_test_seqs_feas_all[seqs_cat_feas]
df_test_seqs_num_feas = df_test_seqs_feas_all[seqs_num_feas]
logger.info('df_train_seqs_cat_feas: {}'.format(df_train_seqs_cat_feas.shape))
logger.info('df_train_seqs_num_feas: {}'.format(df_train_seqs_num_feas.shape))
logger.info('df_test_seqs_cat_feas: {}'.format(df_test_seqs_cat_feas.shape))
logger.info('df_test_seqs_num_feas: {}'.format(df_test_seqs_num_feas.shape))

id_count = products_encoded.shape[0]

# train_preds_encoded = pickle.load(open('./data/train_preds_all_encoded_new_phase2.pkl', 'rb'))  # (len_train, 100)
# test_preds_encoded = pickle.load(open('./data/test_preds_all_encoded_new_phase2.pkl', 'rb'))  # (len_test, 100)
# test_preds = pickle.load(open('./data/test_preds_phase2.pkl', 'rb'))
if task == 'task1':
    train_preds_encoded = pickle.load(open('./data/train_preds_{}_one_phase2_noleak_encoded.pkl'.format(recall), 'rb'))  # (len_train, 100)
    test_preds_encoded = pickle.load(open('./data/test_preds_{}_one_phase2_encoded.pkl'.format(recall), 'rb'))  # (len_test, 100)
    logger.info('./data/train_preds_{}_one_phase2_noleak_encoded.pkl'.format(recall))
    logger.info('train_preds_encoded: {}'.format(len(train_preds_encoded)))
    logger.info('./data/test_preds_{}_one_phase2_encoded.pkl'.format(recall))
    logger.info('test_preds_encoded: {}'.format(len(test_preds_encoded)))
elif task == 'task2':
    train_preds_encoded = pickle.load(open('./data/train_preds_{}_one_onlytask2_phase2_noleak_encoded.pkl'.format(recall), 'rb'))  # (len_train, 100)
    test_preds_encoded = pickle.load(open('./data/test_preds_{}_one_onlytask2_phase2_encoded.pkl'.format(recall), 'rb'))  # (len_test, 100)
    logger.info('./data/train_preds_{}_one_onlytask2_phase2_noleak_encoded.pkl'.format(recall))
    logger.info('train_preds_encoded: {}'.format(len(train_preds_encoded)))
    logger.info('./data/test_preds_{}_one_onlytask2_phase2_encoded.pkl'.format(recall))
    logger.info('test_preds_encoded: {}'.format(len(test_preds_encoded)))

logger.info('Cutting the candidate_set to {}'.format(len_candidate_set))
cut_train_preds_encoded = [lst[:len_candidate_set] for lst in tqdm(train_preds_encoded, total=len(train_preds_encoded))]
df_train_encoded['recall'] = cut_train_preds_encoded
cut_test_preds_encoded = [lst[:len_candidate_set] for lst in tqdm(test_preds_encoded, total=len(test_preds_encoded))]
df_test_encoded['recall'] = cut_test_preds_encoded

logger.info('Eval the prev_items')
df_train_encoded['prev_items'] = df_train_encoded['prev_items'].apply(eval)
df_test_encoded['prev_items'] = df_test_encoded['prev_items'].apply(eval)

if task == 'task1':
    df_test = pd.read_csv('data/phase2/sessions_test_task1.csv')
    logger.info('df_test Task 1 Phase 2: {}'.format(df_test.shape))
elif task == 'task2':
    df_test = pd.read_csv('data/phase2/sessions_test_task2.csv')
    logger.info('df_test Task 2 Phase 2: {}'.format(df_test.shape))

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
data_feature['len_emb_features'] = int(add_desc) + int(add_title) + int(add_w2v) + int(add_pemb)
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

# 加载模型等

logger.info('create model')

products_input = {name: torch.tensor(products_encoded[name].values).to(device) for name in products_encoded.columns}

model = SessionGraph(config, data_feature, products_input, word2vec_embedding, titles_embedding, descs_embedding).to(device)
logger.info(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=lr_patience, factor=lr_decay_ratio)

for name, param in model.named_parameters():
    logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))
total_num = sum([param.nelement() for param in model.parameters()])
logger.info('Total parameter numbers: {}'.format(total_num))


# 数据集DataLoader

trn_idx_list = np.load('data/5fold_trn_idx_list.npy', allow_pickle=True)
val_idx_list = np.load('data/5fold_val_idx_list.npy', allow_pickle=True)
logger.info('Fold {}: trn_idx {}'.format(Fold, len(trn_idx_list[Fold])))
logger.info('Fold {}: val_idx {}'.format(Fold, len(val_idx_list[Fold])))

train_set = Data(df_train_encoded.iloc[trn_idx_list[Fold]], shuffle=True)
val_set = Data(df_train_encoded.iloc[val_idx_list[Fold]], shuffle=True)
test_set = Data(df_test_encoded, shuffle=False)
logger.info('train_set: {}'.format(len(train_set)))
logger.info('val_set: {}'.format(len(val_set)))
logger.info('test_set: {}'.format(len(test_set)))


def collate_fn(indices):
    batch_prev_items = []
    batch_locale = []
    batch_candidate_set = []
    batch_len = []
    batch_mask = []
    batch_label = []
    batch_label_index = []  # 交叉熵需要的是label在候选集中的index
    batch_seq_cat = []
    batch_seq_num = []
    for item in indices:
        batch_len.append(len(item[0]))  # prev_items
    max_len = max(batch_len)
    for item in indices:
        l = len(item[0])
        batch_mask.append([1] * (l) + [0] * (max_len - l))  # 0代表padding的位置，需要mask
    for item in indices:
        # ['prev_items', 'locale', 'recall', 'next_item', 'seqs_cat_feas', 'seqs_num_feas']
        prev_items = item[0].copy()
        while (len(prev_items) < max_len):
            prev_items.append(id_count)  # embdding的时候id_count+1，把id_count作为padding了
        batch_prev_items.append(prev_items)
        batch_locale.append(item[1])
        batch_candidate_set.append(item[2].copy())
        batch_label.append(item[3])
        if item[3] in item[2]:
            batch_label_index.append(item[2].index(item[3]))
        else:
            batch_label_index.append(len(item[2]))
        batch_seq_cat.append(item[4])
        batch_seq_num.append(item[5])
    return [torch.LongTensor(batch_prev_items).to(device), torch.LongTensor(batch_locale).to(device), 
            torch.LongTensor(batch_candidate_set).to(device),
            torch.LongTensor(batch_len).to(device), torch.LongTensor(batch_mask).to(device), 
            torch.LongTensor(batch_label).to(device), torch.LongTensor(batch_label_index).to(device),
            torch.LongTensor(batch_seq_cat).to(device), torch.FloatTensor(batch_seq_num).to(device)]


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
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
    load_path_opt = '{}/{}_{}_opt.pt'.format(load_dir, load_exp_id, model_name)
    optimizer.load_state_dict(torch.load(load_path_opt, map_location='cpu'))
    # print(model.device)

best_epoch = -1
# 训练
if train:
    logger.info('Training...')
    output_dir = 'ckpt/{}'.format(exp_id)
    ensure_dir(output_dir)
    ac_all = []
    mrr_all = []
    min_val_loss = float('inf')
    max_val_mrr = 0.0
    # best_epoch = -1
    for epoch in range(epochs):
        # train
        logger.info('start train epoch {}'.format(epoch))
        model.train()
        train_loss_list = []
        train_mrr_list = []
        for batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_mask, \
                batch_label, batch_label_index, batch_seq_cat, batch_seq_num in tqdm(train_loader, desc='train model {}'.format(exp_id), total=len(train_loader)):
            optimizer.zero_grad()
            score, loss = model.predict(batch_prev_items=batch_prev_items, batch_locale=batch_locale, 
                                        batch_candidate_set=batch_candidate_set, batch_len=batch_len, 
                                        batch_label=batch_label_index, batch_mask=batch_mask,
                                        batch_seq_cat=batch_seq_cat, batch_seq_num=batch_seq_num)
            loss.backward(retain_graph=True)
            train_loss_list.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            with torch.no_grad():
                sorted_indices = torch.argsort(score, dim=1, descending=True)
                sorted_candidate_set = batch_candidate_set.gather(dim=1, index=sorted_indices)
                for i in range(len(sorted_candidate_set)):
                    pred = sorted_candidate_set[i].tolist()
                    try:
                        pred_result = pred.index(batch_label[i].item())
                        train_mrr_list.append(1 / (pred_result + 1))
                    except:
                        train_mrr_list.append(0)
        train_loss = np.mean(train_loss_list)
        train_mrr = np.mean(train_mrr_list)
        # val
        val_hit = 0
        val_loss_list= []
        val_mrr_list = []
        with torch.no_grad():
            model.eval()
            for batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_mask, \
                    batch_label, batch_label_index, batch_seq_cat, batch_seq_num in tqdm(val_loader, desc='val model {}'.format(exp_id), total=len(val_loader)):
                score, loss = model.predict(batch_prev_items=batch_prev_items, batch_locale=batch_locale, 
                                            batch_candidate_set=batch_candidate_set, batch_len=batch_len, 
                                            batch_label=batch_label_index, batch_mask=batch_mask,
                                            batch_seq_cat=batch_seq_cat, batch_seq_num=batch_seq_num)  # (batch_size, candidate_size)
                val_loss_list.append(loss.item())
                val_hit += score.argmax(dim=-1).eq(batch_label_index).sum().item()
                sorted_indices = torch.argsort(score, dim=1, descending=True)
                sorted_candidate_set = batch_candidate_set.gather(dim=1, index=sorted_indices)
                for i in range(len(sorted_candidate_set)):
                    pred = sorted_candidate_set[i].tolist()
                    try:
                        pred_result = pred.index(batch_label[i].item())
                        val_mrr_list.append(1 / (pred_result + 1))
                    except:
                        val_mrr_list.append(0)
        val_ac = val_hit / len(val_set)
        val_loss = np.mean(val_loss_list)
        val_mrr = np.mean(val_mrr_list)
        mrr_all.append(val_mrr)
        ac_all.append(val_ac)
        lr_scheduler.step(val_mrr)
        lr = optimizer.param_groups[0]['lr']
        logger.info('Train Epoch {}/{}: Train Loss {:.6f}, Train MRR {:.6f}, Val Loss {:.6f}, Val AC {:.6f}, Val MRR {:.6f}, lr {}'.format(
            epoch, epochs, train_loss, train_mrr, val_loss, val_ac, val_mrr, lr))
        if val_mrr > max_val_mrr:
            min_val_loss = val_loss
            max_val_mrr = val_mrr
            best_epoch = epoch
            save_path = '{}/{}_{}.pt'.format(output_dir, exp_id, model_name)
            logger.info('Save model to {}'.format(save_path))
            torch.save(model.state_dict(), save_path)
            save_path_opt = '{}/{}_{}_opt.pt'.format(output_dir, exp_id, model_name)
            torch.save(optimizer.state_dict(), save_path_opt)
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
test_scores = []
test_res = []
model.eval()
for batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_mask, \
        _, batch_label_index, batch_seq_cat, batch_seq_num in tqdm(test_loader, desc='test model {}'.format(exp_id), total=len(test_loader)):
    score, _ = model.predict(batch_prev_items=batch_prev_items, batch_locale=batch_locale, 
                                    batch_candidate_set=batch_candidate_set, batch_len=batch_len, 
                                    batch_label=batch_label_index, batch_mask=batch_mask,
                                    batch_seq_cat=batch_seq_cat, batch_seq_num=batch_seq_num)  # (batch_size, 100)
    test_scores.append(score.detach().cpu().numpy())
    sorted_indices = torch.argsort(score, dim=1, descending=True)
    sorted_candidate_set = batch_candidate_set.gather(dim=1, index=sorted_indices)  # (B, 100)
    test_res.append(sorted_candidate_set.detach().cpu().numpy())
test_scores = np.concatenate(test_scores, axis=0)

logger.info('Decoding the results...')
test_res = np.concatenate(test_res, axis=0)
assert len(test_res) == len(test_preds_encoded)
test_res_unencoded = []
for ind, x in tqdm(enumerate(test_res), total=len(test_res)):
    x = list(x) + test_preds_encoded[ind][len_candidate_set:]
    assert len(x) == 100
    x_unencoded = [id2product[id_] for id_ in x]
    test_res_unencoded.append(x_unencoded)


logger.info('Saving...')
df_test['next_item_prediction'] = test_res_unencoded
logger.info(df_test['next_item_prediction'].apply(len).describe())
logger.info('Save Fold {} res to {}'.format(Fold, 'output/{}_{}_{}_{}.parquet'.format(seed, exp_id, model_name, best_epoch)))
df_test[['locale', 'next_item_prediction']].to_parquet('output/{}_{}_{}_{}.parquet'.format(seed, exp_id, model_name, best_epoch), engine='pyarrow')
logger.info('Save Fold {} score to {}, shape = {}'.format(Fold, 'output/{}_{}_{}_{}.npy'.format(seed, exp_id, model_name, best_epoch), test_scores.shape))
np.save('output/scores_{}_{}_{}_{}_Fold{}.npy'.format(seed, exp_id, model_name, best_epoch, Fold), test_scores)
np.save('output/preds_{}_{}_{}_{}_Fold{}.npy'.format(seed, exp_id, model_name, best_epoch, Fold), test_res)
