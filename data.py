import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset
import numpy as np

# class NNDataset(Dataset):
#     def __init__(self, df):
#         super(NNDataset, self).__init__()
#         self.df = df.copy()
    
#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, idx):
#         # ['prev_items', 'next_item', 'locale', 'last_item', 'recall']
#         sample = self.df.iloc[idx].values
#         return sample[0], sample[2], sample[4], sample[1]

class NNDataset(Dataset):
    def __init__(self, df):
        super(NNDataset, self).__init__()
        self.df = df.copy().values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # ['prev_items', 'next_item', 'locale', 'last_item', 'recall']
        sample = self.df[idx]
        return sample[0], sample[2], sample[4], sample[1]
    
# class NNDatasetV2(Dataset):
#     def __init__(self, df, df_seq_cat_fea, df_seq_num_fea):
#         super(NNDatasetV2, self).__init__()
#         self.df = df.copy()
#         self.df_seq_cat_fea = df_seq_cat_fea.copy()
#         self.df_seq_num_fea = df_seq_num_fea.copy()
#         assert len(self.df) == len(self.df_seq_cat_fea)
    
#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, idx):
#         # ['prev_items', 'next_item', 'locale', 'last_item', 'recall']
#         sample = self.df.iloc[idx].values
#         return sample[0], sample[2], sample[4], sample[1], self.df_seq_cat_fea.iloc[idx].values, self.df_seq_num_fea.iloc[idx].values


class NNDatasetV2(Dataset):
    def __init__(self, df, df_seq_cat_fea, df_seq_num_fea):
        super(NNDatasetV2, self).__init__()
        self.df = df.copy().values
        self.df_seq_cat_fea = df_seq_cat_fea.copy().values
        self.df_seq_num_fea = df_seq_num_fea.copy().values
        assert len(self.df) == len(self.df_seq_cat_fea)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # ['prev_items', 'next_item', 'locale', 'last_item', 'recall']
        sample = self.df[idx]
        return sample[0], sample[2], sample[4], sample[1], self.df_seq_cat_fea[idx], self.df_seq_num_fea[idx]

   
class NNDatasetV3(Dataset):
    def __init__(self, df, df_seq_cat_fea, df_seq_num_fea):
        super(NNDatasetV3, self).__init__()
        self.df = df.copy().values
        self.df_seq_cat_fea = df_seq_cat_fea.copy().values
        self.df_seq_num_fea = df_seq_num_fea.copy().values
        assert len(self.df) == len(self.df_seq_cat_fea)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # ['prev_items', 'next_item', 'locale', 'recall', 'label', 'index']
        sample = self.df[idx]
        return sample[0], sample[2], sample[3], sample[1], sample[4], sample[5], self.df_seq_cat_fea[idx], self.df_seq_num_fea[idx]


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data.copy()['prev_items'].values
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data.copy()['next_item'].values)
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items, mask, targets

