import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset

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
