import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset

class Product(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()
        self.emb_dim = config['emb_dim']
        self.len_locale = data_feature['len_locale']
        self.len_encode_brand = data_feature['len_encode_brand']
        self.len_encode_color = data_feature['len_encode_color']
        self.len_encode_size = data_feature['len_encode_size']
        self.len_encode_model = data_feature['len_encode_model']
        self.len_encode_material = data_feature['len_encode_material']
        self.len_encode_author = data_feature['len_encode_author']
        self.dense_bins = data_feature['dense_bins']

        self.locale_emb = nn.Embedding(self.len_locale, self.emb_dim)
        
        self.price_emb = nn.Linear(1, self.emb_dim)
        self.len_title_emb = nn.Linear(1, self.emb_dim)
        self.len_desc_emb = nn.Linear(1, self.emb_dim)
        
        self.encode_brand_emb = nn.Embedding(self.len_encode_brand, self.emb_dim)
        self.encode_color_emb = nn.Embedding(self.len_encode_color, self.emb_dim)
        self.encode_size_emb = nn.Embedding(self.len_encode_size, self.emb_dim)
        self.encode_model_emb = nn.Embedding(self.len_encode_model, self.emb_dim)
        self.encode_material_emb = nn.Embedding(self.len_encode_material, self.emb_dim)
        self.encode_author_emb = nn.Embedding(self.len_encode_author, self.emb_dim)

        self.encode_price_emb = nn.Embedding(self.dense_bins, self.emb_dim)
        self.encode_len_title_emb = nn.Embedding(self.dense_bins, self.emb_dim)
        self.encode_len_desc_emb = nn.Embedding(self.dense_bins, self.emb_dim)

    def forward(self, batch_products):
        """
            batch_products: dict 
        """
        locale_emb = self.locale_emb(batch_products['locale'])
        
        price_emb = self.price_emb(batch_products['price'].unsqueeze(-1))
        len_title_emb = self.len_title_emb(batch_products['len_title'].unsqueeze(-1))
        len_desc_emb = self.len_desc_emb(batch_products['len_desc'].unsqueeze(-1))
        
        encode_brand_emb = self.encode_brand_emb(batch_products['encode_brand'])
        encode_color_emb = self.encode_color_emb(batch_products['encode_color'])
        encode_size_emb = self.encode_size_emb(batch_products['encode_size'])
        encode_model_emb = self.encode_model_emb(batch_products['encode_model'])
        encode_material_emb = self.encode_material_emb(batch_products['encode_material'])
        encode_author_emb = self.encode_author_emb(batch_products['encode_author'])

        encode_price_emb = self.encode_price_emb(batch_products['encode_price'])
        encode_len_title_emb = self.encode_len_title_emb(batch_products['encode_len_title'])
        encode_len_desc_emb = self.encode_len_desc_emb(batch_products['encode_len_desc'])

        # 将所有特征的表征按照一定的方式组合起来得到这个产品的向量表示
        products_vec = torch.cat([locale_emb, 
                                  price_emb, len_title_emb, len_desc_emb,
                                  encode_brand_emb, encode_color_emb, encode_size_emb, 
                                  encode_model_emb, encode_material_emb, encode_author_emb, 
                                  encode_price_emb, encode_len_title_emb, encode_len_desc_emb], dim=1)
        return products_vec


class ProductEmbedding(nn.Module):

    def __init__(self, config, data_feature, products_input, word2vec_embedding, titles_embedding, descs_embedding):
        super().__init__()
        self.emb_dim = config['emb_dim']
        self.device = config['device']
        self.product_fea = Product(config, data_feature).to(self.device)
        self.id_count = data_feature['id_count']
        self.w2v_vector_size = data_feature['w2v_vector_size']
        self.sentence_vector_size = data_feature['sentence_vector_size']

        product_fea_emb = self.product_fea(products_input)  # (id_count, 208)
        self.padding_emb = torch.zeros((1, product_fea_emb.shape[1]), requires_grad=False).to(self.device)
        self.product_fea_emb = torch.cat([product_fea_emb, self.padding_emb], dim=0)  # (id_count + 1, 208)
        self.title_emb = nn.Embedding(self.id_count + 1, embedding_dim=self.sentence_vector_size, padding_idx=self.id_count)
        self.title_linear = nn.Linear(self.sentence_vector_size, self.emb_dim * 2)
        self.desc_emb = nn.Embedding(self.id_count + 1, embedding_dim=self.sentence_vector_size, padding_idx=self.id_count)
        self.desc_linear = nn.Linear(self.sentence_vector_size, self.emb_dim * 2)
        self.w2v_emb = nn.Embedding(self.id_count + 1, embedding_dim=self.w2v_vector_size, padding_idx=self.id_count)
        self.w2v_linear = nn.Linear(self.w2v_vector_size, self.emb_dim * 2)
        self.title_emb.weight.data[:self.id_count].copy_(torch.tensor(titles_embedding))
        self.desc_emb.weight.data[:self.id_count].copy_(torch.tensor(descs_embedding))
        self.w2v_emb.weight.data[:self.id_count].copy_(torch.tensor(word2vec_embedding))
        self.title_emb.weight.requires_grad = False
        self.desc_emb.weight.requires_grad = False
        self.w2v_emb.weight.requires_grad = False

    def forward(self, batch_prev_items, batch_candidate_set=None):
        """
            batch_prev_items: (B, len)
            batch_candidate_set: (B, 100)
        """
        # print(batch_prev_items.shape, batch_candidate_set.shape, batch_label.shape)
        # print(self.product_fea_emb[batch_prev_items].shape)
        # 对输入序列中的每个商品，获取它的嵌入表示并拼接
        batch_prev_items_emb = torch.cat([
            self.product_fea_emb[batch_prev_items],  # 商品特征嵌入
            self.title_linear(self.title_emb(batch_prev_items)), 
            self.desc_linear(self.desc_emb(batch_prev_items)), 
            self.w2v_linear(self.w2v_emb(batch_prev_items))
        ], dim=-1)
        # 对候选集中的每个商品，获取它的嵌入表示并拼接
        if batch_candidate_set is not None:
            batch_candidate_set_emb = torch.cat([
                self.product_fea_emb[batch_candidate_set],  # 商品特征嵌入
                self.title_linear(self.title_emb(batch_candidate_set)), 
                self.desc_linear(self.desc_emb(batch_candidate_set)), 
                self.w2v_linear(self.w2v_emb(batch_candidate_set))
            ], dim=-1)
        else:
            batch_candidate_set_emb = None
        # # 对标签序列中的每个商品，获取它的嵌入表示并拼接
        # batch_label_emb = torch.cat([
        #     self.product_fea_emb[batch_label],  # 商品特征嵌入
        #     self.title_linear(self.title_emb(batch_label)), 
        #     self.desc_linear(self.desc_emb(batch_label)), 
        #     self.w2v_linear(self.w2v_emb(batch_label))
        # ], dim=-1)
        # print(batch_prev_items_emb.shape, batch_candidate_set_emb.shape, batch_label_emb.shape)
        return batch_prev_items_emb, batch_candidate_set_emb


class IntraAttention(nn.Module):
    """对序列经过 LSTM 后的隐藏层向量序列做 Attention 强化
    key: 当前序列经过 LSTM 后的隐藏层向量序列
    query: 轨迹向量序列的最后一个状态
    """

    def __init__(self, hidden_size):
        super(IntraAttention, self).__init__()
        # 模型参数
        self.hidden_size = hidden_size
        # 模型结构
        self.w1 = nn.Linear(in_features=self.hidden_size, out_features=1, bias=False)
        self.w2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.w3 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)

    def forward(self, query, key, mask=None):
        """前馈

        Args:
            query (tensor): shape (batch_size, hidden_size)
            key (tensor): shape (batch_size, seq_len, hidden_size)
            mask (tensor): padding mask, 1 表示非补齐值, 0 表示补齐值 shape (batch_size, seq_len)
        Return:
            attn_hidden (tensor): shape (batch_size, hidden_size)
        """
        attn_weight = torch.bmm(key, query.unsqueeze(2)).squeeze(2) # shape (batch_size, seq_len)
        if mask is not None:
            mask = attn_weight.masked_fill(mask==0, -1e9) # mask 
        attn_weight = torch.softmax(attn_weight, dim=1).unsqueeze(2) # shape (batch_size, seq_len, 1)
        attn_hidden = torch.sum(attn_weight * key, dim=1)
        return attn_hidden
    

class BaseModel(nn.Module):

    def __init__(self, config, data_feature, products_input, word2vec_embedding, titles_embedding, descs_embedding):
        super(BaseModel, self).__init__()

        self.hidden_size = config['hid_dim']
        self.device = config['device']
        self.layers = config['layers']
        self.dropout = config['dropout']
        self.emb_dim = config['emb_dim']
        self.bidirectional  = config['bidirectional']
        if self.bidirectional:
            self.num_direction = 2
        else:
            self.num_direction = 1
        self.id_count = data_feature['id_count']

        self.input_size = data_feature['len_features'] * self.emb_dim + data_feature['len_emb_features'] * 2 * self.emb_dim

        self.product_emb = ProductEmbedding(config, data_feature, products_input, word2vec_embedding, titles_embedding, descs_embedding).to(self.device)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.layers, batch_first=True, bidirectional=self.bidirectional)
        self.intra_attn = IntraAttention(hidden_size=self.hidden_size * self.num_direction)
        self.dropout = nn.Dropout(p=self.dropout)
        self.output = nn.Linear(in_features=self.hidden_size * self.num_direction, out_features=self.id_count)  # 所有的id的个数

        self.loss_func = nn.CrossEntropyLoss(ignore_index=data_feature['len_candidate_set'])  # 候选集大小100

    def forward(self, batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_mask=None, train=True):
        """
        Args:
            batch_prev_items: (B, len)
            batch_locale: (B,)
            batch_candidate_set: (B, 100)
            batch_len: (B,)
            batch_label: (B,)
            batch_mask: (B, len)
        
        Return:
            candidate_prob (tensor): 对候选集下一跳的概率预测 (batch_size, candidate_size)
        """
        batch_prev_items_emb, batch_candidate_set_emb = self.product_emb(batch_prev_items, None)
        # batch_prev_items_emb (B, len, input_size)
        # batch_candidate_set_emb (B, 100, input_size) or None

        input_emb = self.dropout(batch_prev_items_emb)
        
        self.lstm.flatten_parameters()

        if batch_mask is not None:
            # LSTM with Mask
            pack_input = pack_padded_sequence(input_emb, lengths=batch_len.cpu(), batch_first=True, enforce_sorted=False)
            pack_lstm_hidden, (hn, cn) = self.lstm(pack_input)
            lstm_hidden, _ = pad_packed_sequence(pack_lstm_hidden, batch_first=True) # (B, len, hidden_size)
        else:
            lstm_hidden, (hn, cn) = self.lstm(input_emb) # (B, len, hidden_size)
        
        if batch_mask is not None:
            # 获取序列最后一个非补齐值对应的 hidden
            lstm_last_index = batch_len - 1 # (batch_size)
            lstm_last_index = lstm_last_index.reshape(lstm_last_index.shape[0], 1, -1) # (B, 1, 1)
            lstm_last_index = lstm_last_index.repeat(1, 1, self.hidden_size * self.num_direction) # (B, 1, hidden_size)
            lstm_last_hidden = torch.gather(lstm_hidden, dim=1, index=lstm_last_index).squeeze(1) # (B, hidden_size)
        else:
            lstm_last_hidden = lstm_hidden[:, -1, :] # (B, hidden_size)
        attn_hidden = self.intra_attn(query=lstm_last_hidden, key=lstm_hidden, mask=batch_mask) # (B, hidden_size)
        attn_hidden = self.dropout(attn_hidden) # (B, hidden_size)
        
        # 使用线性层直接预测
        score = self.output(attn_hidden)  # (batch_size, id_count)
        
        # 根据 candidate_set 选出对应 candidate 的 score
        candidate_score = torch.gather(score, dim=1, index=batch_candidate_set)  # (batch_size, candidate_count)
        return candidate_score

    def predict(self, batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_mask=None):
        """预测
        Return:
            candidate_prob (tensor): softmax 后对候选集下一跳的概率预测 (batch_size, candidate_size)
        """
        score = self.forward(batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_mask, False)
        loss = self.loss_func(score, batch_label)
        return torch.softmax(score, dim=1), loss

    def calculate_loss(self, batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_mask=None):
        """
        Return:
            loss (tensor): 交叉损失熵 (1)
        """
        score = self.forward(batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_mask, True)
        loss = self.loss_func(score, batch_label)
        return loss


class Matcher(nn.Module):
    """Matcher 匹配打分
    根据当前轨迹隐藏层表征与候选集表征之间计算一个匹配程度，也就是下一跳的评分

    目前因为候选集表征与隐藏层表征维度不一样，所以对候选集表征过一个线性映射来计算。
    """

    def __init__(self, hidden_size, item_emb_size):
        super(Matcher, self).__init__()
        self.hidden_size = hidden_size
        self.item_emb_size = item_emb_size
        self.linear = nn.Linear(in_features=self.item_emb_size, out_features=self.hidden_size)

    def forward(self, items_hidden, candidate_emb):
        """前馈

        Args:
            items_hidden (tensor): 历史序列的隐藏层表征 (batch_size, hidden_size)
            candidate_emb (tensor): 候选集表征 (batch_size, candidate_size, item_emb_size)
        """
        candidate_hidden = self.linear(candidate_emb).permute(0, 2, 1) # (batch_size, hidden_size, candidate_size)
        score = torch.bmm(items_hidden.unsqueeze(1), candidate_hidden).squeeze(1) # (batch_size, candidate_size)
        return score


class MatcherV2(nn.Module):
    """候选集与当前轨迹状态之间的注意力模块
    """

    def __init__(self, hidden_size, item_emb_size, dropout):
        super(MatcherV2, self).__init__()
        self.out_linear = nn.Linear(in_features=hidden_size, out_features=1, bias=False)
        self.w1_linear = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.w2_linear = nn.Linear(in_features=item_emb_size, out_features=hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key):
        """
        计算 query 与 key 之间的相似度
        计算方法为可学习前馈神经网络：attn_weight = w_out * tanh(w_1 * query + w_2 * key)
        Args:
            query: 历史序列的隐藏层表征 shape: (batch_size, hidden_size)
            key: 候选集的嵌入向量 shape: (batch_size, candidate_size, item_emb_size)
            hidden_size = item_emb_size

        Returns:
            candidate_weight: 当前状态与候选集之间的相关性向量。shape: (batch_size, candidate_size)
        """
        query_hidden = torch.relu(self.w1_linear(query).unsqueeze(1))  # shape: (batch_size, 1, hidden_size)
        key_hidden = torch.relu(self.w2_linear(key))  # shape: (batch_size, candidate_size, hidden_size)
        candidate_weight = torch.tanh(query_hidden + key_hidden)  # shape: (batch_size, candidate_size, hidden_size)
        candidate_weight = self.dropout(candidate_weight)
        out = self.out_linear(candidate_weight).squeeze(2)  # shape: (batch_size, candidate_size)
        return out
    

class SeqFeatureEmbedding(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()
        self.emb_dim = config['emb_dim']
        self.hid_dim = config['hid_dim']
        self.len_seqs_num_feas = data_feature['len_seqs_num_feas']
        self.len_seqs_cat_feas = data_feature['len_seqs_cat_feas']
        self.seq_emb_factor = data_feature['seq_emb_factor']
        self.idNUNIQUE = data_feature['idNUNIQUE']
        self.idCOUNT = data_feature['idCOUNT']

        self.brandNUNIQUE = data_feature['brandNUNIQUE']
        self.brandCOUNT = data_feature['brandCOUNT']

        self.colorNUNIQUE = data_feature['colorNUNIQUE']
        self.colorCOUNT = data_feature['colorCOUNT']

        self.sizeNUNIQUE = data_feature['sizeNUNIQUE']
        self.sizeCOUNT = data_feature['sizeCOUNT']

        self.modelNUNIQUE = data_feature['modelNUNIQUE']
        self.modelCOUNT = data_feature['modelCOUNT']

        self.materialNUNIQUE = data_feature['materialNUNIQUE']
        self.materialCOUNT = data_feature['materialCOUNT']

        self.authorNUNIQUE = data_feature['authorNUNIQUE']
        self.authorCOUNT = data_feature['authorCOUNT']

        self.idNUNIQUE_emb = nn.Embedding(self.idNUNIQUE, self.emb_dim)
        self.idCOUNT_emb = nn.Embedding(self.idCOUNT, self.emb_dim)
        self.brandNUNIQUE_emb = nn.Embedding(self.brandNUNIQUE, self.emb_dim)
        self.brandCOUNT_emb = nn.Embedding(self.brandCOUNT, self.emb_dim)
        self.colorNUNIQUE_emb = nn.Embedding(self.colorNUNIQUE, self.emb_dim)
        self.colorCOUNT_emb = nn.Embedding(self.colorCOUNT, self.emb_dim)
        self.sizeNUNIQUE_emb = nn.Embedding(self.sizeNUNIQUE, self.emb_dim)
        self.sizeCOUNT_emb = nn.Embedding(self.sizeCOUNT, self.emb_dim)
        self.modelNUNIQUE_emb = nn.Embedding(self.modelNUNIQUE, self.emb_dim)
        self.modelCOUNT_emb = nn.Embedding(self.modelCOUNT, self.emb_dim)
        self.materialNUNIQUE_emb = nn.Embedding(self.materialNUNIQUE, self.emb_dim)
        self.materialCOUNT_emb = nn.Embedding(self.materialCOUNT, self.emb_dim)
        self.authorNUNIQUE_emb = nn.Embedding(self.authorNUNIQUE, self.emb_dim)
        self.authorCOUNT_emb = nn.Embedding(self.authorCOUNT, self.emb_dim)

        self.seq_cat_emb = nn.Sequential(nn.Linear(self.emb_dim * self.len_seqs_cat_feas, self.hid_dim),
                                         nn.ReLU(),
                                         nn.Linear(self.hid_dim, self.emb_dim * (self.seq_emb_factor // 2)))
        self.seq_num_emb = nn.Sequential(nn.Linear(self.len_seqs_num_feas, self.hid_dim),
                                         nn.ReLU(),
                                         nn.Linear(self.hid_dim, self.emb_dim * (self.seq_emb_factor // 2)))

    def forward(self, batch_seq_cat, batch_seq_num):
        """
            batch_seq_cat: (B, len_seqs_cat_feas)
            batch_seq_num: (B, len_seqs_num_feas)
        """
        idNUNIQUE_emb = self.idNUNIQUE_emb(batch_seq_cat[:, 0])
        idCOUNT_emb = self.idCOUNT_emb(batch_seq_cat[:, 1])
        brandNUNIQUE_emb = self.brandNUNIQUE_emb(batch_seq_cat[:, 2])
        brandCOUNT_emb = self.brandCOUNT_emb(batch_seq_cat[:, 3])
        colorNUNIQUE_emb = self.colorNUNIQUE_emb(batch_seq_cat[:, 4])
        colorCOUNT_emb = self.colorCOUNT_emb(batch_seq_cat[:, 5])
        sizeNUNIQUE_emb = self.sizeNUNIQUE_emb(batch_seq_cat[:, 6])
        sizeCOUNT_emb = self.sizeCOUNT_emb(batch_seq_cat[:, 7])
        modelNUNIQUE_emb = self.modelNUNIQUE_emb(batch_seq_cat[:, 8])
        modelCOUNT_emb = self.modelCOUNT_emb(batch_seq_cat[:, 9])
        materialNUNIQUE_emb = self.materialNUNIQUE_emb(batch_seq_cat[:, 10])
        materialCOUNT_emb = self.materialCOUNT_emb(batch_seq_cat[:, 11])
        authorNUNIQUE_emb = self.authorNUNIQUE_emb(batch_seq_cat[:, 12])
        authorCOUNT_emb = self.authorCOUNT_emb(batch_seq_cat[:, 13])

        sparse_vec = torch.cat([idNUNIQUE_emb, idCOUNT_emb,
                                  brandNUNIQUE_emb, brandCOUNT_emb,
                                  colorNUNIQUE_emb, colorCOUNT_emb, 
                                  sizeNUNIQUE_emb, sizeCOUNT_emb, 
                                  modelNUNIQUE_emb, modelCOUNT_emb,
                                  materialNUNIQUE_emb, materialCOUNT_emb,
                                  authorNUNIQUE_emb, authorCOUNT_emb
                                  ], dim=1)  # (B, 14 * emb_dim)
        sparse_vec = self.seq_cat_emb(sparse_vec)   # (B, 2 * emb_dim)
        dense_vec = self.seq_num_emb(batch_seq_num)  # (B, 2 * emb_dim)
        return torch.cat([sparse_vec, dense_vec], dim=1)   # (B, 4 * emb_dim)


# class SeqFeatureEmbedding(nn.Module):

#     def __init__(self, config, data_feature):
#         super().__init__()
#         self.emb_dim = config['emb_dim']
#         self.hid_dim = config['hid_dim']
#         self.len_seqs_num_feas = data_feature['len_seqs_num_feas']
#         self.len_seqs_cat_feas = data_feature['len_seqs_cat_feas']
#         self.seq_emb_factor = data_feature['seq_emb_factor']
#         self.idNUNIQUE = data_feature['idNUNIQUE']
#         self.idCOUNT = data_feature['idCOUNT']

#         self.brandNUNIQUE = data_feature['brandNUNIQUE']
#         self.brandCOUNT = data_feature['brandCOUNT']

#         self.colorNUNIQUE = data_feature['colorNUNIQUE']
#         self.colorCOUNT = data_feature['colorCOUNT']

#         self.sizeNUNIQUE = data_feature['sizeNUNIQUE']
#         self.sizeCOUNT = data_feature['sizeCOUNT']

#         self.modelNUNIQUE = data_feature['modelNUNIQUE']
#         self.modelCOUNT = data_feature['modelCOUNT']

#         self.materialNUNIQUE = data_feature['materialNUNIQUE']
#         self.materialCOUNT = data_feature['materialCOUNT']

#         self.authorNUNIQUE = data_feature['authorNUNIQUE']
#         self.authorCOUNT = data_feature['authorCOUNT']

#         self.seq_cat_emb = nn.Sequential(nn.Linear(self.len_seqs_cat_feas, self.hid_dim),
#                                          nn.ReLU(),
#                                          nn.Linear(self.hid_dim, self.emb_dim * (self.seq_emb_factor // 2)))
#         self.seq_num_emb = nn.Sequential(nn.Linear(self.len_seqs_num_feas, self.hid_dim),
#                                          nn.ReLU(),
#                                          nn.Linear(self.hid_dim, self.emb_dim * (self.seq_emb_factor // 2)))

#     def forward(self, batch_seq_cat, batch_seq_num):
#         """
#             batch_seq_cat: (B, len_seqs_cat_feas)
#             batch_seq_num: (B, len_seqs_num_feas)
#         """
#         sparse_vec = self.seq_cat_emb(batch_seq_cat.float())   # (B, 2 * emb_dim)
#         dense_vec = self.seq_num_emb(batch_seq_num)  # (B, 2 * emb_dim)
#         return torch.cat([sparse_vec, dense_vec], dim=1)   # (B, 4 * emb_dim)
    
class MatchModel(nn.Module):

    def __init__(self, config, data_feature, products_input, word2vec_embedding, titles_embedding, descs_embedding):
        super(MatchModel, self).__init__()

        self.hidden_size = config['hid_dim']
        self.device = config['device']
        self.layers = config['layers']
        self.dropout_p = config['dropout']
        self.emb_dim = config['emb_dim']
        self.attn_match = config['attn_match']
        self.bidirectional  = config['bidirectional']
        if self.bidirectional:
            self.num_direction = 2
        else:
            self.num_direction = 1
        self.id_count = data_feature['id_count']

        self.input_size = data_feature['len_features'] * self.emb_dim + data_feature['len_emb_features'] * 2 * self.emb_dim

        self.product_emb = ProductEmbedding(config, data_feature, products_input, word2vec_embedding, titles_embedding, descs_embedding).to(self.device)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.layers, batch_first=True, bidirectional=self.bidirectional)
        self.intra_attn = IntraAttention(hidden_size=self.hidden_size * self.num_direction)
        self.dropout = nn.Dropout(p=self.dropout_p)
        if self.attn_match:
            self.output = MatcherV2(hidden_size=self.hidden_size * self.num_direction, item_emb_size=self.input_size, dropout=self.dropout_p)
        else:
            self.output = Matcher(hidden_size=self.hidden_size * self.num_direction, item_emb_size=self.input_size)

        self.loss_func = nn.CrossEntropyLoss(ignore_index=data_feature['len_candidate_set'])  # 候选集大小100

    def forward(self, batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_mask=None, train=True):
        """
        Args:
            batch_prev_items: (B, len)
            batch_locale: (B,)
            batch_candidate_set: (B, 100)
            batch_len: (B,)
            batch_label: (B,)
            batch_mask: (B, len)
        
        Return:
            candidate_prob (tensor): 对候选集下一跳的概率预测 (batch_size, candidate_size)
        """
        batch_prev_items_emb, batch_candidate_set_emb = self.product_emb(batch_prev_items, batch_candidate_set)
        # batch_prev_items_emb (B, len, input_size)
        # batch_candidate_set_emb (B, 100, input_size) or None

        input_emb = self.dropout(batch_prev_items_emb)
        
        self.lstm.flatten_parameters()

        if batch_mask is not None:
            # LSTM with Mask
            pack_input = pack_padded_sequence(input_emb, lengths=batch_len.cpu(), batch_first=True, enforce_sorted=False)
            pack_lstm_hidden, (hn, cn) = self.lstm(pack_input)
            lstm_hidden, _ = pad_packed_sequence(pack_lstm_hidden, batch_first=True) # (B, len, hidden_size)
        else:
            lstm_hidden, (hn, cn) = self.lstm(input_emb) # (B, len, hidden_size)
        
        if batch_mask is not None:
            # 获取序列最后一个非补齐值对应的 hidden
            lstm_last_index = batch_len - 1 # (batch_size)
            lstm_last_index = lstm_last_index.reshape(lstm_last_index.shape[0], 1, -1) # (B, 1, 1)
            lstm_last_index = lstm_last_index.repeat(1, 1, self.hidden_size * self.num_direction) # (B, 1, hidden_size)
            lstm_last_hidden = torch.gather(lstm_hidden, dim=1, index=lstm_last_index).squeeze(1) # (B, hidden_size)
        else:
            lstm_last_hidden = lstm_hidden[:, -1, :] # (B, hidden_size)

        attn_hidden = self.intra_attn(query=lstm_last_hidden, key=lstm_hidden, mask=batch_mask) # (B, hidden_size)
        attn_hidden = self.dropout(attn_hidden) # (B, hidden_size)
        
        # Matcher
        candidate_score = self.output(attn_hidden, batch_candidate_set_emb)  # (batch_size, candidate_size)
        return candidate_score

    def predict(self, batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_mask=None):
        """预测
        Return:
            candidate_prob (tensor): softmax 后对候选集下一跳的概率预测 (batch_size, candidate_size)
        """
        score = self.forward(batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_mask, False)
        loss = self.loss_func(score, batch_label)
        return torch.softmax(score, dim=1), loss

    def calculate_loss(self, batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_mask=None):
        """
        Return:
            loss (tensor): 交叉损失熵 (1)
        """
        score = self.forward(batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_mask, True)
        loss = self.loss_func(score, batch_label)
        return loss


class MatchModelV2(nn.Module):

    def __init__(self, config, data_feature, products_input, word2vec_embedding, titles_embedding, descs_embedding):
        super(MatchModelV2, self).__init__()

        self.hidden_size = config['hid_dim']
        self.emb_dim = config['emb_dim']
        self.device = config['device']
        self.layers = config['layers']
        self.dropout_p = config['dropout']
        self.emb_dim = config['emb_dim']
        self.attn_match = config['attn_match']
        self.bidirectional  = config['bidirectional']
        if self.bidirectional:
            self.num_direction = 2
        else:
            self.num_direction = 1
        self.id_count = data_feature['id_count']

        self.input_size = data_feature['len_features'] * self.emb_dim + data_feature['len_emb_features'] * 2 * self.emb_dim

        self.product_emb = ProductEmbedding(config, data_feature, products_input, word2vec_embedding, titles_embedding, descs_embedding).to(self.device)
        self.seq_fea_emb = SeqFeatureEmbedding(config, data_feature).to(self.device)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.layers, batch_first=True, bidirectional=self.bidirectional).to(self.device)
        self.intra_attn = IntraAttention(hidden_size=self.hidden_size * self.num_direction).to(self.device)
        self.dropout = nn.Dropout(p=self.dropout_p).to(self.device)
        if self.attn_match:
            self.output = MatcherV2(hidden_size=self.hidden_size * self.num_direction + data_feature['seq_emb_factor'] * self.emb_dim, 
                                    item_emb_size=self.input_size, dropout=self.dropout_p).to(self.device)
        else:
            self.output = Matcher(hidden_size=self.hidden_size * self.num_direction + data_feature['seq_emb_factor'] * self.emb_dim, 
                                  item_emb_size=self.input_size).to(self.device)

        self.loss_func = nn.CrossEntropyLoss(ignore_index=data_feature['len_candidate_set']).to(self.device)  # 候选集大小100

    def forward(self, batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_seq_cat, batch_seq_num, batch_mask=None, train=True):
        """
        Args:
            batch_prev_items: (B, len)
            batch_locale: (B,)
            batch_candidate_set: (B, 100)
            batch_len: (B,)
            batch_label: (B,)
            batch_mask: (B, len)
            batch_seq_cat: (B, len_seqs_cat_feas)
            batch_seq_num: (B, len_seqs_num_feas)
        
        Return:
            candidate_prob (tensor): 对候选集下一跳的概率预测 (batch_size, candidate_size)
        """
        batch_prev_items_emb, batch_candidate_set_emb = self.product_emb(batch_prev_items, batch_candidate_set)
        # batch_prev_items_emb (B, len, input_size)
        # batch_candidate_set_emb (B, 100, input_size) or None

        input_emb = self.dropout(batch_prev_items_emb)
        
        self.lstm.flatten_parameters()

        if batch_mask is not None:
            # LSTM with Mask
            pack_input = pack_padded_sequence(input_emb, lengths=batch_len.cpu(), batch_first=True, enforce_sorted=False)
            pack_lstm_hidden, (hn, cn) = self.lstm(pack_input)
            lstm_hidden, _ = pad_packed_sequence(pack_lstm_hidden, batch_first=True) # (B, len, hidden_size)
        else:
            lstm_hidden, (hn, cn) = self.lstm(input_emb) # (B, len, hidden_size)
        
        if batch_mask is not None:
            # 获取序列最后一个非补齐值对应的 hidden
            lstm_last_index = batch_len - 1 # (batch_size)
            lstm_last_index = lstm_last_index.reshape(lstm_last_index.shape[0], 1, -1) # (B, 1, 1)
            lstm_last_index = lstm_last_index.repeat(1, 1, self.hidden_size * self.num_direction) # (B, 1, hidden_size)
            lstm_last_hidden = torch.gather(lstm_hidden, dim=1, index=lstm_last_index).squeeze(1) # (B, hidden_size)
        else:
            lstm_last_hidden = lstm_hidden[:, -1, :] # (B, hidden_size)

        attn_hidden = self.intra_attn(query=lstm_last_hidden, key=lstm_hidden, mask=batch_mask) # (B, hidden_size)
        attn_hidden = self.dropout(attn_hidden) # (B, hidden_size)

        # 人工序列特征
        seq_fea = self.seq_fea_emb(batch_seq_cat, batch_seq_num)  # (B, data_feature['seq_emb_factor'] * emb_dim)
        query = torch.cat([attn_hidden, seq_fea], dim=1)  # (B, hidden_size + data_feature['seq_emb_factor'] * emb_dim)
        # Matcher
        candidate_score = self.output(query, batch_candidate_set_emb)  # (batch_size, candidate_size)
        return candidate_score

    def predict(self, batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_seq_cat, batch_seq_num, batch_mask=None):
        """预测
        Return:
            candidate_prob (tensor): softmax 后对候选集下一跳的概率预测 (batch_size, candidate_size)
        """
        score = self.forward(batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_seq_cat, batch_seq_num, batch_mask, False)
        loss = self.loss_func(score, batch_label)
        return torch.softmax(score, dim=1), loss

    def calculate_loss(self, batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_seq_cat, batch_seq_num, batch_mask=None):
        """
        Return:
            loss (tensor): 交叉损失熵 (1)
        """
        score = self.forward(batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_seq_cat, batch_seq_num, batch_mask, True)
        loss = self.loss_func(score, batch_label)
        return loss

