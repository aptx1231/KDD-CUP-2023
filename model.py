import torch
import math
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F


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
        self.div = config['div']

        self.locale_emb = nn.Embedding(self.len_locale, self.emb_dim // self.div)
        
        self.price_emb = nn.Linear(1, self.emb_dim // self.div)
        self.len_title_emb = nn.Linear(1, self.emb_dim // self.div)
        self.len_desc_emb = nn.Linear(1, self.emb_dim // self.div)
        
        self.encode_brand_emb = nn.Embedding(self.len_encode_brand, self.emb_dim // self.div)
        self.encode_color_emb = nn.Embedding(self.len_encode_color, self.emb_dim // self.div)
        self.encode_size_emb = nn.Embedding(self.len_encode_size, self.emb_dim // self.div)
        self.encode_model_emb = nn.Embedding(self.len_encode_model, self.emb_dim // self.div)
        self.encode_material_emb = nn.Embedding(self.len_encode_material, self.emb_dim // self.div)
        self.encode_author_emb = nn.Embedding(self.len_encode_author, self.emb_dim // self.div)

        self.encode_price_emb = nn.Embedding(self.dense_bins, self.emb_dim // self.div)
        self.encode_len_title_emb = nn.Embedding(self.dense_bins, self.emb_dim // self.div)
        self.encode_len_desc_emb = nn.Embedding(self.dense_bins, self.emb_dim // self.div)

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
        self.add_title = config['add_title']
        self.add_desc = config['add_desc']
        self.add_w2v = config['add_w2v']
        self.add_pemb = config['add_pemb']
        self.product_fea = Product(config, data_feature).to(self.device)
        self.id_count = data_feature['id_count']
        self.w2v_vector_size = data_feature['w2v_vector_size']
        self.sentence_vector_size = data_feature['sentence_vector_size']
        self.pro_emb_factor = config['pro_emb_factor']
        self.grad_title = config['grad_title']
        self.grad_desc = config['grad_desc']
        self.grad_w2v = config['grad_w2v']
        self.grad_pemb = config['grad_pemb']
        self.pca = config['pca']
        self.pca_dim = config['pca_dim']

        product_fea_emb = self.product_fea(products_input)  # (id_count, 208)
        self.padding_emb = torch.zeros((1, product_fea_emb.shape[1]), requires_grad=False).to(self.device)
        self.product_fea_emb = torch.cat([product_fea_emb, self.padding_emb], dim=0)  # (id_count + 1, 208)
        if not self.pca:
            if self.add_title:
                self.title_emb = nn.Embedding(self.id_count + 1, embedding_dim=self.sentence_vector_size, padding_idx=self.id_count)
                self.title_linear = nn.Linear(self.sentence_vector_size, self.emb_dim * self.pro_emb_factor)
                self.title_emb.weight.data[:self.id_count].copy_(torch.tensor(titles_embedding))
                self.title_emb.weight.requires_grad = self.grad_title
            if self.add_desc:
                self.desc_emb = nn.Embedding(self.id_count + 1, embedding_dim=self.sentence_vector_size, padding_idx=self.id_count)
                self.desc_linear = nn.Linear(self.sentence_vector_size, self.emb_dim * self.pro_emb_factor)
                self.desc_emb.weight.data[:self.id_count].copy_(torch.tensor(descs_embedding))
                self.desc_emb.weight.requires_grad = self.grad_desc 
        else:
            if self.add_title:
                self.title_emb = nn.Embedding(self.id_count + 1, embedding_dim=self.pca_dim, padding_idx=self.id_count)
                self.title_emb.weight.data[:self.id_count].copy_(torch.tensor(titles_embedding))
                # self.title_emb.weight.requires_grad = self.grad_title
            if self.add_desc:
                self.desc_emb = nn.Embedding(self.id_count + 1, embedding_dim=self.pca_dim, padding_idx=self.id_count)
                self.desc_emb.weight.data[:self.id_count].copy_(torch.tensor(descs_embedding))
                # self.desc_emb.weight.requires_grad = self.grad_desc 
        
        if self.add_w2v:
            self.w2v_emb = nn.Embedding(self.id_count + 1, embedding_dim=self.w2v_vector_size, padding_idx=self.id_count)
            self.w2v_linear = nn.Linear(self.w2v_vector_size, self.emb_dim * self.pro_emb_factor)
            self.w2v_emb.weight.data[:self.id_count].copy_(torch.tensor(word2vec_embedding))
            self.w2v_emb.weight.requires_grad = self.grad_w2v
        if self.add_pemb:
            self.pemb = nn.Embedding(self.id_count + 1, embedding_dim=self.emb_dim * self.pro_emb_factor, padding_idx=self.id_count)
            self.pemb.weight.requires_grad = self.grad_pemb

    def forward(self, batch_prev_items, batch_candidate_set=None):
        """
            batch_prev_items: (B, len)
            batch_candidate_set: (B, 100)
        """
        # print(batch_prev_items.shape, batch_candidate_set.shape, batch_label.shape)
        # print(self.product_fea_emb[batch_prev_items].shape)
        # 对输入序列中的每个商品，获取它的嵌入表示并拼接
        batch_prev_items_emb_list = [self.product_fea_emb[batch_prev_items]]  # 商品特征嵌入
        if not self.pca:
            if self.add_title:
                batch_prev_items_emb_list.append(self.title_linear(self.title_emb(batch_prev_items)))
            if self.add_desc:
                batch_prev_items_emb_list.append(self.desc_linear(self.desc_emb(batch_prev_items)))
        else:
            if self.add_title:
                batch_prev_items_emb_list.append(self.title_emb(batch_prev_items))
            if self.add_desc:
                batch_prev_items_emb_list.append(self.desc_emb(batch_prev_items))
        if self.add_w2v:
            batch_prev_items_emb_list.append(self.w2v_linear(self.w2v_emb(batch_prev_items)))
        if self.add_pemb:
            batch_prev_items_emb_list.append(self.pemb(batch_prev_items))
        batch_prev_items_emb = torch.cat(batch_prev_items_emb_list, dim=-1)
        # 对候选集中的每个商品，获取它的嵌入表示并拼接
        if batch_candidate_set is not None:
            batch_candidate_set_emb_list = [self.product_fea_emb[batch_candidate_set]]  # 商品特征嵌入
            if not self.pca:
                if self.add_title:
                    batch_candidate_set_emb_list.append(self.title_linear(self.title_emb(batch_candidate_set)))
                if self.add_desc:
                    batch_candidate_set_emb_list.append(self.desc_linear(self.desc_emb(batch_candidate_set)))
            else:
                if self.add_title:
                    batch_candidate_set_emb_list.append(self.title_emb(batch_candidate_set))
                if self.add_desc:
                    batch_candidate_set_emb_list.append(self.desc_emb(batch_candidate_set))
            if self.add_w2v:
                batch_candidate_set_emb_list.append(self.w2v_linear(self.w2v_emb(batch_candidate_set)))
            if self.add_pemb:
                batch_candidate_set_emb_list.append(self.pemb(batch_candidate_set))
            batch_candidate_set_emb = torch.cat(batch_candidate_set_emb_list, dim=-1)
        else:
            batch_candidate_set_emb = None
        return batch_prev_items_emb, batch_candidate_set_emb


class ProductV2(nn.Module):
    # 简化产品特征，去掉len_title len_desc，以及连续特征，只保留类别特征，dense-bins=100
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
        self.div = config['div']

        self.locale_emb = nn.Embedding(self.len_locale, self.emb_dim // self.div)
        
        self.encode_brand_emb = nn.Embedding(self.len_encode_brand, self.emb_dim // self.div)
        self.encode_color_emb = nn.Embedding(self.len_encode_color, self.emb_dim // self.div)
        self.encode_size_emb = nn.Embedding(self.len_encode_size, self.emb_dim // self.div)
        self.encode_model_emb = nn.Embedding(self.len_encode_model, self.emb_dim // self.div)
        self.encode_material_emb = nn.Embedding(self.len_encode_material, self.emb_dim // self.div)
        self.encode_author_emb = nn.Embedding(self.len_encode_author, self.emb_dim // self.div)

        self.encode_price_emb = nn.Embedding(self.dense_bins, self.emb_dim)

    def forward(self, batch_products):
        """
            batch_products: dict 
        """
        locale_emb = self.locale_emb(batch_products['locale'])
        encode_brand_emb = self.encode_brand_emb(batch_products['encode_brand'])
        encode_color_emb = self.encode_color_emb(batch_products['encode_color'])
        encode_size_emb = self.encode_size_emb(batch_products['encode_size'])
        encode_model_emb = self.encode_model_emb(batch_products['encode_model'])
        encode_material_emb = self.encode_material_emb(batch_products['encode_material'])
        encode_author_emb = self.encode_author_emb(batch_products['encode_author'])

        encode_price_emb = self.encode_price_emb(batch_products['encode_price'])

        # 将所有特征的表征按照一定的方式组合起来得到这个产品的向量表示
        products_vec = torch.cat([locale_emb, 
                                  encode_brand_emb, encode_color_emb, encode_size_emb, 
                                  encode_model_emb, encode_material_emb, encode_author_emb, 
                                  encode_price_emb], dim=1)
        return products_vec
    

class ProductEmbeddingV2(nn.Module):
    # 区别就是调用ProductV2
    def __init__(self, config, data_feature, products_input, word2vec_embedding, titles_embedding, descs_embedding):
        super().__init__()
        self.emb_dim = config['emb_dim']
        self.device = config['device']
        self.add_title = config['add_title']
        self.add_desc = config['add_desc']
        self.add_w2v = config['add_w2v']
        self.add_pemb = config['add_pemb']
        self.product_fea = ProductV2(config, data_feature).to(self.device)
        self.id_count = data_feature['id_count']
        self.w2v_vector_size = data_feature['w2v_vector_size']
        self.sentence_vector_size = data_feature['sentence_vector_size']
        self.pro_emb_factor = config['pro_emb_factor']
        self.grad_title = config['grad_title']
        self.grad_desc = config['grad_desc']
        self.grad_w2v = config['grad_w2v']
        self.grad_pemb = config['grad_pemb']
        self.pca = config['pca']
        self.pca_dim = config['pca_dim']

        product_fea_emb = self.product_fea(products_input)  # (id_count, 208)
        self.padding_emb = torch.zeros((1, product_fea_emb.shape[1]), requires_grad=False).to(self.device)
        self.product_fea_emb = torch.cat([product_fea_emb, self.padding_emb], dim=0)  # (id_count + 1, 208)
        if not self.pca:
            if self.add_title:
                self.title_emb = nn.Embedding(self.id_count + 1, embedding_dim=self.sentence_vector_size, padding_idx=self.id_count)
                self.title_linear = nn.Linear(self.sentence_vector_size, self.emb_dim * self.pro_emb_factor)
                self.title_emb.weight.data[:self.id_count].copy_(torch.tensor(titles_embedding))
                self.title_emb.weight.requires_grad = self.grad_title
            if self.add_desc:
                self.desc_emb = nn.Embedding(self.id_count + 1, embedding_dim=self.sentence_vector_size, padding_idx=self.id_count)
                self.desc_linear = nn.Linear(self.sentence_vector_size, self.emb_dim * self.pro_emb_factor)
                self.desc_emb.weight.data[:self.id_count].copy_(torch.tensor(descs_embedding))
                self.desc_emb.weight.requires_grad = self.grad_desc 
        else:
            if self.add_title:
                self.title_emb = nn.Embedding(self.id_count + 1, embedding_dim=self.pca_dim, padding_idx=self.id_count)
                self.title_emb.weight.data[:self.id_count].copy_(torch.tensor(titles_embedding))
                # self.title_emb.weight.requires_grad = self.grad_title
            if self.add_desc:
                self.desc_emb = nn.Embedding(self.id_count + 1, embedding_dim=self.pca_dim, padding_idx=self.id_count)
                self.desc_emb.weight.data[:self.id_count].copy_(torch.tensor(descs_embedding))
                # self.desc_emb.weight.requires_grad = self.grad_desc 
        
        if self.add_w2v:
            self.w2v_emb = nn.Embedding(self.id_count + 1, embedding_dim=self.w2v_vector_size, padding_idx=self.id_count)
            self.w2v_linear = nn.Linear(self.w2v_vector_size, self.emb_dim * self.pro_emb_factor)
            self.w2v_emb.weight.data[:self.id_count].copy_(torch.tensor(word2vec_embedding))
            self.w2v_emb.weight.requires_grad = self.grad_w2v
        if self.add_pemb:
            self.pemb = nn.Embedding(self.id_count + 1, embedding_dim=self.emb_dim * self.pro_emb_factor, padding_idx=self.id_count)
            self.pemb.weight.requires_grad = self.grad_pemb

    def forward(self, batch_prev_items, batch_candidate_set=None):
        """
            batch_prev_items: (B, len)
            batch_candidate_set: (B, 100)
        """
        # print(batch_prev_items.shape, batch_candidate_set.shape, batch_label.shape)
        # print(self.product_fea_emb[batch_prev_items].shape)
        # 对输入序列中的每个商品，获取它的嵌入表示并拼接
        batch_prev_items_emb_list = [self.product_fea_emb[batch_prev_items]]  # 商品特征嵌入
        if not self.pca:
            if self.add_title:
                batch_prev_items_emb_list.append(self.title_linear(self.title_emb(batch_prev_items)))
            if self.add_desc:
                batch_prev_items_emb_list.append(self.desc_linear(self.desc_emb(batch_prev_items)))
        else:
            if self.add_title:
                batch_prev_items_emb_list.append(self.title_emb(batch_prev_items))
            if self.add_desc:
                batch_prev_items_emb_list.append(self.desc_emb(batch_prev_items))
        if self.add_w2v:
            batch_prev_items_emb_list.append(self.w2v_linear(self.w2v_emb(batch_prev_items)))
        if self.add_pemb:
            batch_prev_items_emb_list.append(self.pemb(batch_prev_items))
        batch_prev_items_emb = torch.cat(batch_prev_items_emb_list, dim=-1)
        # 对候选集中的每个商品，获取它的嵌入表示并拼接
        if batch_candidate_set is not None:
            batch_candidate_set_emb_list = [self.product_fea_emb[batch_candidate_set]]  # 商品特征嵌入
            if not self.pca:
                if self.add_title:
                    batch_candidate_set_emb_list.append(self.title_linear(self.title_emb(batch_candidate_set)))
                if self.add_desc:
                    batch_candidate_set_emb_list.append(self.desc_linear(self.desc_emb(batch_candidate_set)))
            else:
                if self.add_title:
                    batch_candidate_set_emb_list.append(self.title_emb(batch_candidate_set))
                if self.add_desc:
                    batch_candidate_set_emb_list.append(self.desc_emb(batch_candidate_set))
            if self.add_w2v:
                batch_candidate_set_emb_list.append(self.w2v_linear(self.w2v_emb(batch_candidate_set)))
            if self.add_pemb:
                batch_candidate_set_emb_list.append(self.pemb(batch_candidate_set))
            batch_candidate_set_emb = torch.cat(batch_candidate_set_emb_list, dim=-1)
        else:
            batch_candidate_set_emb = None
        return batch_prev_items_emb, batch_candidate_set_emb


class ProductV3(nn.Module):
    # 简化产品特征，类别特征重新编码，nan变成0对应padding
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
        self.div = config['div']

        self.locale_emb = nn.Embedding(self.len_locale, self.emb_dim // self.div)
        
        self.encode_brand_emb = nn.Embedding(self.len_encode_brand, self.emb_dim // self.div, padding_idx=0)
        self.encode_color_emb = nn.Embedding(self.len_encode_color, self.emb_dim // self.div, padding_idx=0)
        self.encode_size_emb = nn.Embedding(self.len_encode_size, self.emb_dim // self.div, padding_idx=0)
        self.encode_model_emb = nn.Embedding(self.len_encode_model, self.emb_dim // self.div, padding_idx=0)
        self.encode_material_emb = nn.Embedding(self.len_encode_material, self.emb_dim // self.div, padding_idx=0)
        self.encode_author_emb = nn.Embedding(self.len_encode_author, self.emb_dim // self.div, padding_idx=0)

        self.encode_price_emb = nn.Embedding(self.dense_bins, self.emb_dim)

    def forward(self, batch_products):
        """
            batch_products: dict 
        """
        locale_emb = self.locale_emb(batch_products['locale'])
        encode_brand_emb = self.encode_brand_emb(batch_products['encode_brand'])
        encode_color_emb = self.encode_color_emb(batch_products['encode_color'])
        encode_size_emb = self.encode_size_emb(batch_products['encode_size'])
        encode_model_emb = self.encode_model_emb(batch_products['encode_model'])
        encode_material_emb = self.encode_material_emb(batch_products['encode_material'])
        encode_author_emb = self.encode_author_emb(batch_products['encode_author'])

        encode_price_emb = self.encode_price_emb(batch_products['encode_price'])

        # 将所有特征的表征按照一定的方式组合起来得到这个产品的向量表示
        products_vec = torch.cat([locale_emb, 
                                  encode_brand_emb, encode_color_emb, encode_size_emb, 
                                  encode_model_emb, encode_material_emb, encode_author_emb, 
                                  encode_price_emb], dim=1)
        return products_vec
    

class ProductEmbeddingV3(nn.Module):
    # 区别就是调用ProductV3
    def __init__(self, config, data_feature, products_input, word2vec_embedding, titles_embedding, descs_embedding):
        super().__init__()
        self.emb_dim = config['emb_dim']
        self.device = config['device']
        self.add_title = config['add_title']
        self.add_desc = config['add_desc']
        self.add_w2v = config['add_w2v']
        self.add_pemb = config['add_pemb']
        self.product_fea = ProductV3(config, data_feature).to(self.device)
        self.id_count = data_feature['id_count']
        self.w2v_vector_size = data_feature['w2v_vector_size']
        self.sentence_vector_size = data_feature['sentence_vector_size']
        self.pro_emb_factor = config['pro_emb_factor']
        self.grad_title = config['grad_title']
        self.grad_desc = config['grad_desc']
        self.grad_w2v = config['grad_w2v']
        self.grad_pemb = config['grad_pemb']
        self.pca = config['pca']
        self.pca_dim = config['pca_dim']
        
        product_fea_emb = self.product_fea(products_input)  # (id_count, 208)
        self.padding_emb = torch.zeros((1, product_fea_emb.shape[1]), requires_grad=False).to(self.device)
        self.product_fea_emb = torch.cat([product_fea_emb, self.padding_emb], dim=0)  # (id_count + 1, 208)
        if not self.pca:
            if self.add_title:
                self.title_emb = nn.Embedding(self.id_count + 1, embedding_dim=self.sentence_vector_size, padding_idx=self.id_count)
                self.title_linear = nn.Linear(self.sentence_vector_size, self.emb_dim * self.pro_emb_factor)
                self.title_emb.weight.data[:self.id_count].copy_(torch.tensor(titles_embedding))
                self.title_emb.weight.requires_grad = self.grad_title
            if self.add_desc:
                self.desc_emb = nn.Embedding(self.id_count + 1, embedding_dim=self.sentence_vector_size, padding_idx=self.id_count)
                self.desc_linear = nn.Linear(self.sentence_vector_size, self.emb_dim * self.pro_emb_factor)
                self.desc_emb.weight.data[:self.id_count].copy_(torch.tensor(descs_embedding))
                self.desc_emb.weight.requires_grad = self.grad_desc 
        else:
            if self.add_title:
                self.title_emb = nn.Embedding(self.id_count + 1, embedding_dim=self.pca_dim, padding_idx=self.id_count)
                self.title_emb.weight.data[:self.id_count].copy_(torch.tensor(titles_embedding))
                # self.title_emb.weight.requires_grad = self.grad_title
            if self.add_desc:
                self.desc_emb = nn.Embedding(self.id_count + 1, embedding_dim=self.pca_dim, padding_idx=self.id_count)
                self.desc_emb.weight.data[:self.id_count].copy_(torch.tensor(descs_embedding))
                # self.desc_emb.weight.requires_grad = self.grad_desc 
        
        if self.add_w2v:
            self.w2v_emb = nn.Embedding(self.id_count + 1, embedding_dim=self.w2v_vector_size, padding_idx=self.id_count)
            self.w2v_linear = nn.Linear(self.w2v_vector_size, self.emb_dim * self.pro_emb_factor)
            self.w2v_emb.weight.data[:self.id_count].copy_(torch.tensor(word2vec_embedding))
            self.w2v_emb.weight.requires_grad = self.grad_w2v
        if self.add_pemb:
            self.pemb = nn.Embedding(self.id_count + 1, embedding_dim=self.emb_dim * self.pro_emb_factor, padding_idx=self.id_count)
            self.pemb.weight.requires_grad = self.grad_pemb

    def forward(self, batch_prev_items, batch_candidate_set=None):
        """
            batch_prev_items: (B, len)
            batch_candidate_set: (B, 100)
        """
        # print(batch_prev_items.shape, batch_candidate_set.shape, batch_label.shape)
        # print(self.product_fea_emb[batch_prev_items].shape)
        # 对输入序列中的每个商品，获取它的嵌入表示并拼接
        batch_prev_items_emb_list = [self.product_fea_emb[batch_prev_items]]  # 商品特征嵌入
        if not self.pca:
            if self.add_title:
                batch_prev_items_emb_list.append(self.title_linear(self.title_emb(batch_prev_items)))
            if self.add_desc:
                batch_prev_items_emb_list.append(self.desc_linear(self.desc_emb(batch_prev_items)))
        else:
            if self.add_title:
                batch_prev_items_emb_list.append(self.title_emb(batch_prev_items))
            if self.add_desc:
                batch_prev_items_emb_list.append(self.desc_emb(batch_prev_items))
        if self.add_w2v:
            batch_prev_items_emb_list.append(self.w2v_linear(self.w2v_emb(batch_prev_items)))
        if self.add_pemb:
            batch_prev_items_emb_list.append(self.pemb(batch_prev_items))
        batch_prev_items_emb = torch.cat(batch_prev_items_emb_list, dim=-1)
        # 对候选集中的每个商品，获取它的嵌入表示并拼接
        if batch_candidate_set is not None:
            batch_candidate_set_emb_list = [self.product_fea_emb[batch_candidate_set]]  # 商品特征嵌入
            if not self.pca:
                if self.add_title:
                    batch_candidate_set_emb_list.append(self.title_linear(self.title_emb(batch_candidate_set)))
                if self.add_desc:
                    batch_candidate_set_emb_list.append(self.desc_linear(self.desc_emb(batch_candidate_set)))
            else:
                if self.add_title:
                    batch_candidate_set_emb_list.append(self.title_emb(batch_candidate_set))
                if self.add_desc:
                    batch_candidate_set_emb_list.append(self.desc_emb(batch_candidate_set))
            if self.add_w2v:
                batch_candidate_set_emb_list.append(self.w2v_linear(self.w2v_emb(batch_candidate_set)))
            if self.add_pemb:
                batch_candidate_set_emb_list.append(self.pemb(batch_candidate_set))
            batch_candidate_set_emb = torch.cat(batch_candidate_set_emb_list, dim=-1)
        else:
            batch_candidate_set_emb = None
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
    

class IntraAttentionV2(nn.Module):
    """对序列经过 LSTM 后的隐藏层向量序列做 Attention 强化
    key: 当前序列经过 LSTM 后的隐藏层向量序列
    query: 轨迹向量序列的最后一个状态
    """

    def __init__(self, hidden_size):
        super(IntraAttentionV2, self).__init__()
        # 模型参数
        self.hidden_size = hidden_size
        # 模型结构
        self.v_t = nn.Linear(in_features=self.hidden_size, out_features=1, bias=False)
        self.a_1 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.a_2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)

    def forward(self, query, key, mask=None):
        """前馈

        Args:
            query (tensor): shape (batch_size, hidden_size)
            key (tensor): shape (batch_size, seq_len, hidden_size)
            mask (tensor): padding mask, 1 表示非补齐值, 0 表示补齐值 shape (batch_size, seq_len)
        Return:
            attn_hidden (tensor): shape (batch_size, hidden_size)
        """
        key_hidden = self.a_1(key)  # shape: (batch_size, len, hidden_size)
        query_hidden = self.a_2(query).unsqueeze(1)  # shape: (batch_size, 1, hidden_size)
        query_hidden_masked = mask.unsqueeze(2).expand_as(key_hidden) * query_hidden  # (B, len, hidden_size)
        attn_hidden = self.v_t(torch.sigmoid(key_hidden + query_hidden_masked).view(-1, self.hidden_size)).view(mask.size())   # (B, len)
        attn_hidden = torch.sum(attn_hidden.unsqueeze(2).expand_as(key) * key, 1) # (B, hidden_size)
        return attn_hidden
    

class IntraAttentionV3(nn.Module):
    """对序列经过 LSTM 后的隐藏层向量序列做 Attention 强化
    key: 当前序列经过 LSTM 后的隐藏层向量序列
    query: 轨迹向量序列的最后一个状态
    """

    def __init__(self, hidden_size):
        super(IntraAttentionV3, self).__init__()
        # 模型参数
        self.hidden_size = hidden_size
        # 模型结构
        self.w1_linear = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.w2_linear = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.w3_linear = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)

    def forward(self, query, key, mask=None):
        """前馈

        Args:
            query (tensor): shape (batch_size, hidden_size)
            key (tensor): shape (batch_size, seq_len, hidden_size)
            mask (tensor): padding mask, 1 表示非补齐值, 0 表示补齐值 shape (batch_size, seq_len)
        Return:
            attn_hidden (tensor): shape (batch_size, hidden_size)
        """
        query_hidden = self.w1_linear(query)  # shape: (batch_size, hidden_size)
        key_hidden = self.w2_linear(key)  # shape: (batch_size, seq_len, hidden_size)
        value_hidden = self.w3_linear(key)  # shape: (batch_size, seq_len, hidden_size)
        attn_weight = torch.bmm(key_hidden, query_hidden.unsqueeze(2)).squeeze(2) # shape (batch_size, seq_len)
        if mask is not None:
            mask = attn_weight.masked_fill(mask==0, -1e9) # mask 
        attn_weight = torch.softmax(attn_weight, dim=1).unsqueeze(2) # shape (batch_size, seq_len, 1)
        attn_hidden = torch.sum(attn_weight * value_hidden, dim=1)
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
        self.pro_emb_factor = config['pro_emb_factor']

        self.input_size = data_feature['len_features'] * self.emb_dim + data_feature['len_emb_features'] * self.pro_emb_factor * self.emb_dim

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
    

class MatcherV3(nn.Module):
    """候选集与当前轨迹状态之间的注意力模块
    """

    def __init__(self, hidden_size, item_emb_size):
        super(MatcherV3, self).__init__()
        self.w1_linear = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.w2_linear = nn.Linear(in_features=item_emb_size, out_features=hidden_size, bias=False)

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
        query_hidden = self.w1_linear(query).unsqueeze(1)  # shape: (batch_size, 1, hidden_size)
        key_hidden = self.w2_linear(key)  # shape: (batch_size, candidate_size, hidden_size)
        out = torch.bmm(query_hidden, key_hidden.transpose(1, 2))  # (batch_size, 1, candidate_size)
        out = out.squeeze(1)  # shape: (batch_size, candidate_size)
        return out
    

class SeqFeatureEmbedding(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()
        self.emb_dim = config['emb_dim']
        self.hid_dim = config['hid_dim']
        self.len_seqs_num_feas = data_feature['len_seqs_num_feas']
        self.len_seqs_cat_feas = data_feature['len_seqs_cat_feas']
        self.seq_emb_factor = data_feature['seq_emb_factor']
        # 序列中手动统计出来的类别特征编码
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


class SeqFeatureEmbeddingV2(nn.Module):
    # 连续特征进行了分桶，类别特征不再做MLP，保持原状
    def __init__(self, config, data_feature):
        super().__init__()
        self.emb_dim = config['emb_dim']
        self.hid_dim = config['hid_dim']
        self.seq_emb_factor = data_feature['seq_emb_factor']
        self.dense_bins = data_feature['dense_bins']

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

        self.encode_priceMEAN_emb = nn.Embedding(self.dense_bins, self.emb_dim)
        self.encode_priceSTD_emb = nn.Embedding(self.dense_bins, self.emb_dim)
        self.encode_priceMIN_emb = nn.Embedding(self.dense_bins, self.emb_dim)
        self.encode_priceMAX_emb = nn.Embedding(self.dense_bins, self.emb_dim)
        self.encode_priceSUM_emb = nn.Embedding(self.dense_bins, self.emb_dim)

        # self.seq_cat_emb = nn.Sequential(nn.Linear(self.emb_dim * self.len_seqs_cat_feas, self.hid_dim),
        #                                  nn.ReLU(),
        #                                  nn.Linear(self.hid_dim, self.emb_dim * self.seq_emb_factor))

    def forward(self, batch_seq_cat, batch_seq_num):
        """
            batch_seq_cat: (B, len_seqs_cat_feas)
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

        encode_priceMEAN_emb = self.encode_priceMEAN_emb(batch_seq_cat[:, 14])
        encode_priceSTD_emb = self.encode_priceSTD_emb(batch_seq_cat[:, 15])
        encode_priceMIN_emb = self.encode_priceMIN_emb(batch_seq_cat[:, 16])
        encode_priceMAX_emb = self.encode_priceMAX_emb(batch_seq_cat[:, 17])
        encode_priceSUM_emb = self.encode_priceSUM_emb(batch_seq_cat[:, 18])

        sparse_vec = torch.cat([idNUNIQUE_emb, idCOUNT_emb,
                                  brandNUNIQUE_emb, brandCOUNT_emb,
                                  colorNUNIQUE_emb, colorCOUNT_emb, 
                                  sizeNUNIQUE_emb, sizeCOUNT_emb, 
                                  modelNUNIQUE_emb, modelCOUNT_emb,
                                  materialNUNIQUE_emb, materialCOUNT_emb,
                                  authorNUNIQUE_emb, authorCOUNT_emb,
                                  encode_priceMEAN_emb,
                                  encode_priceSTD_emb,
                                  encode_priceMIN_emb,
                                  encode_priceMAX_emb,
                                  encode_priceSUM_emb
                                  ], dim=1)  # (B, 19 * emb_dim)
        return sparse_vec   # (B, 19 * emb_dim)


class SeqFeatureEmbeddingV3(nn.Module):
    # 连续特征进行了分桶，类别特征不再做MLP，保持原状
    # 补充了几个连续特征
    # 补充了序列的众数这一特征！
    def __init__(self, config, data_feature, product_emb):
        super().__init__()
        self.emb_dim = config['emb_dim']
        self.hid_dim = config['hid_dim']
        self.seq_emb_factor = data_feature['seq_emb_factor']
        self.dense_bins = data_feature['dense_bins']
        self.div = config['div']
        self.add_mode = config['add_mode']
        self.product_emb = product_emb

        self.idNUNIQUE = data_feature['idNUNIQUE']
        self.idCOUNT = data_feature['idCOUNT']

        self.brandNUNIQUE = data_feature['encode_brandNUNIQUE']
        self.brandCOUNT = data_feature['encode_brandCOUNT']

        self.colorNUNIQUE = data_feature['encode_colorNUNIQUE']
        self.colorCOUNT = data_feature['encode_colorCOUNT']

        self.sizeNUNIQUE = data_feature['encode_sizeNUNIQUE']
        self.sizeCOUNT = data_feature['encode_sizeCOUNT']

        self.modelNUNIQUE = data_feature['encode_modelNUNIQUE']
        self.modelCOUNT = data_feature['encode_modelCOUNT']

        self.materialNUNIQUE = data_feature['encode_materialNUNIQUE']
        self.materialCOUNT = data_feature['encode_materialCOUNT']

        self.authorNUNIQUE = data_feature['encode_authorNUNIQUE']
        self.authorCOUNT = data_feature['encode_authorCOUNT']

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

        self.encode_priceMEAN_emb = nn.Embedding(self.dense_bins, self.emb_dim)
        self.encode_priceSTD_emb = nn.Embedding(self.dense_bins, self.emb_dim)
        self.encode_priceMIN_emb = nn.Embedding(self.dense_bins, self.emb_dim)
        self.encode_priceMAX_emb = nn.Embedding(self.dense_bins, self.emb_dim)
        self.encode_priceSUM_emb = nn.Embedding(self.dense_bins, self.emb_dim)
        self.encode_pricePTP_emb = nn.Embedding(self.dense_bins, self.emb_dim)
        self.encode_priceQUANTILE75_emb = nn.Embedding(self.dense_bins, self.emb_dim)
        self.encode_priceQUANTILE25_emb = nn.Embedding(self.dense_bins, self.emb_dim)

    def forward(self, batch_seq_cat, batch_seq_num):
        """
            batch_seq_cat: (B, len_seqs_cat_feas)
        """
        idNUNIQUE_emb = self.idNUNIQUE_emb(batch_seq_cat[:, 0])
        idCOUNT_emb = self.idCOUNT_emb(batch_seq_cat[:, 1])
        
        brandNUNIQUE_emb = self.brandNUNIQUE_emb(batch_seq_cat[:, 2])
        brandCOUNT_emb = self.brandCOUNT_emb(batch_seq_cat[:, 3])
        if self.add_mode:
            encode_brand_emb = self.product_emb.encode_brand_emb(batch_seq_cat[:, 4])

        colorNUNIQUE_emb = self.colorNUNIQUE_emb(batch_seq_cat[:, 5])
        colorCOUNT_emb = self.colorCOUNT_emb(batch_seq_cat[:, 6])
        if self.add_mode:
            encode_color_emb = self.product_emb.encode_color_emb(batch_seq_cat[:, 7])

        sizeNUNIQUE_emb = self.sizeNUNIQUE_emb(batch_seq_cat[:, 8])
        sizeCOUNT_emb = self.sizeCOUNT_emb(batch_seq_cat[:, 9])
        if self.add_mode:
            encode_size_emb = self.product_emb.encode_size_emb(batch_seq_cat[:, 10])
        
        modelNUNIQUE_emb = self.modelNUNIQUE_emb(batch_seq_cat[:, 11])
        modelCOUNT_emb = self.modelCOUNT_emb(batch_seq_cat[:, 12])
        if self.add_mode:
            encode_model_emb = self.product_emb.encode_model_emb(batch_seq_cat[:, 13])

        materialNUNIQUE_emb = self.materialNUNIQUE_emb(batch_seq_cat[:, 14])
        materialCOUNT_emb = self.materialCOUNT_emb(batch_seq_cat[:, 15])
        if self.add_mode:
            encode_material_emb = self.product_emb.encode_material_emb(batch_seq_cat[:, 16])

        authorNUNIQUE_emb = self.authorNUNIQUE_emb(batch_seq_cat[:, 17])
        authorCOUNT_emb = self.authorCOUNT_emb(batch_seq_cat[:, 18])
        if self.add_mode:
            encode_author_emb = self.product_emb.encode_author_emb(batch_seq_cat[:, 19])

        encode_priceMEAN_emb = self.encode_priceMEAN_emb(batch_seq_cat[:, 20])
        encode_priceSTD_emb = self.encode_priceSTD_emb(batch_seq_cat[:, 21])
        encode_priceMIN_emb = self.encode_priceMIN_emb(batch_seq_cat[:, 22])
        encode_priceMAX_emb = self.encode_priceMAX_emb(batch_seq_cat[:, 23])
        encode_priceSUM_emb = self.encode_priceSUM_emb(batch_seq_cat[:, 24])
        encode_pricePTP_emb = self.encode_pricePTP_emb(batch_seq_cat[:, 25])
        encode_priceQUANTILE75_emb = self.encode_priceQUANTILE75_emb(batch_seq_cat[:, 26])
        encode_priceQUANTILE25_emb = self.encode_priceQUANTILE25_emb(batch_seq_cat[:, 27])

        cat_list = [idNUNIQUE_emb, idCOUNT_emb,
                    brandNUNIQUE_emb, brandCOUNT_emb, 
                    colorNUNIQUE_emb, colorCOUNT_emb, 
                    sizeNUNIQUE_emb, sizeCOUNT_emb, 
                    modelNUNIQUE_emb, modelCOUNT_emb, 
                    materialNUNIQUE_emb, materialCOUNT_emb, 
                    authorNUNIQUE_emb, authorCOUNT_emb, 
                    encode_priceMEAN_emb,
                    encode_priceSTD_emb,
                    encode_priceMIN_emb,
                    encode_priceMAX_emb,
                    encode_priceSUM_emb,
                    encode_pricePTP_emb,
                    encode_priceQUANTILE75_emb,
                    encode_priceQUANTILE25_emb,
                    ]
        if self.add_mode:
            cat_list += [encode_brand_emb, encode_color_emb, encode_size_emb, encode_model_emb, encode_material_emb, encode_author_emb]
        sparse_vec = torch.cat(cat_list, dim=1)  # (B, 28 * emb_dim)
        return sparse_vec   # (B, 28 * emb_dim)
    

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
        self.intra = config['intra']
        self.feature = config['feature']
        self.div = config['div']
        self.pro_emb_factor = config['pro_emb_factor']

        if self.feature.lower() == 'v1':
            self.product_emb = ProductEmbedding(config, data_feature, products_input, 
                                                word2vec_embedding, titles_embedding, descs_embedding)
            self.input_size = data_feature['len_features'] * self.emb_dim // self.div + data_feature['len_emb_features'] * self.pro_emb_factor * self.emb_dim
        elif self.feature.lower() == 'v2':
            self.product_emb = ProductEmbeddingV2(config, data_feature, products_input, 
                                                  word2vec_embedding, titles_embedding, descs_embedding)
            self.input_size = 7 * self.emb_dim // self.div + self.emb_dim + data_feature['len_emb_features'] * self.pro_emb_factor * self.emb_dim
        elif self.feature.lower() == 'v3':
            self.product_emb = ProductEmbeddingV3(config, data_feature, products_input, 
                                                  word2vec_embedding, titles_embedding, descs_embedding)
            self.input_size = 7 * self.emb_dim // self.div + self.emb_dim + data_feature['len_emb_features'] * self.pro_emb_factor * self.emb_dim
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.layers, batch_first=True, bidirectional=self.bidirectional)
        if self.intra == 'intra':
            self.intra_attn = IntraAttention(hidden_size=self.hidden_size * self.num_direction)
        elif self.intra == 'intra2':
            self.intra_attn = IntraAttentionV2(hidden_size=self.hidden_size * self.num_direction)
        elif self.intra == 'intra3':
            self.intra_attn = IntraAttentionV3(hidden_size=self.hidden_size * self.num_direction)
        self.dropout = nn.Dropout(p=self.dropout_p)
        output_dim = 2 * self.hidden_size * self.num_direction
        
        if self.attn_match:
            self.output = MatcherV2(hidden_size=output_dim, item_emb_size=self.input_size, dropout=self.dropout_p)
        else:
            self.output = MatcherV3(hidden_size=output_dim, item_emb_size=self.input_size)
            # self.output = Matcher(hidden_size=output_dim, item_emb_size=self.input_size)

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
        attn_hidden = torch.cat([attn_hidden, lstm_last_hidden], 1) # (B, 2 * hidden_size)
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
        self.seq_model = config['seq_model']
        self.feature = config['feature']
        self.div = config['div']
        self.add_mode = config['add_mode']
        if self.bidirectional:
            self.num_direction = 2
        else:
            self.num_direction = 1
        self.id_count = data_feature['id_count']
        self.intra = config['intra']
        self.pro_emb_factor = config['pro_emb_factor']

        if self.feature.lower() == 'v1':
            self.product_emb = ProductEmbedding(config, data_feature, products_input, 
                                                word2vec_embedding, titles_embedding, descs_embedding)
            self.seq_fea_emb = SeqFeatureEmbedding(config, data_feature)
            seq_fea_dim = data_feature['seq_emb_factor'] * self.emb_dim
            self.input_size = data_feature['len_features'] * self.emb_dim // self.div + data_feature['len_emb_features'] * self.pro_emb_factor * self.emb_dim
        elif self.feature.lower() == 'v2':
            self.product_emb = ProductEmbeddingV2(config, data_feature, products_input, 
                                                  word2vec_embedding, titles_embedding, descs_embedding)
            self.seq_fea_emb = SeqFeatureEmbeddingV2(config, data_feature)
            seq_fea_dim = data_feature['len_seqs_cat_feas'] * self.emb_dim
            self.input_size = 7 * self.emb_dim // self.div + self.emb_dim + data_feature['len_emb_features'] * self.pro_emb_factor * self.emb_dim
        elif self.feature.lower() == 'v3':
            self.product_emb = ProductEmbeddingV3(config, data_feature, products_input, 
                                                  word2vec_embedding, titles_embedding, descs_embedding)
            self.seq_fea_emb = SeqFeatureEmbeddingV3(config, data_feature, product_emb=self.product_emb.product_fea)
            if self.add_mode:
                seq_fea_dim = data_feature['len_seqs_cat_feas'] * self.emb_dim
            else:
                seq_fea_dim = (data_feature['len_seqs_cat_feas'] - 6) * self.emb_dim
            self.input_size = 7 * self.emb_dim // self.div + self.emb_dim + data_feature['len_emb_features'] * self.pro_emb_factor * self.emb_dim

        if self.seq_model.lower() == 'lstm':
            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, 
                                num_layers=self.layers, batch_first=True, bidirectional=self.bidirectional)
            if self.intra == 'intra':
                self.intra_attn = IntraAttention(hidden_size=self.hidden_size * self.num_direction)
            elif self.intra == 'intra2':
                self.intra_attn = IntraAttentionV2(hidden_size=self.hidden_size * self.num_direction)
            elif self.intra == 'intra3':
                self.intra_attn = IntraAttentionV3(hidden_size=self.hidden_size * self.num_direction)
            output_dim = 2 * self.hidden_size * self.num_direction + seq_fea_dim
        elif self.seq_model.lower() == 'transformer':
            self.start_linear = nn.Linear(self.input_size, self.hidden_size)
            self.transformer = Transformer(config=config, data_feature=data_feature)
            output_dim = self.hidden_size + seq_fea_dim
        self.dropout = nn.Dropout(p=self.dropout_p)

        if self.attn_match:
            self.output = MatcherV2(hidden_size=output_dim, item_emb_size=self.input_size, dropout=self.dropout_p)
        else:
            self.output = MatcherV3(hidden_size=output_dim, item_emb_size=self.input_size)
            # self.output = Matcher(hidden_size=output_dim, item_emb_size=self.input_size)

        self.loss_func = nn.CrossEntropyLoss(ignore_index=data_feature['len_candidate_set']) # 候选集大小100

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
        
        if self.seq_model.lower() == 'lstm':
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
            attn_hidden = torch.cat([attn_hidden, lstm_last_hidden], 1) # (B, 2 * hidden_size)
        elif self.seq_model.lower() == 'transformer':
            input_emb = self.start_linear(input_emb)  # (B, len, hidden_size)
            attention_out = self.transformer(input_emb, batch_mask)  # (B, len, hidden_size)
            input_mask_expanded = batch_mask.unsqueeze(-1).expand(attention_out.size()).float()  # (B, len, hidden_size)
            sum_embeddings = torch.sum(attention_out * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            attn_hidden = sum_embeddings / sum_mask  # (B, hidden_size)

        attn_hidden = self.dropout(attn_hidden) # (B, 2 * hidden_size)

        # 人工序列特征
        seq_fea = self.seq_fea_emb(batch_seq_cat, batch_seq_num)  # (B, seq_fea_dim)
        query = torch.cat([attn_hidden, seq_fea], dim=1)  # (B, 2 * hidden_size + seq_fea_dim)
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


def drop_path_func(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_func(x, self.drop_prob, self.training)
    

class MultiHeadedAttention(nn.Module):

    def __init__(self, num_heads, d_model, dim_out, attn_drop=0., proj_drop=0., device=torch.device('cpu')):
        super().__init__()
        assert d_model % num_heads == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.device = device
        self.scale = self.d_k ** -0.5  # 1/sqrt(dk)

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.dropout = nn.Dropout(p=attn_drop)

        self.proj = nn.Linear(d_model, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, padding_masks, future_mask=False, output_attentions=False):
        """

        Args:
            x: (B, T, d_model)
            padding_masks: (B, 1, T, T) padding_mask
            future_mask: True/False
            batch_temporal_mat: (B, T, T)

        Returns:

        """
        batch_size, seq_len, d_model = x.shape

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # l(x) --> (B, T, d_model)
        # l(x).view() --> (B, T, head, d_k)
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (x, x, x))]
        # q, k, v --> (B, head, T, d_k)

        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # (B, head, T, T)

        if padding_masks is not None:
            scores.masked_fill_(padding_masks == 0, float('-inf'))

        if future_mask:  # 单向，每个位置只跟前边的位置做注意力
            mask_postion = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).bool().to(self.device)
            scores.masked_fill_(mask_postion, float('-inf'))

        p_attn = F.softmax(scores, dim=-1)  # (B, head, T, T)
        p_attn = self.dropout(p_attn)
        out = torch.matmul(p_attn, value)  # (B, head, T, d_k)

        # 3) "Concat" using a view and apply a final linear.
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)  # (B, T, d_model)
        out = self.proj(out)  # (B, T, N, D)
        out = self.proj_drop(out)
        if output_attentions:
            return out, p_attn  # (B, T, dim_out), (B, head, T, T)
        else:
            return out, None  # (B, T, dim_out)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        """Position-wise Feed-Forward Networks
        Args:
            in_features:
            hidden_features:
            out_features:
            act_layer:
            drop:
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, d_model, attn_heads, feed_forward_hidden, drop_path,
                 attn_drop, dropout, type_ln='post', device=torch.device('cpu')):
        """

        Args:
            d_model: hidden size of transformer
            attn_heads: head sizes of multi-head attention
            feed_forward_hidden: feed_forward_hidden, usually 4*d_model
            drop_path: encoder dropout rate
            attn_drop: attn dropout rate
            dropout: dropout rate
            type_ln:
        """

        super().__init__()
        self.attention = MultiHeadedAttention(num_heads=attn_heads, d_model=d_model, dim_out=d_model,
                                              attn_drop=attn_drop, proj_drop=dropout, device=device)
        self.mlp = Mlp(in_features=d_model, hidden_features=feed_forward_hidden,
                       out_features=d_model, act_layer=nn.GELU, drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.type_ln = type_ln

    def forward(self, x, padding_masks, future_mask=False, output_attentions=False):
        """

        Args:
            x: (B, T, d_model)
            padding_masks: (B, 1, T, T)
            future_mask: True/False

        Returns:
            (B, T, d_model)

        """
        if self.type_ln == 'pre':
            attn_out, attn_score = self.attention(self.norm1(x), padding_masks=padding_masks,
                                                  future_mask=future_mask, output_attentions=output_attentions)
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            attn_out, attn_score = self.attention(x, padding_masks=padding_masks,
                                                  future_mask=future_mask, output_attentions=output_attentions)
            x = self.norm1(x + self.drop_path(attn_out))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            raise ValueError('Error type_ln {}'.format(self.type_ln))
        return x
    

class Transformer(nn.Module):
    def __init__(self, config, data_feature):
        super().__init__()
        self.config = config

        self.d_model = self.config['hid_dim']
        self.n_layers = config['layers']
        self.attn_heads = self.config['heads']
        self.mlp_ratio = self.config.get('mlp_ratio', 4)
        self.dropout = self.config['dropout']
        self.drop_path = self.config.get('drop_path', 0.3)
        self.attn_drop = self.config['dropout']
        self.type_ln = self.config.get('type_ln', 'post')
        self.future_mask = self.config.get('future_mask', False)        
        self.device = self.config['device']
        self.feed_forward_hidden = self.d_model * self.mlp_ratio

        enc_dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.n_layers)]  # stochastic depth decay rule
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model=self.d_model, attn_heads=self.attn_heads,
                              feed_forward_hidden=self.feed_forward_hidden, drop_path=enc_dpr[i],
                              attn_drop=self.attn_drop, dropout=self.dropout,
                              type_ln=self.type_ln, device=self.device) for i in range(self.n_layers)])
        
    
    def forward(self, x, padding_masks):
        """

        Args:
            x: (B, T, d_model)
            padding_masks: (B, T), 1 mask 0 padding
            future_mask: True/False

        Returns:
            (B, T, d_model)

        """
        padding_masks_input = padding_masks.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  # (B, 1, T, T)
        embedding_output = x
        for transformer in self.transformer_blocks:
            embedding_output = transformer.forward(
                x=embedding_output, padding_masks=padding_masks_input,
                future_mask=self.future_mask, output_attentions=False)  # (B, T, d_model)
        return embedding_output  # (B, T, d_model)


class BinaryModel(nn.Module):

    def __init__(self, config, data_feature, products_input, word2vec_embedding, titles_embedding, descs_embedding):
        super(BinaryModel, self).__init__()

        self.hidden_size = config['hid_dim']
        self.emb_dim = config['emb_dim']
        self.device = config['device']
        self.layers = config['layers']
        self.dropout_p = config['dropout']
        self.emb_dim = config['emb_dim']
        self.attn_match = config['attn_match']
        self.bidirectional  = config['bidirectional']
        self.seq_model = config['seq_model']
        self.feature = config['feature']
        self.div = config['div']
        self.add_mode = config['add_mode']
        if self.bidirectional:
            self.num_direction = 2
        else:
            self.num_direction = 1
        self.id_count = data_feature['id_count']
        self.intra = config['intra']
        self.pro_emb_factor = config['pro_emb_factor']

        if self.feature.lower() == 'v1':
            self.product_emb = ProductEmbedding(config, data_feature, products_input, 
                                                word2vec_embedding, titles_embedding, descs_embedding)
            self.seq_fea_emb = SeqFeatureEmbedding(config, data_feature)
            seq_fea_dim = data_feature['seq_emb_factor'] * self.emb_dim
            self.input_size = data_feature['len_features'] * self.emb_dim // self.div + data_feature['len_emb_features'] * self.pro_emb_factor * self.emb_dim
        elif self.feature.lower() == 'v2':
            self.product_emb = ProductEmbeddingV2(config, data_feature, products_input, 
                                                  word2vec_embedding, titles_embedding, descs_embedding)
            self.seq_fea_emb = SeqFeatureEmbeddingV2(config, data_feature)
            seq_fea_dim = data_feature['len_seqs_cat_feas'] * self.emb_dim
            self.input_size = 7 * self.emb_dim // self.div + self.emb_dim + data_feature['len_emb_features'] * self.pro_emb_factor * self.emb_dim
        elif self.feature.lower() == 'v3':
            self.product_emb = ProductEmbeddingV3(config, data_feature, products_input, 
                                                  word2vec_embedding, titles_embedding, descs_embedding)
            self.seq_fea_emb = SeqFeatureEmbeddingV3(config, data_feature, product_emb=self.product_emb.product_fea)
            if self.add_mode:
                seq_fea_dim = data_feature['len_seqs_cat_feas'] * self.emb_dim
            else:
                seq_fea_dim = (data_feature['len_seqs_cat_feas'] - 6) * self.emb_dim
            self.input_size = 7 * self.emb_dim // self.div + self.emb_dim + data_feature['len_emb_features'] * self.pro_emb_factor * self.emb_dim

        if self.seq_model.lower() == 'lstm':
            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, 
                                num_layers=self.layers, batch_first=True, bidirectional=self.bidirectional)
            if self.intra == 'intra':
                self.intra_attn = IntraAttention(hidden_size=self.hidden_size * self.num_direction)
            elif self.intra == 'intra2':
                self.intra_attn = IntraAttentionV2(hidden_size=self.hidden_size * self.num_direction)
            elif self.intra == 'intra3':
                self.intra_attn = IntraAttentionV3(hidden_size=self.hidden_size * self.num_direction)
            output_dim = 2 * self.hidden_size * self.num_direction + seq_fea_dim + self.input_size
        elif self.seq_model.lower() == 'transformer':
            self.start_linear = nn.Linear(self.input_size, self.hidden_size)
            self.transformer = Transformer(config=config, data_feature=data_feature)
            output_dim = self.hidden_size + seq_fea_dim + self.input_size
        self.dropout = nn.Dropout(p=self.dropout_p)

        self.output_layer = nn.Linear(output_dim, 1, bias=False)

    def forward(self, batch_prev_items, batch_locale, batch_candidate, batch_len, batch_label, batch_seq_cat, batch_seq_num, batch_mask=None, train=True):
        """
        Args:
            batch_prev_items: (B, len)
            batch_locale: (B,)
            batch_candidate: (B,)
            batch_len: (B,)
            batch_label: (B, 1)
            batch_mask: (B, len)
            batch_seq_cat: (B, len_seqs_cat_feas)
            batch_seq_num: (B, len_seqs_num_feas)
        
        Return:
            candidate_prob (tensor): 对候选集下一跳的概率预测 (batch_size, candidate_size)
        """
        batch_prev_items_emb, batch_candidate_emb = self.product_emb(batch_prev_items, batch_candidate)
        # batch_prev_items_emb (B, len, input_size)
        # batch_candidate_emb (B, input_size) or None

        input_emb = self.dropout(batch_prev_items_emb)
        
        if self.seq_model.lower() == 'lstm':
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
            attn_hidden = torch.cat([attn_hidden, lstm_last_hidden], 1) # (B, 2 * hidden_size)
        elif self.seq_model.lower() == 'transformer':
            input_emb = self.start_linear(input_emb)  # (B, len, hidden_size)
            attention_out = self.transformer(input_emb, batch_mask)  # (B, len, hidden_size)
            input_mask_expanded = batch_mask.unsqueeze(-1).expand(attention_out.size()).float()  # (B, len, hidden_size)
            sum_embeddings = torch.sum(attention_out * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            attn_hidden = sum_embeddings / sum_mask  # (B, hidden_size)
        
        attn_hidden = self.dropout(attn_hidden) # (B, hidden_size)
        # 人工序列特征
        seq_fea = self.seq_fea_emb(batch_seq_cat, batch_seq_num)  # (B, seq_fea_dim)
        query = torch.cat([attn_hidden, seq_fea], dim=1)  # (B, hidden_size + seq_fea_dim)
        
        output = torch.cat([query, batch_candidate_emb], dim=1)  # (B, hidden_size + seq_fea_dim + input_size)
        output_logit = self.output_layer(output)  # (B, 1)
        y_pred = torch.sigmoid(output_logit)  # (B, 1)
        return y_pred

    def predict(self, batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_seq_cat, batch_seq_num, batch_mask=None):
        """预测
        Return:
            y_pred (tensor): softmax 后对候选集下一跳的概率预测 (batch_size, 1)
        """
        y_pred = self.forward(batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_seq_cat, batch_seq_num, batch_mask, False)
        loss = F.binary_cross_entropy(y_pred, batch_label)
        return y_pred, loss



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


class GNN(nn.Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(nn.Module):

    def __init__(self, config, data_feature, products_input, word2vec_embedding, titles_embedding, descs_embedding):
        super(SessionGraph, self).__init__()

        self.hidden_size = config['hid_dim']
        self.emb_dim = config['emb_dim']
        self.device = config['device']
        # self.layers = config['layers']
        # self.dropout_p = config['dropout']
        self.emb_dim = config['emb_dim']
        self.feature = config['feature']
        self.div = config['div']
        self.add_mode = config['add_mode']
        self.batch_size = config['batch_size']
        self.n_node = config['id_count']
        self.nonhybrid = True
        self.step = 2
        self.id_count = data_feature['id_count']
        self.pro_emb_factor = config['pro_emb_factor']
        
        self.gnn = GNN(self.hidden_size, step=self.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()

        if self.feature.lower() == 'v1':
            self.product_emb = ProductEmbedding(config, data_feature, products_input, 
                                                word2vec_embedding, titles_embedding, descs_embedding)
            self.input_size = data_feature['len_features'] * self.emb_dim // self.div + data_feature['len_emb_features'] * self.pro_emb_factor * self.emb_dim
        elif self.feature.lower() == 'v2':
            self.product_emb = ProductEmbeddingV2(config, data_feature, products_input, 
                                                  word2vec_embedding, titles_embedding, descs_embedding)
            self.input_size = 7 * self.emb_dim // self.div + self.emb_dim + data_feature['len_emb_features'] * self.pro_emb_factor * self.emb_dim
        elif self.feature.lower() == 'v3':
            self.product_emb = ProductEmbeddingV3(config, data_feature, products_input, 
                                                  word2vec_embedding, titles_embedding, descs_embedding)
            self.input_size = 7 * self.emb_dim // self.div + self.emb_dim + data_feature['len_emb_features'] * self.pro_emb_factor * self.emb_dim

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.product_emb.product_fea_emb.weight[:-1]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A):
        hidden, _ = self.product_emb(inputs, None)
        hidden = self.gnn(A, hidden)
        return hidden


class Dice(nn.Module):
    """The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.

    Input shape:
        - 2 dims: [batch_size, embedding_size(features)]
        - 3 dims: [batch_size, num_features, embedding_size(features)]

    Output shape:
        - Same shape as input.

    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
        - https://github.com/zhougr1993/DeepInterestNetwork, https://github.com/fanoping/DIN-pytorch
    """

    def __init__(self, emb_size, dim=2, epsilon=1e-8, device='cpu'):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3

        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        # wrap alpha in nn.Parameter to make it trainable
        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((emb_size,)).to(device))
        else:
            self.alpha = nn.Parameter(torch.zeros((emb_size, 1)).to(device))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        return out


class Identity(nn.Module):

    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs
    

def activation_layer(act_name, hidden_size=None, dice_dim=2):
    """Construct activation layers

    Args:
        act_name: str or nn.Module, name of activation function
        hidden_size: int, used for Dice activation
        dice_dim: int, used for Dice activation
    Return:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'linear':
            act_layer = Identity()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'dice':
            assert dice_dim
            act_layer = Dice(hidden_size, dice_dim)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer


class DNN(nn.Module):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **inputs_dim**: input feature dimension.

        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input
    

class LocalActivationUnit(nn.Module):
    """The LocalActivationUnit used in DIN with which the representation of
        user interests varies adaptively given different candidate items.

    Input shape
        - A list of two 3D tensor with shape:  ``(batch_size, 1, embedding_size)`` and ``(batch_size, T, embedding_size)``

    Output shape
        - 3D tensor with shape: ``(batch_size, T, 1)``.

    Arguments
        - **hidden_units**:list of positive integer, the attention net layer number and units in each layer.

        - **activation**: Activation function to use in attention net.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix of attention net.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout in attention net.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not in attention net.

        - **seed**: A Python integer to use as random seed.

    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, hidden_units=(64, 32), embedding_dim=4, activation='sigmoid', dropout_rate=0, dice_dim=3,
                 l2_reg=0, use_bn=False):
        super(LocalActivationUnit, self).__init__()

        self.dnn = DNN(inputs_dim=4 * embedding_dim,
                       hidden_units=hidden_units,
                       activation=activation,
                       l2_reg=l2_reg,
                       dropout_rate=dropout_rate,
                       dice_dim=dice_dim,
                       use_bn=use_bn)

        self.dense = nn.Linear(hidden_units[-1], 1)

    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        user_behavior_len = user_behavior.size(1)

        queries = query.expand(-1, user_behavior_len, -1)

        attention_input = torch.cat([queries, user_behavior, queries - user_behavior, queries * user_behavior],
                                    dim=-1)  # as the source code, subtraction simulates verctors' difference
        attention_output = self.dnn(attention_input)

        attention_score = self.dense(attention_output)  # [B, T, 1]

        return attention_score
    

class AttentionSequencePoolingLayer(nn.Module):
    """The Attentional sequence pooling operation used in DIN & DIEN.

        Arguments
          - **att_hidden_units**:list of positive integer, the attention net layer number and units in each layer.

          - **att_activation**: Activation function to use in attention net.

          - **weight_normalization**: bool.Whether normalize the attention score of local activation unit.

          - **supports_masking**:If True,the input need to support masking.

        References
          - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
      """

    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', weight_normalization=False,
                 return_score=False, supports_masking=False, embedding_dim=4, **kwargs):
        super(AttentionSequencePoolingLayer, self).__init__()
        self.return_score = return_score
        self.weight_normalization = weight_normalization
        self.supports_masking = supports_masking
        self.local_att = LocalActivationUnit(hidden_units=att_hidden_units, embedding_dim=embedding_dim,
                                             activation=att_activation,
                                             dropout_rate=0, use_bn=False)

    def forward(self, query, keys, keys_length, mask=None):
        """
        Input shape
          - A list of three tensor: [query,keys,keys_length]

          - query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)``

          - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``

          - keys_length is a 2D tensor with shape: ``(batch_size, 1)``

        Output shape
          - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.
        """
        batch_size, max_length, _ = keys.size()

        # Mask
        if self.supports_masking:
            if mask is None:
                raise ValueError("When supports_masking=True,input must support masking")
            keys_masks = mask.unsqueeze(1)
        else:
            keys_masks = torch.arange(max_length, device=keys_length.device, dtype=keys_length.dtype).repeat(batch_size,
                                                                                                             1)  # [B, T]
            keys_masks = keys_masks < keys_length.view(-1, 1)  # 0, 1 mask
            keys_masks = keys_masks.unsqueeze(1)  # [B, 1, T]

        attention_score = self.local_att(query, keys)  # [B, T, 1]

        outputs = torch.transpose(attention_score, 1, 2)  # [B, 1, T]

        if self.weight_normalization:
            paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            paddings = torch.zeros_like(outputs)

        outputs = torch.where(keys_masks, outputs, paddings)  # [B, 1, T]

        # Scale
        # outputs = outputs / (keys.shape[-1] ** 0.05)

        if self.weight_normalization:
            outputs = F.softmax(outputs, dim=-1)  # [B, 1, T]

        if not self.return_score:
            # Weighted sum
            outputs = torch.matmul(outputs, keys)  # [B, 1, E]

        return outputs


class PredictionLayer(nn.Module):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")

        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, X):
        output = X
        if self.use_bias:
            output += self.bias
        if self.task == "binary":
            output = torch.sigmoid(output)
        return output


class DIN(nn.Module):
    """Instantiates the Deep Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool. Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return:  A PyTorch model instance.

    """
    def __init__(self, config, data_feature, products_input, word2vec_embedding, titles_embedding, descs_embedding):
        super(DIN, self).__init__()

        self.hidden_size = config['hid_dim']
        self.emb_dim = config['emb_dim']
        self.device = config['device']
        self.layers = config['layers']
        self.dropout_p = config['dropout']
        self.emb_dim = config['emb_dim']
        self.seq_model = config['seq_model']
        self.feature = config['feature']
        self.div = config['div']
        self.add_mode = config['add_mode']
        self.id_count = data_feature['id_count']
        self.intra = config['intra']
        self.pro_emb_factor = config['pro_emb_factor']
        self.pca = config['pca']
        self.pca_dim = config['pca_dim']

        dnn_use_bn = False
        dnn_hidden_units = (self.hidden_size, self.hidden_size // 2)
        dnn_activation = 'relu'
        att_hidden_size = (self.hidden_size // 4, self.hidden_size // 16)
        att_activation = 'Dice'
        att_weight_normalization = False
        l2_reg_dnn = 0.0
        dnn_dropout = config['dropout']

        if self.feature.lower() == 'v1':
            self.product_emb = ProductEmbedding(config, data_feature, products_input, 
                                                word2vec_embedding, titles_embedding, descs_embedding)
            self.seq_fea_emb = SeqFeatureEmbedding(config, data_feature)
            seq_fea_dim = data_feature['seq_emb_factor'] * self.emb_dim
            self.input_size = data_feature['len_features'] * self.emb_dim // self.div + data_feature['len_emb_features'] * self.pro_emb_factor * self.emb_dim
        elif self.feature.lower() == 'v2':
            self.product_emb = ProductEmbeddingV2(config, data_feature, products_input, 
                                                  word2vec_embedding, titles_embedding, descs_embedding)
            self.seq_fea_emb = SeqFeatureEmbeddingV2(config, data_feature)
            seq_fea_dim = data_feature['len_seqs_cat_feas'] * self.emb_dim
            self.input_size = 7 * self.emb_dim // self.div + self.emb_dim + data_feature['len_emb_features'] * self.pro_emb_factor * self.emb_dim
        elif self.feature.lower() == 'v3':
            self.product_emb = ProductEmbeddingV3(config, data_feature, products_input, 
                                                  word2vec_embedding, titles_embedding, descs_embedding)
            self.seq_fea_emb = SeqFeatureEmbeddingV3(config, data_feature, product_emb=self.product_emb.product_fea)
            if self.add_mode:
                seq_fea_dim = data_feature['len_seqs_cat_feas'] * self.emb_dim
            else:
                seq_fea_dim = (data_feature['len_seqs_cat_feas'] - 6) * self.emb_dim
            self.input_size = 7 * self.emb_dim // self.div + self.emb_dim + data_feature['len_emb_features'] * self.pro_emb_factor * self.emb_dim
        if self.pca:
            self.input_size += self.pca_dim * 2

        att_emb_dim = self.input_size
        self.attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                       embedding_dim=att_emb_dim,
                                                       att_activation=att_activation,
                                                       return_score=False,
                                                       supports_masking=False,
                                                       weight_normalization=att_weight_normalization)
        dnn_input_dim = self.input_size + seq_fea_dim + self.input_size
        self.dnn = DNN(inputs_dim=dnn_input_dim,
                       hidden_units=dnn_hidden_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self.out = PredictionLayer(task='binary')
    
    def forward(self, batch_prev_items, batch_locale, batch_candidate, batch_len, batch_label, batch_seq_cat, batch_seq_num, batch_mask=None, train=True):
        """
        Args:
            batch_prev_items: (B, len)
            batch_locale: (B,)
            batch_candidate: (B,)
            batch_len: (B,)
            batch_label: (B, 1)
            batch_mask: (B, len)
            batch_seq_cat: (B, len_seqs_cat_feas)
            batch_seq_num: (B, len_seqs_num_feas)
        
        Return:
            candidate_prob (tensor): 对候选集下一跳的概率预测 (batch_size, candidate_size)
        """
        batch_prev_items_emb, batch_candidate_emb = self.product_emb(batch_prev_items, batch_candidate)
        # batch_prev_items_emb (B, len, input_size)
        # batch_candidate_emb (B, input_size) or None
        batch_candidate_emb = batch_candidate_emb.unsqueeze(1)  # (B, 1, input_size)

        seq_fea = self.seq_fea_emb(batch_seq_cat, batch_seq_num).unsqueeze(1)  # (B, 1, seq_fea_dim)
        deep_input_emb = torch.cat([batch_candidate_emb, seq_fea], dim=-1)  # (B, 1, input_size + seq_fea_dim)

        query_emb = batch_candidate_emb  # (B, 1, input_size)
        keys_emb = batch_prev_items_emb  #  (B, len, input_size)
        keys_length = batch_len.unsqueeze(1)  # (B, 1)

        hist = self.attention(query_emb, keys_emb, keys_length) # (B, 1, input_size)

        # deep part
        deep_input_emb = torch.cat((deep_input_emb, hist), dim=-1)  # (B, 1, input_size + seq_fea_dim + input_size)
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)  # (B, input_size + seq_fea_dim + input_size)

        dnn_output = self.dnn(deep_input_emb)  # (B, h_out)
        dnn_logit = self.dnn_linear(dnn_output)  # (B, 1)

        y_pred = self.out(dnn_logit)

        return y_pred

    def predict(self, batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_seq_cat, batch_seq_num, batch_mask=None):
        """预测
        Return:
            y_pred (tensor): softmax 后对候选集下一跳的概率预测 (batch_size, 1)
        """
        y_pred = self.forward(batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_seq_cat, batch_seq_num, batch_mask, False)
        loss = F.binary_cross_entropy(y_pred, batch_label)
        return y_pred, loss


class DIEN(nn.Module):
    """Instantiates the Deep Interest Evolution Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param gru_type: str,can be GRU AIGRU AUGRU AGRU
    :param use_negsampling: bool, whether or not use negtive sampling
    :param alpha: float ,weight of auxiliary_loss
    :param use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param dnn_activation: Activation function to use in DNN
    :param att_hidden_units: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, config, data_feature, products_input, word2vec_embedding, titles_embedding, descs_embedding):
        super(DIEN, self).__init__()

        self.hidden_size = config['hid_dim']
        self.emb_dim = config['emb_dim']
        self.device = config['device']
        self.layers = config['layers']
        self.dropout_p = config['dropout']
        self.emb_dim = config['emb_dim']
        self.seq_model = config['seq_model']
        self.feature = config['feature']
        self.div = config['div']
        self.add_mode = config['add_mode']
        self.id_count = data_feature['id_count']
        self.intra = config['intra']
        self.pro_emb_factor = config['pro_emb_factor']
        self.pca = config['pca']
        self.pca_dim = config['pca_dim']

        dnn_use_bn = False
        dnn_hidden_units = (self.hidden_size, self.hidden_size // 2)
        dnn_activation = 'relu'
        att_hidden_size = (self.hidden_size // 4, self.hidden_size // 16)
        att_activation = 'Dice'
        att_weight_normalization = False
        l2_reg_dnn = 0.0
        dnn_dropout = config['dropout']

        gru_type="GRU"
        use_negsampling=False
        alpha=1.0
        use_bn=False
        dnn_hidden_units=(self.hidden_size, self.hidden_size // 2)
        dnn_activation='relu'
        att_hidden_units=(self.hidden_size // 4, self.hidden_size // 16)
        att_activation="relu"
        att_weight_normalization=True,
        l2_reg_dnn=0
        l2_reg_embedding=1e-6
        dnn_dropout=config['dropout']
        init_std=0.0001

        if self.feature.lower() == 'v1':
            self.product_emb = ProductEmbedding(config, data_feature, products_input, 
                                                word2vec_embedding, titles_embedding, descs_embedding)
            self.seq_fea_emb = SeqFeatureEmbedding(config, data_feature)
            seq_fea_dim = data_feature['seq_emb_factor'] * self.emb_dim
            self.input_size = data_feature['len_features'] * self.emb_dim // self.div + data_feature['len_emb_features'] * self.pro_emb_factor * self.emb_dim
        elif self.feature.lower() == 'v2':
            self.product_emb = ProductEmbeddingV2(config, data_feature, products_input, 
                                                  word2vec_embedding, titles_embedding, descs_embedding)
            self.seq_fea_emb = SeqFeatureEmbeddingV2(config, data_feature)
            seq_fea_dim = data_feature['len_seqs_cat_feas'] * self.emb_dim
            self.input_size = 7 * self.emb_dim // self.div + self.emb_dim + data_feature['len_emb_features'] * self.pro_emb_factor * self.emb_dim
        elif self.feature.lower() == 'v3':
            self.product_emb = ProductEmbeddingV3(config, data_feature, products_input, 
                                                  word2vec_embedding, titles_embedding, descs_embedding)
            self.seq_fea_emb = SeqFeatureEmbeddingV3(config, data_feature, product_emb=self.product_emb.product_fea)
            if self.add_mode:
                seq_fea_dim = data_feature['len_seqs_cat_feas'] * self.emb_dim
            else:
                seq_fea_dim = (data_feature['len_seqs_cat_feas'] - 6) * self.emb_dim
            self.input_size = 7 * self.emb_dim // self.div + self.emb_dim + data_feature['len_emb_features'] * self.pro_emb_factor * self.emb_dim
        if self.pca:
            self.input_size += self.pca_dim * 2

        # interest extractor layer
        self.interest_extractor = InterestExtractor(input_size=self.input_size, use_neg=use_negsampling, init_std=init_std)
        # interest evolution layer
        self.interest_evolution = InterestEvolving(
            input_size=self.input_size,
            gru_type=gru_type,
            use_neg=use_negsampling,
            init_std=init_std,
            att_hidden_size=att_hidden_units,
            att_activation=att_activation,
            att_weight_normalization=att_weight_normalization)
        # DNN layer
        dnn_input_dim = self.input_size + seq_fea_dim + self.input_size
        self.dnn = DNN(dnn_input_dim, dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, use_bn,
                       init_std=init_std)
        self.linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self.out = PredictionLayer(task='binary')
        # prediction layer
        # inherit -> self.out

        # init
        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, batch_prev_items, batch_locale, batch_candidate, batch_len, batch_label, batch_seq_cat, batch_seq_num, batch_mask=None, train=True):
        """
        Args:
            batch_prev_items: (B, len)
            batch_locale: (B,)
            batch_candidate: (B,)
            batch_len: (B,)
            batch_label: (B, 1)
            batch_mask: (B, len)
            batch_seq_cat: (B, len_seqs_cat_feas)
            batch_seq_num: (B, len_seqs_num_feas)
        
        Return:
            candidate_prob (tensor): 对候选集下一跳的概率预测 (batch_size, candidate_size)
        """
        batch_prev_items_emb, batch_candidate_emb = self.product_emb(batch_prev_items, batch_candidate)
        # batch_prev_items_emb (B, len, input_size)
        # batch_candidate_emb (B, input_size) or None

        seq_fea = self.seq_fea_emb(batch_seq_cat, batch_seq_num)  # (B, seq_fea_dim)
        deep_input_emb = torch.cat([batch_candidate_emb, seq_fea], dim=-1)  # (B, input_size + seq_fea_dim)

        query_emb = batch_candidate_emb  # (B, input_size)
        keys_emb = batch_prev_items_emb  #  (B, len, input_size)
        keys_length = batch_len  # (B)

        neg_keys_emb = None

        # [b, T, H],  [1]  (b<H)
        masked_interest, aux_loss = self.interest_extractor(keys_emb, keys_length, neg_keys_emb)
        # self.add_auxiliary_loss(aux_loss, self.alpha)
        # [B, H]
        hist = self.interest_evolution(query_emb, masked_interest, keys_length)
        # [B, 1]
        deep_input_emb = torch.cat((deep_input_emb, hist), dim=-1)  # (B, input_size + seq_fea_dim + input_size)
        output = self.linear(self.dnn(deep_input_emb))
        y_pred = self.out(output)
        return y_pred

    def predict(self, batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_seq_cat, batch_seq_num, batch_mask=None):
        """预测
        Return:
            y_pred (tensor): softmax 后对候选集下一跳的概率预测 (batch_size, 1)
        """
        y_pred = self.forward(batch_prev_items, batch_locale, batch_candidate_set, batch_len, batch_label, batch_seq_cat, batch_seq_num, batch_mask, False)
        loss = F.binary_cross_entropy(y_pred, batch_label)
        return y_pred, loss


class InterestExtractor(nn.Module):
    def __init__(self, input_size, use_neg=False, init_std=0.001, device='cpu'):
        super(InterestExtractor, self).__init__()
        self.use_neg = use_neg
        self.gru = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        if self.use_neg:
            self.auxiliary_net = DNN(input_size * 2, [100, 50, 1], 'sigmoid', init_std=init_std, device=device)
        for name, tensor in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)
        self.to(device)

    def forward(self, keys, keys_length, neg_keys=None):
        """
        Parameters
        ----------
        keys: 3D tensor, [B, T, H]
        keys_length: 1D tensor, [B]
        neg_keys: 3D tensor, [B, T, H]

        Returns
        -------
        masked_interests: 2D tensor, [b, H]
        aux_loss: [1]
        """
        batch_size, max_length, dim = keys.size()
        zero_outputs = torch.zeros(batch_size, dim, device=keys.device)
        aux_loss = torch.zeros((1,), device=keys.device)

        # create zero mask for keys_length, to make sure 'pack_padded_sequence' safe
        mask = keys_length > 0
        masked_keys_length = keys_length[mask]

        # batch_size validation check
        if masked_keys_length.shape[0] == 0:
            return zero_outputs,

        masked_keys = torch.masked_select(keys, mask.view(-1, 1, 1)).view(-1, max_length, dim)

        packed_keys = pack_padded_sequence(masked_keys, lengths=masked_keys_length.cpu(), batch_first=True,
                                           enforce_sorted=False)
        packed_interests, _ = self.gru(packed_keys)
        interests, _ = pad_packed_sequence(packed_interests, batch_first=True, padding_value=0.0,
                                           total_length=max_length)

        if self.use_neg and neg_keys is not None:
            masked_neg_keys = torch.masked_select(neg_keys, mask.view(-1, 1, 1)).view(-1, max_length, dim)
            aux_loss = self._cal_auxiliary_loss(
                interests[:, :-1, :],
                masked_keys[:, 1:, :],
                masked_neg_keys[:, 1:, :],
                masked_keys_length - 1)

        return interests, aux_loss

    def _cal_auxiliary_loss(self, states, click_seq, noclick_seq, keys_length):
        # keys_length >= 1
        mask_shape = keys_length > 0
        keys_length = keys_length[mask_shape]
        if keys_length.shape[0] == 0:
            return torch.zeros((1,), device=states.device)

        _, max_seq_length, embedding_size = states.size()
        states = torch.masked_select(states, mask_shape.view(-1, 1, 1)).view(-1, max_seq_length, embedding_size)
        click_seq = torch.masked_select(click_seq, mask_shape.view(-1, 1, 1)).view(-1, max_seq_length, embedding_size)
        noclick_seq = torch.masked_select(noclick_seq, mask_shape.view(-1, 1, 1)).view(-1, max_seq_length,
                                                                                       embedding_size)
        batch_size = states.size()[0]

        mask = (torch.arange(max_seq_length, device=states.device).repeat(
            batch_size, 1) < keys_length.view(-1, 1)).float()

        click_input = torch.cat([states, click_seq], dim=-1)
        noclick_input = torch.cat([states, noclick_seq], dim=-1)
        embedding_size = embedding_size * 2

        click_p = self.auxiliary_net(click_input.view(
            batch_size * max_seq_length, embedding_size)).view(
            batch_size, max_seq_length)[mask > 0].view(-1, 1)
        click_target = torch.ones(
            click_p.size(), dtype=torch.float, device=click_p.device)

        noclick_p = self.auxiliary_net(noclick_input.view(
            batch_size * max_seq_length, embedding_size)).view(
            batch_size, max_seq_length)[mask > 0].view(-1, 1)
        noclick_target = torch.zeros(
            noclick_p.size(), dtype=torch.float, device=noclick_p.device)

        loss = F.binary_cross_entropy(
            torch.cat([click_p, noclick_p], dim=0),
            torch.cat([click_target, noclick_target], dim=0))

        return loss


class InterestEvolving(nn.Module):
    __SUPPORTED_GRU_TYPE__ = ['GRU', 'AIGRU', 'AGRU', 'AUGRU']

    def __init__(self,
                 input_size,
                 gru_type='GRU',
                 use_neg=False,
                 init_std=0.001,
                 att_hidden_size=(64, 16),
                 att_activation='sigmoid',
                 att_weight_normalization=False):
        super(InterestEvolving, self).__init__()
        if gru_type not in InterestEvolving.__SUPPORTED_GRU_TYPE__:
            raise NotImplementedError("gru_type: {gru_type} is not supported")
        self.gru_type = gru_type
        self.use_neg = use_neg

        if gru_type == 'GRU':
            self.attention = AttentionSequencePoolingLayer(embedding_dim=input_size,
                                                           att_hidden_units=att_hidden_size,
                                                           att_activation=att_activation,
                                                           weight_normalization=att_weight_normalization,
                                                           return_score=False)
            self.interest_evolution = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == 'AIGRU':
            self.attention = AttentionSequencePoolingLayer(embedding_dim=input_size,
                                                           att_hidden_units=att_hidden_size,
                                                           att_activation=att_activation,
                                                           weight_normalization=att_weight_normalization,
                                                           return_score=True)
            self.interest_evolution = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == 'AGRU' or gru_type == 'AUGRU':
            self.attention = AttentionSequencePoolingLayer(embedding_dim=input_size,
                                                           att_hidden_units=att_hidden_size,
                                                           att_activation=att_activation,
                                                           weight_normalization=att_weight_normalization,
                                                           return_score=True)
            self.interest_evolution = DynamicGRU(input_size=input_size, hidden_size=input_size,
                                                 gru_type=gru_type)
        for name, tensor in self.interest_evolution.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    @staticmethod
    def _get_last_state(states, keys_length):
        # states [B, T, H]
        batch_size, max_seq_length, _ = states.size()

        mask = (torch.arange(max_seq_length, device=keys_length.device).repeat(
            batch_size, 1) == (keys_length.view(-1, 1) - 1))

        return states[mask]

    def forward(self, query, keys, keys_length, mask=None):
        """
        Parameters
        ----------
        query: 2D tensor, [B, H]
        keys: (masked_interests), 3D tensor, [b, T, H]
        keys_length: 1D tensor, [B]

        Returns
        -------
        outputs: 2D tensor, [B, H]
        """
        batch_size, dim = query.size()
        max_length = keys.size()[1]

        # check batch validation
        zero_outputs = torch.zeros(batch_size, dim, device=query.device)
        mask = keys_length > 0
        # [B] -> [b]
        keys_length = keys_length[mask]
        if keys_length.shape[0] == 0:
            return zero_outputs

        # [B, H] -> [b, 1, H]
        query = torch.masked_select(query, mask.view(-1, 1)).view(-1, dim).unsqueeze(1)

        if self.gru_type == 'GRU':
            packed_keys = pack_padded_sequence(keys, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False)
            packed_interests, _ = self.interest_evolution(packed_keys)
            interests, _ = pad_packed_sequence(packed_interests, batch_first=True, padding_value=0.0,
                                               total_length=max_length)
            outputs = self.attention(query, interests, keys_length.unsqueeze(1))  # [b, 1, H]
            outputs = outputs.squeeze(1)  # [b, H]
        elif self.gru_type == 'AIGRU':
            att_scores = self.attention(query, keys, keys_length.unsqueeze(1))  # [b, 1, T]
            interests = keys * att_scores.transpose(1, 2)  # [b, T, H]
            packed_interests = pack_padded_sequence(interests, lengths=keys_length.cpu(), batch_first=True,
                                                    enforce_sorted=False)
            _, outputs = self.interest_evolution(packed_interests)
            outputs = outputs.squeeze(0) # [b, H]
        elif self.gru_type == 'AGRU' or self.gru_type == 'AUGRU':
            att_scores = self.attention(query, keys, keys_length.unsqueeze(1)).squeeze(1)  # [b, T]
            packed_interests = pack_padded_sequence(keys, lengths=keys_length.cpu(), batch_first=True,
                                                    enforce_sorted=False)
            packed_scores = pack_padded_sequence(att_scores, lengths=keys_length.cpu(), batch_first=True,
                                                 enforce_sorted=False)
            outputs = self.interest_evolution(packed_interests, packed_scores)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True, padding_value=0.0, total_length=max_length)
            # pick last state
            outputs = InterestEvolving._get_last_state(outputs, keys_length) # [b, H]
        # [b, H] -> [B, H]
        zero_outputs[mask] = outputs
        return zero_outputs


class AGRUCell(nn.Module):
    """ Attention based GRU (AGRU)

        Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_ih)
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.register_parameter('weight_ih', self.weight_ih)
        # (W_hr|W_hz|W_hh)
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            # (b_ir|b_iz|b_ih)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            # (b_hr|b_hz|b_hh)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_hh', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor, )
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, inputs, hx, att_score):
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, _, i_n = gi.chunk(3, 1)
        h_r, _, h_n = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        # update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)

        att_score = att_score.view(-1, 1)
        hy = (1. - att_score) * hx + att_score * new_state
        return hy


class AUGRUCell(nn.Module):
    """ Effect of GRU with attentional update gate (AUGRU)

        Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AUGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_ih)
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.register_parameter('weight_ih', self.weight_ih)
        # (W_hr|W_hz|W_hh)
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            # (b_ir|b_iz|b_ih)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            # (b_hr|b_hz|b_hh)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor, )
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, inputs, hx, att_score):
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)

        att_score = att_score.view(-1, 1)
        update_gate = att_score * update_gate
        hy = (1. - update_gate) * hx + update_gate * new_state
        return hy


class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, gru_type='AGRU'):
        super(DynamicGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if gru_type == 'AGRU':
            self.rnn = AGRUCell(input_size, hidden_size, bias)
        elif gru_type == 'AUGRU':
            self.rnn = AUGRUCell(input_size, hidden_size, bias)

    def forward(self, inputs, att_scores=None, hx=None):
        if not isinstance(inputs, PackedSequence) or not isinstance(att_scores, PackedSequence):
            raise NotImplementedError("DynamicGRU only supports packed input and att_scores")

        inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
        att_scores, _, _, _ = att_scores

        max_batch_size = int(batch_sizes[0])
        if hx is None:
            hx = torch.zeros(max_batch_size, self.hidden_size,
                             dtype=inputs.dtype, device=inputs.device)

        outputs = torch.zeros(inputs.size(0), self.hidden_size,
                              dtype=inputs.dtype, device=inputs.device)

        begin = 0
        for batch in batch_sizes:
            new_hx = self.rnn(
                inputs[begin:begin + batch],
                hx[0:batch],
                att_scores[begin:begin + batch])
            outputs[begin:begin + batch] = new_hx
            hx = new_hx
            begin += batch
        return PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices)
