import numpy as np
import rdkit
import rdkit.Chem as Chem
import networkx as nx
from build_vocab import WordVocab
import pandas as pd

import os
import torch.nn as nn

from utils import *

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GINConv
from pool import GraphMultisetTransformer
from guided_cross_attention_model import GuidedCrossAttention


#############################


class SpatialGroupEnhance_for_1D(nn.Module):
    def __init__(self, groups=32):
        super(SpatialGroupEnhance_for_1D, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.weight = Parameter(torch.zeros(1, groups, 1))
        self.bias = Parameter(torch.ones(1, groups, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):  # (b, c, h)
        b, c, h = x.size()
        x = x.view(b * self.groups, -1, h)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h)
        x = x * self.sig(t)
        x = x.view(b, c, h)
        return x


class LinkAttention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(LinkAttention, self).__init__()
        self.query = nn.Linear(input_dim, n_heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, masks):
        query = self.query(x).transpose(1, 2)  # (B,heads,seq_len)
        value = x  # (B,seq_len,hidden_dim)

        minus_inf = -9e15 * torch.ones_like(query)  # (B,heads,seq_len)
        e = torch.where(masks > 0.5, query, minus_inf)  # (B,heads,seq_len)
        a = self.softmax(e)  # (B,heads,seq_len)

        out = torch.matmul(a, value)  # (B,heads,seq_len) * (B,seq_len,hidden_dim) = (B,heads,hidden_dim)
        out = torch.mean(out, dim=1).squeeze()  # (B,hidden_dim)
        return out, a


class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=128, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GINConvNet, self).__init__()

        dim = 256
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # combined layers
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)  # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        # target = data.target
        x = F.relu(self.conv1(x, edge_index))  # (B,seq_len,hidden_dim)
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))  # (B,seq_len,hidden_dim)
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))  # (B,seq_len,hidden_dim)
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))  # (B,seq_len,hidden_dim)
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))  # (B,seq_len,hidden_dim)
        x = self.bn5(x)
        out = global_add_pool(x, batch)  # (B,hidden_dim)
        # x = F.relu(self.fc1_xd(x)) # (B,hidden_dim)
        # x = F.dropout(x, p=0.2, training=self.training) # (B,hidden_dim)
        # # concat
        # xc = x
        # xc = self.fc1(xc)
        # xc = self.relu(xc)
        # xc = self.dropout(xc)
        # xc = self.fc2(xc)
        # xc = self.relu(xc)
        # xc = self.dropout(xc)
        # out = self.out(xc)
        return out


class GraphCNN_P(nn.Module):
    def __init__(self, channel_dims=[512, 256, 256], fc_dim=512, num_classes=256, pooling='MTP'):
        super(GraphCNN_P, self).__init__()

        # Define graph convolutional layers
        gcn_dims = [512] + channel_dims

        gcn_layers = [GCNConv(gcn_dims[i - 1], gcn_dims[i], bias=True) for i in range(1, len(gcn_dims))]

        self.gcn = nn.ModuleList(gcn_layers)
        self.pooling = pooling
        if self.pooling == "MTP":
            self.pool = GraphMultisetTransformer(512, 512, 256, None, 2000, 0.25, ['GMPool_G', 'GMPool_G'],
                                                 num_heads=8, layer_norm=True)
        else:
            self.pool = gmp
        # Define dropout
        self.drop1 = nn.Dropout(p=0.2)

        ###gin_layer
        nn1 = Sequential(Linear(gcn_dims[0], gcn_dims[1]), ReLU(), Linear(gcn_dims[1], gcn_dims[1]))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(gcn_dims[1])

        nn2 = Sequential(Linear(gcn_dims[1], gcn_dims[1]), ReLU(), Linear(gcn_dims[1], gcn_dims[1]))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(gcn_dims[1])

        nn3 = Sequential(Linear(gcn_dims[1], gcn_dims[1]), ReLU(), Linear(gcn_dims[1], gcn_dims[1]))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(gcn_dims[1])

        self.gin_layers = [self.conv1, self.conv2, self.conv3]
        self.bn_layers = [self.bn1, self.bn2, self.bn3]
        self.gin = nn.ModuleList(self.gin_layers)

    def forward(self, x, data, pertubed=False):
        # x = data.x
        # Compute graph convolutional part
        x = self.drop1(x)
        for idx, gin_layer in enumerate(self.gin):
            if idx == 0:
                x = F.relu(gin_layer(x, data.edge_index.long()))
                x = self.bn_layers[idx](x)
            else:
                x = x + F.relu(gin_layer(x, data.edge_index.long()))
                x = self.bn_layers[idx](x)
            if pertubed:
                random_noise = torch.rand_like(x).to(x.device)
                x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
        if self.pooling == 'MTP':
            # Apply GraphMultisetTransformer Pooling
            g_level_feat = self.pool(x, data.batch, data.edge_index.long())
        else:
            g_level_feat = self.pool(x, data.batch)

        n_level_feat = x

        return n_level_feat, g_level_feat


class GraphCNN_D(nn.Module):
    def __init__(self, channel_dims=[512, 256, 512], fc_dim=512, num_classes=256, pooling='MTP'):
        super(GraphCNN_D, self).__init__()

        # Define graph convolutional layers
        gcn_dims = [512] + channel_dims

        gcn_layers = [GCNConv(gcn_dims[i - 1], gcn_dims[i], bias=True) for i in range(1, len(gcn_dims))]

        self.gcn = nn.ModuleList(gcn_layers)
        self.pooling = pooling
        if self.pooling == "MTP":
            self.pool = GraphMultisetTransformer(512, 512, 256, None, 200, 0.25, ['GMPool_G', 'GMPool_G'],
                                                 num_heads=8, layer_norm=True)
        else:
            self.pool = gmp
        # Define dropout
        self.drop1 = nn.Dropout(p=0.2)

        ###gin_layer
        nn1 = Sequential(Linear(gcn_dims[0], gcn_dims[1]), ReLU(), Linear(gcn_dims[1], gcn_dims[1]))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(gcn_dims[1])

        nn2 = Sequential(Linear(gcn_dims[1], gcn_dims[1]), ReLU(), Linear(gcn_dims[1], gcn_dims[1]))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(gcn_dims[1])

        nn3 = Sequential(Linear(gcn_dims[1], gcn_dims[1]), ReLU(), Linear(gcn_dims[1], gcn_dims[1]))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(gcn_dims[1])

        self.gin_layers = [self.conv1, self.conv2, self.conv3]
        self.bn_layers = [self.bn1, self.bn2, self.bn3]
        self.gin = nn.ModuleList(self.gin_layers)

    def forward(self, x, data, pertubed=False):
        # x = data.x
        # Compute graph convolutional part
        x = self.drop1(x)
        for idx, gin_layer in enumerate(self.gin):
            if idx == 0:
                x = F.relu(gin_layer(x, data.edge_index.long()))
                x = self.bn_layers[idx](x)
            else:
                x = x + F.relu(gin_layer(x, data.edge_index.long()))
                x = self.bn_layers[idx](x)
            if pertubed:
                random_noise = torch.rand_like(x).to(x.device)
                x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
        if self.pooling == 'MTP':
            # Apply GraphMultisetTransformer Pooling
            g_level_feat = self.pool(x, data.batch, data.edge_index.long())
        else:
            g_level_feat = self.pool(x, data.batch)

        n_level_feat = x

        return n_level_feat, g_level_feat




class CAN_Layer(nn.Module):
    def __init__(self, hidden_dim, num_heads_d, num_heads_p, group_size_d, group_size_p, agg_mode):
        super(CAN_Layer, self).__init__()
        self.agg_mode = agg_mode
        self.group_size_d = group_size_d  # Control Fusion Scale for drug
        self.group_size_p = group_size_p  # Control Fusion Scale for protein
        self.hidden_dim = hidden_dim
        self.num_heads_d = num_heads_d
        self.num_heads_p = num_heads_p
        self.head_size_d = hidden_dim // num_heads_d
        self.head_size_p = hidden_dim // num_heads_p

        self.query_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_p = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.query_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_d = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def alpha_logits(self, logits, mask_row, mask_col, inf=1e6):
        N, L1, L2, H = logits.shape
        mask_row = mask_row.view(N, L1, 1).repeat(1, 1, H)
        mask_col = mask_col.view(N, L2, 1).repeat(1, 1, H)
        mask_pair = torch.einsum('blh, bkh->blkh', mask_row, mask_col)

        logits = torch.where(mask_pair, logits, logits - inf)
        alpha = torch.softmax(logits, dim=2)
        mask_row = mask_row.view(N, L1, 1, H).repeat(1, 1, L2, 1)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
        return alpha

    def apply_heads(self, x, n_heads, n_ch):
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def group_embeddings(self, x, mask, group_size):
        N, L, D = x.shape
        groups = L // group_size
        x_grouped = x.view(N, groups, group_size, D).mean(dim=2)
        mask_grouped = mask.view(N, groups, group_size).any(dim=2)
        return x_grouped, mask_grouped

    def forward(self, protein, drug, mask_prot, mask_drug):
        # Group embeddings before applying multi-head attention
        protein_grouped, mask_prot_grouped = self.group_embeddings(protein, mask_prot, self.group_size_p)
        drug_grouped, mask_drug_grouped = self.group_embeddings(drug, mask_drug, self.group_size_d)

        # Compute queries, keys, values for both protein and drug after grouping
        query_prot = self.apply_heads(self.query_p(protein_grouped), self.num_heads_p, self.head_size_p)
        key_prot = self.apply_heads(self.key_p(protein_grouped), self.num_heads_p, self.head_size_p)
        value_prot = self.apply_heads(self.value_p(protein_grouped), self.num_heads_p, self.head_size_p)

        query_drug = self.apply_heads(self.query_d(drug_grouped), self.num_heads_d, self.head_size_d)
        key_drug = self.apply_heads(self.key_d(drug_grouped), self.num_heads_d, self.head_size_d)
        value_drug = self.apply_heads(self.value_d(drug_grouped), self.num_heads_d, self.head_size_d)

        # Compute attention scores
        # logits_pp = torch.einsum('blhd, bkhd->blkh', query_prot, key_prot)
        logits_pd = torch.einsum('blhd, bkhd->blkh', query_prot, key_drug)
        logits_dp = torch.einsum('blhd, bkhd->blkh', query_drug, key_prot)
        # logits_dd = torch.einsum('blhd, bkhd->blkh', query_drug, key_drug)

        # alpha_pp = self.alpha_logits(logits_pp, mask_prot_grouped, mask_prot_grouped)
        alpha_pd = self.alpha_logits(logits_pd, mask_prot_grouped, mask_drug_grouped)
        alpha_dp = self.alpha_logits(logits_dp, mask_drug_grouped, mask_prot_grouped)
        # alpha_dd = self.alpha_logits(logits_dd, mask_drug_grouped, mask_drug_grouped)

        prot = torch.einsum('blkh, bkhd->blhd', alpha_pd, value_drug).flatten(-2)
        drug = torch.einsum('blkh, bkhd->blhd', alpha_dp, value_prot).flatten(-2)
        return prot, drug, mask_prot_grouped, mask_drug_grouped, alpha_pd, alpha_dp

        prot_embedding = (torch.einsum('blkh, bkhd->blhd', alpha_pp, value_prot).flatten(-2) +
                          torch.einsum('blkh, bkhd->blhd', alpha_pd, value_drug).flatten(-2)) / 2
        drug_embedding = (torch.einsum('blkh, bkhd->blhd', alpha_dp, value_prot).flatten(-2) +
                          torch.einsum('blkh, bkhd->blhd', alpha_dd, value_drug).flatten(-2)) / 2

        # Continue as usual with the aggregation mode
        if self.agg_mode == "cls":
            prot_embed = prot_embedding[:, 0]  # query : [batch_size, hidden]
            drug_embed = drug_embedding[:, 0]  # query : [batch_size, hidden]
        elif self.agg_mode == "mean_all_tok":
            prot_embed = prot_embedding.mean(1)  # query : [batch_size, hidden]
            drug_embed = drug_embedding.mean(1)  # query : [batch_size, hidden]
        elif self.agg_mode == "mean":
            prot_embed = (prot_embedding * mask_prot_grouped.unsqueeze(-1)).sum(1) / mask_prot_grouped.sum(
                -1).unsqueeze(-1)
            drug_embed = (drug_embedding * mask_drug_grouped.unsqueeze(-1)).sum(1) / mask_drug_grouped.sum(
                -1).unsqueeze(-1)
        else:
            raise NotImplementedError()

        query_embed = torch.cat([prot_embed, drug_embed], dim=1)
        return query_embed


class CrossLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads_d, num_heads_p, group_size_d, group_size_p, agg_mode):
        super(CrossLayer, self).__init__()
        self.agg_mode = agg_mode
        self.group_size_d = group_size_d  # Control Fusion Scale for drug
        self.group_size_p = group_size_p  # Control Fusion Scale for protein
        self.hidden_dim = hidden_dim
        self.num_heads_d = num_heads_d
        self.num_heads_p = num_heads_p
        self.head_size_d = hidden_dim // num_heads_d
        self.head_size_p = hidden_dim // num_heads_p

        self.query_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_p = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.query_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_d = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def alpha_logits(self, logits, mask_row, mask_col, inf=1e6):
        N, L1, L2, H = logits.shape
        mask_row = mask_row.view(N, L1, 1).repeat(1, 1, H)
        mask_col = mask_col.view(N, L2, 1).repeat(1, 1, H)
        mask_pair = torch.einsum('blh, bkh->blkh', mask_row, mask_col)

        logits = torch.where(mask_pair, logits, logits - inf)
        alpha = torch.softmax(logits, dim=2)
        mask_row = mask_row.view(N, L1, 1, H).repeat(1, 1, L2, 1)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
        return alpha

    def apply_heads(self, x, n_heads, n_ch):
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def group_embeddings(self, x, mask, group_size):
        N, L, D = x.shape
        groups = L // group_size
        x_grouped = x.view(N, groups, group_size, D).mean(dim=2)
        mask_grouped = mask.view(N, groups, group_size).any(dim=2)
        return x_grouped, mask_grouped

    def forward(self, protein, drug, mask_prot, mask_drug):
        # Group embeddings before applying multi-head attention
        protein_grouped, mask_prot_grouped = self.group_embeddings(protein, mask_prot, self.group_size_p)
        drug_grouped, mask_drug_grouped = self.group_embeddings(drug, mask_drug, self.group_size_d)

        # Compute queries, keys, values for both protein and drug after grouping
        query_prot = self.apply_heads(self.query_p(protein_grouped), self.num_heads_p, self.head_size_p)
        key_prot = self.apply_heads(self.key_p(protein_grouped), self.num_heads_p, self.head_size_p)
        value_prot = self.apply_heads(self.value_p(protein_grouped), self.num_heads_p, self.head_size_p)

        query_drug = self.apply_heads(self.query_d(drug_grouped), self.num_heads_d, self.head_size_d)
        key_drug = self.apply_heads(self.key_d(drug_grouped), self.num_heads_d, self.head_size_d)
        value_drug = self.apply_heads(self.value_d(drug_grouped), self.num_heads_d, self.head_size_d)

        # Compute attention scores
        logits_pd = torch.einsum('blhd, bkhd->blkh', query_prot, key_drug)
        logits_dp = torch.einsum('blhd, bkhd->blkh', query_drug, key_prot)

        alpha_pd = self.alpha_logits(logits_pd, mask_prot_grouped, mask_drug_grouped)
        alpha_dp = self.alpha_logits(logits_dp, mask_drug_grouped, mask_prot_grouped)

        prot_embedding = torch.einsum('blkh, bkhd->blhd', alpha_pd, value_drug).flatten(-2)
        drug_embedding = torch.einsum('blkh, bkhd->blhd', alpha_dp, value_prot).flatten(-2)

        # Continue as usual with the aggregation mode
        if self.agg_mode == "cls":
            prot_embed = prot_embedding[:, 0]  # query : [batch_size, hidden]
            drug_embed = drug_embedding[:, 0]  # query : [batch_size, hidden]
        elif self.agg_mode == "mean_all_tok":
            prot_embed = prot_embedding.mean(1)  # query : [batch_size, hidden]
            drug_embed = drug_embedding.mean(1)  # query : [batch_size, hidden]
        elif self.agg_mode == "mean":
            prot_embed = (prot_embedding * mask_prot_grouped.unsqueeze(-1)).sum(1) / mask_prot_grouped.sum(
                -1).unsqueeze(-1)
            drug_embed = (drug_embedding * mask_drug_grouped.unsqueeze(-1)).sum(1) / mask_drug_grouped.sum(
                -1).unsqueeze(-1)
        else:
            raise NotImplementedError()

        query_embed = torch.cat([prot_embed, drug_embed], dim=1)
        return query_embed


class FeedForwardLayer(nn.Module):
    def __init__(self, d_in, d_h):
        super().__init__()
        self.lin1 = nn.Linear(d_in, d_h)
        self.lin2 = nn.Linear(d_h, d_in)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_h)

    def forward(self, x):
        x = self.act(self.lin1(x))
        x = self.norm(x)
        x = self.lin2(x)
        return x