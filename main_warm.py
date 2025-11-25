import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
import networkx as nx

import torch

torch.cuda.empty_cache()
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from build_vocab import WordVocab
from utils import *

from dataset import DTADataset
from sklearn.model_selection import KFold
from model import *
from torch import nn as nn
from nt_xent import NT_Xent
import csv
from utils import seed_torch
from torch_geometric.utils import to_dense_batch
from argparse import ArgumentParser

#############################################################################

# CUDA = '0'
# device = torch.device('cuda:'+CUDA)
# LR = 1e-4
# NUM_EPOCHS = 400
# seed = 0
# batch_size = 128
# dataset_name = 'davis'
#
# seed_torch(seed)

CUDA = '0'
device = torch.device('cuda:' + CUDA)
parser = ArgumentParser(description="DTA Training.")
# 优化器学习率
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
# 训练轮数
parser.add_argument('--epochs', type=int, default=180, help='Number of training epochs')
# 随机种子
parser.add_argument('--seed', type=int, default=4, help='Random seed for reproducibility')
# 批次大小
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
# 数据集名称
parser.add_argument('--dataset', type=str, default='kiba',
                    choices=['davis', 'kiba'], help='Dataset name')

args = parser.parse_args()
LR = args.lr
NUM_EPOCHS = args.epochs
seed = args.seed
batch_size = args.batch_size
dataset_name = args.dataset

seed_torch(seed)
#############################################################################
print(seed)


class PMHGT(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, hidden_dim, dropout_rate,
                 alpha, n_heads, bilstm_layers=2, protein_vocab=26,
                 smile_vocab=45, theta=0.5):
        super(PMHGT, self).__init__()
        self.is_bidirectional = True
        # drugs
        self.theta = theta
        self.dropout = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.bilstm_layers = bilstm_layers
        self.n_heads = n_heads
        self.MGNN = GINConvNet(num_features_xd=512, n_output=hidden_dim)

        # SMILES
        self.smiles_vocab = smile_vocab
        self.smiles_embed = nn.Embedding(smile_vocab + 1, 256, padding_idx=0)

        self.is_bidirectional = True
        self.smiles_input_fc = nn.Linear(256, lstm_dim)
        self.smiles_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                   bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.ln1 = torch.nn.LayerNorm(lstm_dim * 2)
        self.enhance1 = SpatialGroupEnhance_for_1D(groups=20)
        self.out_attentions3 = LinkAttention(hidden_dim, n_heads)

        # protein
        self.protein_vocab = protein_vocab
        self.protein_embed = nn.Embedding(protein_vocab + 1, embedding_dim, padding_idx=0)
        self.is_bidirectional = True
        self.protein_input_fc = nn.Linear(embedding_dim, lstm_dim)

        self.protein_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                    bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.ln2 = torch.nn.LayerNorm(lstm_dim * 2)
        self.enhance2 = SpatialGroupEnhance_for_1D(groups=200)
        self.protein_head_fc = nn.Linear(lstm_dim * n_heads, lstm_dim)
        self.protein_out_fc = nn.Linear(2 * lstm_dim, hidden_dim)
        self.out_attentions2 = LinkAttention(hidden_dim, n_heads)

        # link
        self.out_attentions = LinkAttention(hidden_dim, n_heads)
        self.out_fc1 = nn.Linear(hidden_dim * 3, 256 * 8)
        self.out_fc2 = nn.Linear(256 * 8, hidden_dim * 2)

        self.fusion_graph_seq = nn.Linear(hidden_dim * 4, hidden_dim * 2)

        self.out_fc3 = nn.Linear(hidden_dim * 2, 1)
        self.layer_norm = nn.LayerNorm(lstm_dim * 2)

        # Point-wise Feed Forward Network
        self.pwff_1 = nn.Linear(hidden_dim * 4, hidden_dim * 4)
        self.pwff_2 = nn.Linear(hidden_dim * 4, hidden_dim * 4)

        ###交叉注意力
        self.d_gca = GuidedCrossAttention(embed_dim=2 * lstm_dim, num_heads=1)
        self.p_gca = GuidedCrossAttention(embed_dim=2 * lstm_dim, num_heads=1)

        self.one_hot_embed = nn.Embedding(21, 96)
        self.proj_aa = nn.Linear(96, 512)
        self.proj_esm = nn.Linear(1280, 512)

        self.proj_uni = nn.Linear(75, 512)
        self.gcn_d = GraphCNN_P(pooling='MTP')
        self.gcn_p = GraphCNN_D(pooling='MTP')

        ###交叉注意力
        self.d_stru_gca = GuidedCrossAttention(embed_dim=2 * lstm_dim, num_heads=1)
        self.p_stru_gca = GuidedCrossAttention(embed_dim=2 * lstm_dim, num_heads=1)


        self.can_layer_emb = CAN_Layer(hidden_dim=256, num_heads_d=4, num_heads_p=4, group_size_d=1, group_size_p=1,
                                       agg_mode='mean_all_tok')

        ###降维
        self.lin_d1 = nn.Linear(512, 256)
        self.act_d = nn.GELU()
        self.d_norm = nn.LayerNorm(256)
        self.lin_d2 = nn.Linear(256, 256)

        self.p_adaptor_wo_skip_connect = FeedForwardLayer(1280, 512)
        self.lin_p1 = nn.Linear(1280, 512)
        self.act_p = nn.GELU()
        self.p_norm = nn.LayerNorm(512)
        self.lin_p2 = nn.Linear(512, 256)

    def forward(self, data, stu_d, stu_p, reset=False):
        batchsize = len(data.sm)
        smiles = torch.zeros(batchsize, seq_len).to(device).long()
        protein = torch.zeros(batchsize, tar_len).to(device).long()
        smiles_lengths = []
        protein_lengths = []

        smiles = data.smiles.to(device)
        protein = data.protein.to(device)
        smiles_lengths = data.smiles_lengths
        protein_lengths = data.protein_lengths

        smiles = smiles.view(batchsize, -1)
        protein = protein.view(batchsize, -1)


        smiles = self.smiles_embed(smiles)  # B * seq len * emb_dim
        smiles = self.smiles_input_fc(smiles)  # B * seq len * lstm_dim
        smiles = self.enhance1(smiles)

        protein = self.protein_embed(protein)  # B * tar_len * emb_dim
        protein = self.protein_input_fc(protein)  # B * tar_len * lstm_dim
        protein = self.enhance2(protein)

        # drugs and proteins BiLSTM
        smiles, _ = self.smiles_lstm(smiles)  # B * seq len * lstm_dim*2
        smiles = self.ln1(smiles)
        protein, _ = self.protein_lstm(protein)  # B * tar_len * lstm_dim *2
        protein = self.ln2(protein)

        smiles_mask = self.generate_masks(smiles, smiles_lengths, self.n_heads)  # B * head* seq len

        protein_mask = self.generate_masks(protein, protein_lengths, self.n_heads)  # B * head * tar_len

        prot, drug, mask_prot_grouped, mask_drug_grouped = self.can_layer_emb(protein, smiles, protein_mask,
                                                                              smiles_mask)
        mask_prot_grouped = mask_prot_grouped.float().unsqueeze(1).expand(-1, 8, -1)
        mask_drug_grouped = mask_drug_grouped.float().unsqueeze(1).expand(-1, 8, -1)
        smiles_out, smile_attn = self.out_attentions3(drug, mask_drug_grouped)  # B * lstm_dim*2
        protein_out, prot_attn = self.out_attentions2(prot, mask_prot_grouped)  # B * (lstm_dim *2)
        joint_emb = torch.cat([protein_out, smiles_out], dim=1)

        # print(joint_emb.shape)
        ####交叉注意
        x_aa = self.one_hot_embed(stu_p.native_x.long())
        x_aa = self.proj_aa(x_aa)

        ###stu_p
        stu_p.x = stu_p.x.to(torch.float32)
        x_esm = self.proj_esm(stu_p.x)
        x = F.relu(x_aa + x_esm)

        gcn_n_featp, gcn_g_featp = self.gcn_p(x, stu_p)


        xd = self.proj_uni(stu_d.x.float())

        gcn_n_featd, gcn_g_featd = self.gcn_d(xd, stu_d)

        joint_stu = torch.cat([gcn_g_featp, gcn_g_featd], dim=-1)

        out = torch.cat([joint_emb, joint_stu], dim=-1)  # B * (hidden_dim*4)
        # Point-wise Feed Forward Network
        pwff = self.pwff_1(out)
        pwff = nn.ReLU()(pwff)
        pwff = self.dropout(pwff)
        pwff = self.pwff_2(pwff)

        out = pwff + out

        out = self.dropout(self.relu(self.fusion_graph_seq(out)))  # B * (hidden_dim*2)

        out = self.out_fc3(out).squeeze()

        return out, 0

    def generate_masks(self, adj, adj_sizes, n_heads):
        out = torch.ones(adj.shape[0], adj.shape[1])
        max_size = adj.shape[1]
        if isinstance(adj_sizes, int):
            out[0, adj_sizes:max_size] = 0
        else:
            for e_id, drug_len in enumerate(adj_sizes):
                out[e_id, drug_len: max_size] = 0
        # out = out.unsqueeze(1).expand(-1, n_heads, -1)
        return out.cuda(device=adj.device)


def smiles_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    edge_index = np.array(edge_index)
    return c_size, edge_index


#############################################################################


df = pd.read_csv(f'./{dataset_name}_processed.csv')

smiles = set(df['compound_iso_smiles'])
target = set(df['target_key'])

target_seq = {}
for i in range(len(df)):
    target_seq[df.loc[i, 'target_key']] = df.loc[i, 'target_sequence']

smiles_graph = {}
for sm in smiles:
    _, graph = smiles_to_graph(sm)
    smiles_graph[sm] = graph

target_uniprot_dict = {}
target_process_start = {}
target_process_end = {}

for i in range(len(df)):
    target = df.loc[i, 'target_key']
    if dataset_name == 'kiba':
        uniprot = df.loc[i, 'target_key']
    else:
        uniprot = df.loc[i, 'uniprot']
    target_uniprot_dict[target] = uniprot
    target_process_start[target] = df.loc[i, 'target_sequence_start']
    target_process_end[target] = df.loc[i, 'target_sequence_end']

contact_dir = './target_contact_map_' + dataset_name + '/'
target_graph = {}


def target_to_graph(target_key, target_sequence, contact_dir, start, end):
    target_edge_index = []
    target_size = len(target_sequence)
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)
    contact_map = contact_map[start:end, start:end]
    index_row, index_col = np.where(contact_map > 0.8)

    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_edge_index = np.array(target_edge_index)
    return target_size, target_edge_index




drug_vocab = WordVocab.load_vocab('./Vocab/smiles_vocab.pkl')
target_vocab = WordVocab.load_vocab('./Vocab/protein_vocab.pkl')

tar_len = 1000
seq_len = 100

smiles_idx = {}
smiles_emb = {}
smiles_len = {}
for sm in smiles:
    content = []
    flag = 0
    for i in range(len(sm)):
        if flag >= len(sm):
            break
        if (flag + 1 < len(sm)):
            if drug_vocab.stoi.__contains__(sm[flag:flag + 2]):
                content.append(drug_vocab.stoi.get(sm[flag:flag + 2]))
                flag = flag + 2
                continue
        content.append(drug_vocab.stoi.get(sm[flag], drug_vocab.unk_index))
        flag = flag + 1

    if len(content) > seq_len - 2:
        content = content[:seq_len - 2]

    X = [drug_vocab.sos_index] + content + [drug_vocab.eos_index]
    smiles_len[sm] = len(content)
    if seq_len > len(X):
        padding = [drug_vocab.pad_index] * (seq_len - len(X))
        X.extend(padding)

    smiles_emb[sm] = torch.tensor(X)

    if not smiles_idx.__contains__(sm):
        tem = []
        for i, c in enumerate(X):
            if atom_dict.__contains__(c):
                tem.append(i)
        smiles_idx[sm] = tem

target_emb = {}
target_len = {}
for k in target_seq:
    seq = target_seq[k]
    content = []
    flag = 0
    for i in range(len(seq)):
        if flag >= len(seq):
            break
        if (flag + 1 < len(seq)):
            if target_vocab.stoi.__contains__(seq[flag:flag + 2]):
                content.append(target_vocab.stoi.get(seq[flag:flag + 2]))
                flag = flag + 2
                continue
        content.append(target_vocab.stoi.get(seq[flag], target_vocab.unk_index))
        flag = flag + 1

    # 在添加起始符和结束符之前，将 content 截断为 tar_len - 2
    if len(content) > tar_len - 2:
        content = content[:tar_len - 2]

    X = [target_vocab.sos_index] + content + [target_vocab.eos_index]
    target_len[seq] = len(content)
    if tar_len > len(X):
        padding = [target_vocab.pad_index] * (tar_len - len(X))
        X.extend(padding)
    target_emb[seq] = torch.tensor(X)

print("Building dataset...")
dataset = DTADataset(root='./', path='./' + dataset_name + '_processed.csv', smiles_emb=smiles_emb,
                     target_emb=target_emb, smiles_idx=smiles_idx, smiles_graph=smiles_graph, target_graph=target_graph,
                     smiles_len=smiles_len, target_len=target_len)




dataset_name = f"{dataset_name}_processed"
model_name = 'default'

model_file_name = './Model/' + dataset_name + '_' + model_name + '.pt'

load_model_path = model_file_name



print("Building model...")
model = PMHGT(embedding_dim=256, lstm_dim=128, hidden_dim=256, dropout_rate=0.2,
             alpha=0.2, n_heads=8, bilstm_layers=2, protein_vocab=26,
             smile_vocab=45, theta=0.5).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=1e-5, last_epoch=-1)



best_mse = 1000
best_test_mse = 1000
best_epoch = -1
best_test_epoch = -1

for epoch in range(NUM_EPOCHS):
    print("No {} epoch".format(epoch))
    if epoch == 0:
        test_size = (int)(len(dataset) * 0.1)
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset=dataset,
            lengths=[len(dataset) - (test_size * 2), test_size * 2],
            generator=torch.Generator().manual_seed(seed)
        )
        val_dataset, test_dataset = torch.utils.data.random_split(
            dataset=test_dataset,
            lengths=[test_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train(model, train_loader, optimizer, epoch)
    G, P = predicting(model, val_loader)
    val1 = get_mse(G, P)
    # # schedule.step(val1)
    if val1 < best_mse:
        best_mse = val1
        best_epoch = epoch + 1
        G, P = predicting(model, test_loader)
        val2 = get_mse(G, P)
        if model_file_name is not None:
            torch.save(model.state_dict(), model_file_name)
        print('mse improved at epoch ', best_epoch, '; best_mse', best_mse, 'test_mse', val2)
    else:
        print('current mse: ', val1, ' No improvement since epoch ', best_epoch, '; best_mse', best_mse, 'test_mse: ',
              val2)
    schedule.step()

print(model_file_name)
save_model = torch.load(model_file_name)
model_dict = model.state_dict()
state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
model_dict.update(state_dict)
model.load_state_dict(model_dict)
G, P = predicting(model, test_loader)
cindex, rm2, mse, pearson, spearman = calculate_metrics_and_return(G, P, test_loader)
print(f"CI: {cindex:.4f}, RM2: {rm2:.4f}, MSE: {mse:.4f}, Pearson: {pearson:.4f}, Spearman: {spearman:.4f}\n")