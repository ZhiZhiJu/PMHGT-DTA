import os
import numpy as np
import pandas as pd
from Bio.PDB.PDBParser import PDBParser
import torch
from torch_geometric.data import Data
from util import protein_graph
import esm


device = 'cuda:0'
esm_model_path = r"D:\Anew_Study\2025\code\PocketDTA-main\model\esm2_t33_650M_UR50D.pt"

# 加载 ESM 模型
esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(esm_model_path)
batch_converter = alphabet.get_batch_converter()
ckpt = torch.load('D:\Anew_Study\\2025\code-language\ism_t33_650M_uc30pdb\checkpoint.pth')
esm_model.load_state_dict(ckpt)
esm_model = esm_model.to(device)
esm_model.eval()

# 氨基酸残基类型转换字典
restype_1to3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}
restype_3to1 = {v: k for k, v in restype_1to3.items()}

# 读取 CSV 文件
csv_file_path = "D:\Anew_Study\\2025\code\MyDTA\DMFF-DTA-main-davis\\binding_range\davis_processed_updated.csv"
data = pd.read_csv(csv_file_path)

# PDB 文件所在目录
pdb_dir = r"D:\Anew_Study\2025\code\PocketDTA-main\dataset\Davis\pdb"

# 定义存储图数据的目录
output_dir = '../data/davis_GDTA/graphs_DMFF_800'
os.makedirs(output_dir, exist_ok=True)

# 遍历 CSV 文件中的每一行
for index, row in data.iterrows():


    uniprot = row['uniprot']
    sequence = ''
    pdb_file_path = os.path.join(pdb_dir, f'{uniprot}.pdb')

    if not os.path.exists(pdb_file_path):
        print(f"PDB file {pdb_file_path} not found. Skipping...")
        continue

        # 保存每个蛋白质的图数据到单独的 .pt 文件
    output_file_path = os.path.join(output_dir, f"{uniprot}.pt")

    # 仅在文件不存在时保存
    if os.path.exists(output_file_path):
        continue
    # 解析 PDB 文件
    parser = PDBParser()
    try:
        struct = parser.get_structure("x", pdb_file_path)
        model = struct[0]
        chain_id = list(model.child_dict.keys())[0]
        chain = model[chain_id]
        Ca_array = []
        seq_idx_list = list(chain.child_dict.keys())

        for idx in range(seq_idx_list[0][1], seq_idx_list[-1][1] + 1):
            try:
                Ca_array.append(chain[(' ', idx, ' ')]['CA'].get_coord())
            except:
                Ca_array.append([np.nan, np.nan, np.nan])
            try:
                sequence += restype_3to1[chain[(' ', idx, ' ')].get_resname()]
            except:
                sequence += 'X'

        Ca_array = np.array(Ca_array)
        start = int(row['target_sequence_start'])
        end = int(row['target_sequence_end'])
        if sequence != row['Target']:
            print(uniprot)
            print(sequence)
            print(row['Target'])
        if end - start > 1024:
            end = start + 1024
        # Ca_array = Ca_array[start:end]
        sequence = sequence[start:end]

        # file_path = fr"D:\Anew_Study\2025\data\kiba\protein_ism\{uniprot}.npy"
        # 计算距离矩阵
        resi_num = Ca_array.shape[0]
        G = np.dot(Ca_array, Ca_array.T)
        H = np.tile(np.diag(G), (resi_num, 1))
        dismap = (H + H.T - 2 * G) ** 0.5
        dismap = dismap[start:end, start:end]

        # 使用 ESM 模型获取序列嵌入
        batch_labels, batch_strs, batch_tokens = batch_converter([('tmp', sequence)])
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33][0].cpu().numpy()
            esm_embed = token_representations[1:len(sequence) + 1]

        # embed = np.load(file_path)[start:end, :]
        # 构建蛋白质图
        row, col = np.where(dismap <= 8)
        edge = [row, col]
        graph = protein_graph(sequence, edge, esm_embed)

        # 保存每个蛋白质的图数据到单独的 .pt 文件
        output_file_path = os.path.join(output_dir, f"{uniprot}.pt")

        # 仅在文件不存在时保存
        if not os.path.exists(output_file_path):
            torch.save(graph, output_file_path)
            print(f"Saved: {output_file_path}")
        # else:
        #     print(f"File already exists, skipping: {output_file_path}")

        # print(f"Graph for {uniprot} saved to {output_file_path}")

    except Exception as e:
        print(f"Error processing {uniprot}: {e}")

print(f"All graphs saved to {output_dir}")
