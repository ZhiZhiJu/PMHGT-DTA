import os
import requests
import pandas as pd


def download_pdb_file(pdb_id, output_folder):
    url = f"https://alphafold.ebi.ac.uk/files/AF-{pdb_id}-F1-model_v4.pdb"
    output_file = os.path.join(output_folder, f"{pdb_id}.pdb")

    # 检查文件是否已经存在
    if os.path.exists(output_file):
        return

    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_file, 'w') as file:
                file.write(response.text)
            print(f"Successfully downloaded {pdb_id}")
        else:
            print(f"Failed to download PDB file for {pdb_id}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {pdb_id}: {e}")


output_folder = '../data/davis/PDB_AF2'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

file_path = '../data/davis/Davis_protein_mapping.csv'
data = pd.read_csv(file_path)
protein_ids = data['uniprot']

print(f"Starting download of {len(protein_ids)} PDB files...")
for pdb_id in protein_ids:
    download_pdb_file(pdb_id, output_folder)
print("Download process completed.")