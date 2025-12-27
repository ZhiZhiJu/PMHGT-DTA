# PMHGT-DTA
## 📦 Davis and KIBA PDB File
The PDB files for the Davis and KIBA datasets are available for download in the following ways:

1. Download via link: [zhaolongNCU/PocketDTA](https://github.com/zhaolongNCU/PocketDTA)

2. Run the download script directly:

   - For Davis dataset: `download_pdb_davis.py`

   - For KIBA dataset: `download_pdb_kiba.py`



## 🚀 Running Steps

### Step 1: Obtain protein graphs

Run the script:

```bash

python protein_graph.py

```



### Step 2: Obtain drug graphs

Run the preprocessing script:

```bash

python molecular_graph_preprocessing.py

```



### Step 3: Run the core program

Execute the main script:

```bash

python main.py

```

```
