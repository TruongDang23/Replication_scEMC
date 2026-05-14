import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import h5py
import warnings
warnings.filterwarnings("ignore")
import scanpy as sc
from scipy import sparse

ALL_data = dict(
    #
    SNARE = {
            1: 'SNARE', 
            2: 'd1', 
            'files': ['RNA.h5ad', 'ATAC.h5ad'],
            'label_key': 'cell_type',
            'N': 30672, 
            'K': 27, 
            'V': 2, 
            'n_input': [1000,25], 
            'n_hid': [10,256], 
            'n_output': 64
        },
    )


path = '/content/Replication_scEMC/scEMC/datasets'

def load_data(dataset_info, path=path, top_genes=2000, top_peaks=2000):

    X = []
    labels_list = []

    for file_name in dataset_info['files']:

        print(f"Loading {file_name}")

        adata = sc.read_h5ad(
            path + '/' + dataset_info[1] + '/' + file_name
        )

        # ================= RNA =================
        if "RNA" in file_name.upper():

            print(f"Selecting top {top_genes} HVGs")

            # KHÔNG copy
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=top_genes,
                flavor="seurat_v3",
                subset=False
            )

            hvg_mask = adata.var["highly_variable"].values

            # Slice trực tiếp
            adata = adata[:, hvg_mask]

            # normalize sau khi giảm chiều
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        # ================= ATAC / Protein =================
        elif "ATAC" in file_name.upper():

            print(f"Selecting top {top_peaks} peaks")

            counts = np.asarray(adata.X.sum(axis=0)).ravel()

            top_indices = np.argpartition(counts, -top_peaks)[-top_peaks:]

            adata = adata[:, top_indices]

        # ================= Sparse -> Dense =================

        # Chỉ convert sau khi đã giảm còn 2000 features
        if sparse.issparse(adata.X):
            data_view = adata.X.astype(np.float32).toarray()
        else:
            data_view = adata.X.astype(np.float32)

        X.append(torch.from_numpy(data_view))

        # ================= Labels =================

        if dataset_info['label_key'] in adata.obs:

            labels = (
                adata.obs[dataset_info['label_key']]
                .astype('category')
                .cat.codes
                .values
            )

            labels_list.append(labels)

        # giải phóng RAM sớm
        del adata
        del data_view

    Label = labels_list[0]

    size = X[0].shape[0]
    view_num = len(X)

    index = np.arange(size)
    np.random.shuffle(index)

    Y = []

    for v in range(view_num):
        X[v] = X[v][index]
        Y.append(Label[index])

    return X, Y