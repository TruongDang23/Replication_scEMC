import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import h5py
import warnings
warnings.filterwarnings("ignore")
import scanpy as sc

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
    mm = MinMaxScaler()
    
    # 1. Đọc từng file h5ad
    for file_name in dataset_info['files']:
        # Đọc file bằng scanpy
        adata = sc.read_h5ad(path + '/' + dataset_info[1] + '/' + file_name)
        # 2. Lọc Top 2000 Features TRƯỚC KHI chuyển sang ma trận đặc
        if "RNA" in file_name.upper():
            print(f"Processing {file_name}: Filtering HVGs...")
            sc.pp.highly_variable_genes(adata, n_top_genes=top_genes, flavor="seurat_v3", subset=True)
            # Giữ lại 2000 gene biến thiên nhất
            adata = adata[:, adata.var.highly_variable].copy()
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            
        elif "ATAC" in file_name.upper():
            print(f"Processing {file_name}: Filtering Top Peaks...")
            if hasattr(adata.X, 'sum'):
                counts = np.array(adata.X.sum(axis=0)).flatten()
            else:
                counts = np.sum(adata.X, axis=0)
            
            # Lấy index của 2000 peaks lớn nhất
            top_indices = np.argsort(counts)[-top_peaks:]
            adata = adata[:, top_indices].copy()
        # Lấy ma trận dữ liệu (thường là adata.X)
        # Nếu adata.X là sparse matrix, cần .toarray()
        data_view = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

        # Chuẩn hóa
        # std_view = mm.fit_transform(data_view)
        X.append(data_view)
        
        # Lấy label từ field cell_types (chuyển sang dạng số nếu là dạng chuỗi)
        if dataset_info['label_key'] in adata.obs:
            # Chuyển category sang mã số (0, 1, 2...) để training
            labels = adata.obs[dataset_info['label_key']].astype('category').cat.codes.values
            labels_list.append(labels)
    
    # Giả sử Label của các view là giống nhau cho cùng 1 tập tế bào
    # Chúng ta lấy label của view đầu tiên làm chuẩn
    Label = labels_list[0]
    
    size = X[0].shape[0]
    view_num = len(X)
    
    # 2. Shuffle (Giữ nguyên logic của scEMC)
    index = np.arange(size)
    np.random.shuffle(index)
    
    Y = []
    for v in range(view_num):
        X[v] = torch.from_numpy(X[v][index].astype(np.float32))
        Y.append(Label[index]) 

    return X, Y