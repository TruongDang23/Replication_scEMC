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
            'export_label_key': 'Group',   # cột tên cell type dùng làm y_true khi export .npz
            'N': 30672,
            'K': 27, 
            'V': 2, 
            'n_input': [1000,25], 
            'n_hid': [10,256], 
            'n_output': 64
        },
    )


path = '/content/Replication_scEMC/scEMC/datasets'

def load_data(dataset_info, path=path, top_genes=2000, top_peaks=2000, return_meta=False):

    X = []
    labels_list = []
    obs_names_list = []   # barcode gốc của từng file, chỉ dùng để export .npz
    cell_types = None     # tên cell type (str), chỉ dùng để export .npz

    for file_name in dataset_info['files']:

        print(f"Loading {file_name}")

        adata = sc.read_h5ad(
            path + '/' + dataset_info[1] + '/' + file_name
        )

        # ================= RNA =================
        if "RNA" in file_name.upper():

            # ── 2. Bộ lọc Đặc trưng cho scRNA-seq (HVG) ──────────────────────────────
            if 'highly_variable' in adata.var.columns:
                print(f"[RNA] Tìm thấy cột highly_variable sẵn có.")
                if adata.var['highly_variable'].sum() > top_genes and 'dispersions_norm' in adata.var.columns:
                    hvg_subset = adata.var[adata.var['highly_variable'] == True]
                    top_genes = hvg_subset.nlargest(top_genes, 'dispersions_norm').index
                    adata = adata[:, top_genes].copy()
                else:
                    # Nếu không có cột dispersion hoặc số lượng vừa bằng, lấy các gene được đánh dấu True
                    # Nếu số lượng gene lớn hơn top_n_rna mà không có dispersion, lấy top_n_rna gene đầu tiên của tập True
                    true_genes = adata.var_names[adata.var['highly_variable'] == True]
                    adata = adata[:, true_genes[:top_genes]].copy()
            else:
                print(f"[Warning] Không thấy nhãn highly_variable trong RNA, tự động lấy {top_genes} gen đầu.")
                adata = adata[:, :top_genes].copy()
            
            print(f"[RNA] Sau khi lọc lấy đặc trưng biến thiên: {adata.shape}")

        # ================= ATAC / Protein =================
        elif "ATAC" in file_name.upper():

            # ── 3. Bộ lọc Đặc trưng cho scATAC-seq (Sửa đổi cho chuẩn Raw Count) ────
            # Vì file ATAC.h5ad của bạn hiện tại có 2,016 features, chúng ta cần ép nó về đúng 2,000 (top_n_atac)
            if 'highly_variable' in adata.var.columns:
                print(f"[ATAC] Tìm thấy cột highly_variable từ bước xử lý phương sai nhị phân.")
                
                # Sắp xếp và lọc lấy đúng top_n_atac dựa trên cột phương sai thủ công 'binary_variance' mà ta đã lưu
                if 'binary_variance' in adata.var.columns:
                    hvp_subset = adata.var[adata.var['highly_variable'] == True]
                    top_peaks = hvp_subset.nlargest(top_peaks, 'binary_variance').index
                    adata = adata[:, top_peaks].copy()
                else:
                    # Cắt lấy đúng số lượng cột yêu cầu để tránh lệch chiều mạng Neural
                    true_peaks = adata.var_names[adata.var['highly_variable'] == True]
                    adata = adata[:, true_peaks[:top_peaks]].copy()
            else:
                # Nhánh dự phòng an toàn bằng phương sai nhị phân (chứ không dùng tổng sum nữa)
                print(f"[Warning] Không tìm thấy cột highly_variable trong ATAC. Tính lại phương sai nhị phân...")
                X_binary = (adata.X > 0).astype(np.float32)
                p = np.array(X_binary.mean(axis=0)).flatten()
                variances = p * (1.0 - p)
                top_indices = np.argpartition(variances, -top_peaks)[-top_peaks:]
                adata = adata[:, top_indices].copy()
                
            print(f"[ATAC] Sau khi lọc lấy đặc trưng biến thiên: {adata.shape}")

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

        # ================= Barcode + tên cell type (export .npz) =================

        obs_names_list.append(adata.obs_names.to_numpy().astype(str))
        if cell_types is None:
            name_key = dataset_info.get('export_label_key', 'Group')
            if name_key not in adata.obs:
                print(f"[Warning] Không có cột '{name_key}' trong {file_name}, "
                      f"dùng '{dataset_info['label_key']}' làm y_true khi export .npz.")
                name_key = dataset_info['label_key']
            if name_key in adata.obs:
                cell_types = adata.obs[name_key].astype(str).to_numpy()

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

    if not return_meta:
        return X, Y

    # Các view được ghép theo vị trí hàng → barcode phải trùng khớp giữa các file
    for v in range(1, view_num):
        if not np.array_equal(obs_names_list[0], obs_names_list[v]):
            print(f"[Warning] obs_names của {dataset_info['files'][v]} khác {dataset_info['files'][0]} "
                  f"(khác tập cell hoặc khác thứ tự) — các view đang được ghép theo vị trí hàng.")
    if cell_types is None:
        raise ValueError("Không tìm thấy cột tên cell type để làm y_true khi export .npz.")

    # Shuffle cùng `index` với X, Y để giữ đúng thứ tự tế bào
    meta = {
        'cell_ids': obs_names_list[0][index],
        'cell_types': cell_types[index],
    }
    return X, Y, meta