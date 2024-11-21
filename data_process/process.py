import os

import numpy as np
import scipy.io as sio
import torch
from sklearn.utils import shuffle

EDGE_LIFE = False
EDGE_LIFE_WINDOW = 10
MAKE_SYMMETRIC = False


def func_make_symmetric(sparse_tensor, N, TS):
    count = 0
    tensor_idx = torch.LongTensor([])
    tensor_val = torch.DoubleTensor([]).unsqueeze(1)
    A_idx = sparse_tensor._indices()
    A_val = sparse_tensor._values()
    for i in range(TS):
        """每一个时间切片的邻接矩阵对称化"""
        idx = A_idx[0] == i
        mat = torch.sparse.DoubleTensor(A_idx[1:3, idx], A_val[idx], torch.Size([N, N]))
        mat_t = mat.transpose(1, 0)
        sym_mat = (mat + mat_t)
        count = count + sym_mat._nnz()
        vertices = sym_mat._indices().clone().detach()
        time = torch.ones(sym_mat._nnz(), dtype=torch.long) * i
        time = time.unsqueeze(0)
        full = torch.cat((time, vertices), 0)  # 将时间切片和对应邻接矩阵拼接成三维下标（垂直）
        tensor_idx = torch.cat((tensor_idx, full), 1)  # 拼接三维稀疏邻接矩阵下标（水平）
        tensor_val = torch.cat((tensor_val, sym_mat._values().unsqueeze(1)), 0)
    tensor_val.squeeze_(1)
    A = torch.sparse.DoubleTensor(tensor_idx, tensor_val, torch.Size([TS, N, N])).coalesce()
    return A


def func_edge_life(A, N, TS, edge_life_window):
    """前l-1个切片的邻接矩阵添加到当前切片之中"""
    A_new = A.clone()
    A_new._values()[:] = 0
    for t in range(TS):
        idx = (A._indices()[0] >= max(0, t - edge_life_window + 1)) & (A._indices()[0] <= t)
        block = torch.sparse_coo_tensor(A._indices()[0:3, idx], A._values()[idx], torch.Size([TS, N, N]), dtype=torch.double)
        block._indices()[0] = t
        A_new = A_new + block
    return A_new.coalesce()


def func_laplacian_transformation(B, N, TS):
    vertices = torch.LongTensor([range(N), range(N)])
    tensor_idx = torch.LongTensor([])
    tensor_val = torch.DoubleTensor([]).unsqueeze(1)
    for i in range(TS):
        time = torch.ones(N, dtype=torch.long) * i
        time = time.unsqueeze(0)
        full = torch.cat((time, vertices), 0)
        tensor_idx = torch.cat((tensor_idx, full), 1)
        val = torch.ones(N, dtype=torch.double)
        tensor_val = torch.cat((tensor_val, val.unsqueeze(1)), 0)
    tensor_val.squeeze_(1)
    I = torch.sparse_coo_tensor(tensor_idx, tensor_val, torch.Size([TS, N, N]), dtype=torch.double)
    C = B + I
    tensor_idx = torch.LongTensor([])
    tensor_val = torch.DoubleTensor([]).unsqueeze(1)
    for k in range(TS):
        idx = C._indices()[0] == k
        mat = torch.sparse_coo_tensor(C._indices()[1:3, idx], C._values()[idx], torch.Size([N, N]), dtype=torch.double)
        vec = torch.ones([N, 1], dtype=torch.double)
        degree = 1 / torch.sqrt(torch.sparse.mm(mat, vec))
        index = torch.LongTensor(C._indices()[0:3, idx].size())
        index[0] = k
        index[1:3] = mat._indices()
        values = mat._values()
        count = 0
        for i, j in index[1:3].transpose(1, 0):
            values[count] = values[count] * degree[i] * degree[j]
            count = count + 1
        tensor_idx = torch.cat((tensor_idx, index), 1)
        tensor_val = torch.cat((tensor_val, values.unsqueeze(1)), 0)
    tensor_val.squeeze_(1)
    C = torch.sparse_coo_tensor(tensor_idx, tensor_val, torch.Size([TS, N, N]), dtype=torch.double)
    return C.coalesce()


def get_random_idx(num_all, num_ones, random_state=2024):
    no_true = np.ones(num_ones) == 1
    no_false = np.zeros(num_all - num_ones) == 1
    temp = list(no_true) + list(no_false)
    idx = shuffle(torch.tensor(temp), random_state=random_state)
    return idx


def get_dataset(A, idx):
    sz = A.size()
    not_idx = idx == False

    index = torch.LongTensor(A._indices()[0:3, idx].size())  # 传入的是size
    index[0:3] = A._indices()[0:3, idx]
    values = A._values()[idx]
    sub = torch.sparse_coo_tensor(index, values, sz)

    remain_index = torch.LongTensor(A._indices()[0:3, not_idx].size())  # 传入的是size
    remain_index[0:3] = A._indices()[0:3, not_idx]
    remain_values = A._values()[not_idx]
    remain = torch.sparse_coo_tensor(remain_index, remain_values, sz)

    return remain.coalesce(), sub.coalesce()


def split_data(A, N, T, start, end):
    assert (end - start) == T
    idx = (A._indices()[0] >= start) & (A._indices()[0] < end)
    index = torch.LongTensor(A._indices()[0:3, idx].size())
    index[0:3] = A._indices()[0:3, idx]
    index[0] = index[0] - start
    values = A._values()[idx]
    sub = torch.sparse_coo_tensor(index, values, torch.Size([T, N, N]), dtype=torch.double)
    return sub.coalesce()


def get_node_to_index(edge):
    edge = edge.tolist()
    # 确定所有唯一的SRC和DST节点
    unique_nodes = list(set([src for src, _ in edge] + [dst for _, dst in edge]))
    num_nodes = len(unique_nodes)
    # 将节点映射到索引
    node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}
    return node_to_index, num_nodes


def get_adj_idx(node_to_index, edge):
    edge = edge.tolist()
    # 收集行索引和列索引
    rows, cols = [], []
    for src, dst in edge:
        rows.append(node_to_index[src])
        cols.append(node_to_index[dst])

    # 将行索引、列索引和值转换为PyTorch张量
    rows_tensor = torch.tensor(rows, dtype=torch.long)
    cols_tensor = torch.tensor(cols, dtype=torch.long)

    return torch.stack((rows_tensor, cols_tensor), dim=1)


def save_file(tensor_idx, tensor_labels, A, A_train, A_val, A_test, train, val, test, path, file_name):
    A_idx = A._indices()
    A_vals = A._values()

    A_train_idx = A_train._indices()
    A_train_vals = A_train._values()

    A_val_idx = A_val._indices()
    A_val_vals = A_val._values()

    A_test_idx = A_test._indices()
    A_test_vals = A_test._values()

    train_idx = train._indices()
    train_vals = train._values()

    val_idx = val._indices()
    val_vals = val._values()

    test_idx = test._indices()
    test_vals = test._values()

    sio.savemat(os.path.join(path, file_name), {
        'tensor_idx': np.array(tensor_idx),
        'tensor_labels': np.array(tensor_labels),

        'A_idx': np.array(A_idx),
        'A_vals': np.array(A_vals),

        'A_train_idx': np.array(A_train_idx),
        'A_train_vals': np.array(A_train_vals),

        'A_val_idx': np.array(A_val_idx),
        'A_val_vals': np.array(A_val_vals),

        'A_test_idx': np.array(A_test_idx),
        'A_test_vals': np.array(A_test_vals),

        'train_idx': np.array(train_idx),
        'train_vals': np.array(train_vals),

        'test_idx': np.array(test_idx),
        'test_vals': np.array(test_vals),

        'val_idx': np.array(val_idx),
        'val_vals': np.array(val_vals),
    })


def split_data_link(TS, val_rate, test_rate, A, N):
    val_samples = int(TS * val_rate)
    test_samples = int(TS * test_rate)
    T = TS - val_samples - test_samples

    # 分割数据
    A_train = split_data(A, N, T, 0, T)
    A_val = split_data(A, N, T, val_samples, T + val_samples)
    A_test = split_data(A, N, T, val_samples + test_samples, TS)

    return A_train, A_val, A_test, T


def pre_process(A_train, A_val, A_test, N, T):
    print('make_sym...')
    if MAKE_SYMMETRIC:
        A_train_sym = func_make_symmetric(A_train, N, T)
        A_val_sym = func_make_symmetric(A_val, N, T)
        A_test_sym = func_make_symmetric(A_test, N, T)
    else:
        A_train_sym = A_train
        A_val_sym = A_val
        A_test_sym = A_test

    print('edge_life...')
    if EDGE_LIFE:
        A_train_sym_life = func_edge_life(A_train_sym, N, T, EDGE_LIFE_WINDOW)
        A_val_sym_life = func_edge_life(A_val_sym, N, T, EDGE_LIFE_WINDOW)
        A_test_sym_life = func_edge_life(A_test_sym, N, T, EDGE_LIFE_WINDOW)

    else:
        A_train_sym_life = A_train_sym
        A_val_sym_life = A_val_sym
        A_test_sym_life = A_test_sym

    print('func_laplacian_trans...')
    A_train_sym_life_la = func_laplacian_transformation(A_train_sym_life, N, T)
    A_val_sym_life_la = func_laplacian_transformation(A_val_sym_life, N, T)
    A_test_sym_life_la = func_laplacian_transformation(A_test_sym_life, N, T)

    train = A_train_sym_life_la
    val = A_val_sym_life_la
    test = A_test_sym_life_la

    return train, val, test


def process_tab(filename, save_filename):
    with open(filename, 'r') as file:
        content = file.read()

    # 替换所有制表符为一个空格
    processed_content = content.replace('\t', ' ')

    # 打印或写入处理后的内容
    print(processed_content)
    # 或者写回文件
    with open(save_filename, 'w') as file:
        file.write(processed_content)
