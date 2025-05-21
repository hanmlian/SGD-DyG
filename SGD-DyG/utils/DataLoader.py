import random

import numpy as np
import scipy.io as sio
import torch

from ..constant import *
import util


def func_get_floatTensor(saved_content, tensor_name, size, device):
    indices = torch.tensor(np.array(saved_content[tensor_name + '_idx'], dtype=int), dtype=torch.long, device=device)
    values = torch.tensor(saved_content[tensor_name + '_vals'], dtype=torch.float, device=device)
    return torch.sparse_coo_tensor(indices, torch.squeeze(values), size, dtype=torch.float, device=device)


def load_data(args, device):
    util.set_seed(args.seed)
    dataset_name = args.dataset_name
    if dataset_name == DatasetType.BITCOIN_ALPHA.name.lower():
        mat_path = os.path.join(BITCOIN_ALPHA_PATH, BITCOIN_ALPHA_NAME)
        TS = BITCOIN_ALPHA_TS
    elif dataset_name == DatasetType.BITCOIN_OTC.name.lower():
        mat_path = os.path.join(BITCOIN_OTC_PATH, BITCOIN_OTC_NAME)
        TS = BITCOIN_OTC_TS
    elif dataset_name == DatasetType.WIKI_GL.name.lower():
        mat_path = os.path.join(WIKI_GL_PATH, WIKI_GL_NAME)
        TS = WIKI_GL_TS
    elif dataset_name == DatasetType.DIGG.name.lower():
        mat_path = os.path.join(DIGG_PATH, DIGG_NAME)
        TS = DIGG_TS
    elif dataset_name == DatasetType.WIKI_EO.name.lower():
        mat_path = os.path.join(WIKI_EO_PATH, WIKI_EO_NAME)
        TS = WIKI_EO_TS
    else:
        raise Exception('Invalid DataSet')

    val_steps = int(TS * VAL_RATE)
    test_steps = int(TS * TEST_RATE)
    train_steps = TS - val_steps - test_steps

    # Load stuff from mat file
    saved_content = sio.loadmat(mat_path)
    T = np.max(saved_content["tensor_idx"][:, 0]) + 1
    N1 = np.max(saved_content["tensor_idx"][:, 1]) + 1
    N2 = np.max(saved_content["tensor_idx"][:, 2]) + 1
    N = max(N1, N2)
    A_sz = torch.Size([T, N, N])
    train_sz = torch.Size([train_steps, N, N])

    A = func_get_floatTensor(saved_content, 'A', A_sz, device)
    labels, neg_adj = negative_sample(A, BETA, device)

    train = func_get_floatTensor(saved_content, 'train', train_sz, device)
    val = func_get_floatTensor(saved_content, 'val', train_sz, device)
    test = func_get_floatTensor(saved_content, 'test', train_sz, device)

    A_train = []
    for i in range(train_steps):
        idx = train._indices()[0] == i
        A_train.append(torch.sparse_coo_tensor(train._indices()[1:3, idx], train._values()[idx]))
    A_val, neg_adj_val = [], []
    for i in range(train_steps):
        idx = val._indices()[0] == i
        A_val.append(torch.sparse_coo_tensor(val._indices()[1:3, idx], val._values()[idx]))
    A_test, neg_adj_test = [], []
    for i in range(train_steps):
        idx = test._indices()[0] == i
        A_test.append(torch.sparse_coo_tensor(test._indices()[1:3, idx], test._values()[idx]))

    return TS, labels, A_train, A_val, A_test, N


def negative_sample(A, beta, device):
    tensor_idx = torch.zeros([3, A._nnz() + A._nnz() * beta], dtype=torch.long, device=device)
    tensor_val = torch.zeros([A._nnz() + A._nnz() * beta], dtype=torch.float, device=device)
    next_idx = 0

    T, N, _ = A.size()
    neg_adj = []
    for i in range(T):
        num_samples = int(A[i]._nnz() * beta)
        if num_samples == 0:
            continue
        edge_index = A[i]._indices().to(device)
        negative_samples = generate_negative_samples(N, edge_index, num_samples)
        negative_samples = torch.tensor(negative_samples, dtype=torch.long, device=device).t()
        values = torch.zeros(num_samples, dtype=torch.float, device=device)
        neg_A = torch.sparse_coo_tensor(
            negative_samples,
            values,
            torch.Size([N, N]),
            dtype=torch.double
        ).to(device)
        neg_adj.append(neg_A)

        combined_A = A[i].to(device) + neg_A
        num = combined_A._nnz()
        tensor_idx[1:3, next_idx:next_idx + num] = combined_A._indices()
        tensor_idx[0, next_idx:next_idx + num] = i
        tensor_val[next_idx:next_idx + num] = combined_A._values()

        next_idx += num

    return torch.sparse_coo_tensor(tensor_idx, tensor_val, torch.Size([T, N, N]), device=device), neg_adj


def generate_negative_samples(num_nodes, edge_index, num_samples=None):
    existing_edges = set((edge_index[0][i].item(), edge_index[1][i].item()) for i in range(edge_index.shape[1]))

    if num_samples is None:
        num_samples = edge_index.shape[1]

    negative_samples = set()
    while len(negative_samples) < num_samples:
        u, v = random.sample(range(num_nodes), 2)
        if (u, v) not in existing_edges and (v, u) not in existing_edges and u != v:
            negative_samples.add((u, v))

    return list(negative_samples)
