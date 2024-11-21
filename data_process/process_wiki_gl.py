import math

from constant import VAL_RATE, TEST_RATE, WIKI_TS
from data_process.process import *

# Settings
dataset = 'wiki_gl'
print(dataset)

# ia_contacts setting
DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), '../data')
PATH = os.path.join(DATA_DIR_PATH, 'wiki_gl')
FILE_NAME = 'wiki_gl.mat'

TIME_DIM = 3

EDGE_LIFE = False
EDGE_LIFE_WINDOW = 10
MAKE_SYMMETRIC = False


def preprocess_data():
    data = np.loadtxt(os.path.join(PATH, 'wiki_gl.txt'), delimiter=' ', skiprows=1)
    data = data[data[:, 0] != data[:, 1]]  # 去除自环

    max_time = max(data[:, TIME_DIM])
    min_time = min(data[:, TIME_DIM])

    TS = WIKI_TS
    time_delta = math.floor((max_time - min_time) / TS)

    data = torch.tensor(data)

    data_idx = data[:, TIME_DIM] <= min_time + time_delta * TS
    data = data[data_idx]

    node_to_index, N = get_node_to_index(data[:, 0:2])

    tensor_idx = torch.zeros([data.size()[0], 3], dtype=torch.long)
    tensor_val = torch.ones([data.size()[0]], dtype=torch.double)
    tensor_labels = torch.ones([data.size()[0]], dtype=torch.double)

    start = min_time
    for t in range(TS):
        end = start + time_delta
        if t == TS - 1:
            idx = (data[:, TIME_DIM] >= start) & (data[:, TIME_DIM] <= end)
        else:
            idx = (data[:, TIME_DIM] >= start) & (data[:, TIME_DIM] < end)
        start = end
        tensor_idx[idx, 1:3] = get_adj_idx(node_to_index, data[idx, 0:2])
        tensor_idx[idx, 0] = t

    A = torch.sparse.DoubleTensor(tensor_idx.transpose(1, 0), tensor_val, torch.Size([TS, N, N])).coalesce()
    A = torch.sparse.DoubleTensor(A._indices(), torch.ones(A._values().shape), torch.Size([TS, N, N]))
    labels_weight = torch.sparse.DoubleTensor(tensor_idx.transpose(1, 0), tensor_labels, torch.Size([TS, N, N])).coalesce()

    val_samples = int(TS * VAL_RATE)
    test_samples = int(TS * TEST_RATE)
    T = TS - val_samples - test_samples

    # 分割数据
    A_train = split_data(A, N, T, 0, T)
    A_val = split_data(A, N, T, val_samples, T + val_samples)
    A_test = split_data(A, N, T, val_samples + test_samples, TS)

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

    print('store...')
    save_file(tensor_idx, tensor_labels, A, A_train, A_val, A_test, train, val, test, PATH, FILE_NAME)


preprocess_data()
