import torch
from torch import nn
from torch.nn import Module, Parameter


class TensorGraphConvolution(Module):
    def __init__(self, time_slices, in_features, out_features, band_width, tgc_dropout=0.6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.time_slices = time_slices
        self.band_width = band_width
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # self.weight = Parameter(torch.FloatTensor(time_slices, in_features, out_features))
        self.dropout = nn.Dropout(tgc_dropout)
        self.reset_parameters()

    def compute_AtXt(self, A, X, M):
        At = self.func_MProduct(A, M, self.time_slices)
        Xt = torch.matmul(M, X.reshape(self.time_slices, -1)).reshape(X.size())
        N = X.size()[1]
        AtXt = torch.zeros(self.time_slices, N, X.size()[-1], device=M.device)
        for k in range(self.time_slices):
            AtXt[k] = torch.sparse.mm(At[k], Xt[k])
        return AtXt

    def forward(self, adj, input, M):
        h = self.compute_AtXt(adj, input, M)
        output = torch.matmul(self.dropout(h), self.weight)
        return output

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def func_MProduct(self, input, M, time_slices):
        len_M = self.band_width
        device = M.device
        if type(input) == torch.Tensor:
            out = torch.zeros(time_slices, time_slices, device=device)
            for tm in range(time_slices):
                if tm < len_M:
                    out[tm, 0:tm + 1] = M[tm, 0:tm + 1]
                else:
                    out[tm, tm - len_M + 1: tm + 1] = M[tm, tm - len_M + 1: tm + 1]
            output = torch.matmul(out, input.reshape(time_slices, -1))
            return output.reshape(input.shape)
        elif type(input) == list:
            N = input[0].size()[-1]
            sz = torch.Size([N, N])
            res = []
            for tm in range(time_slices):
                temp = torch.sparse_coo_tensor(sz, device=device)
                if tm < len_M:
                    m = M[tm, 0:tm + 1]
                    for i in range(tm + 1):
                        A = input[i]
                        val = m[i] * A._values()
                        temp = temp + torch.sparse_coo_tensor(A._indices(), val, sz, device=device)
                else:
                    m = M[tm, tm - len_M + 1: tm + 1]
                    for i in range(len_M):
                        A = input[tm - len_M + i]
                        val = m[i] * A._values()
                        temp = temp + torch.sparse_coo_tensor(A._indices(), val, sz, device=device)
                res.append(temp.coalesce())
            return res


class FFTLayer(Module):
    def __init__(self, time_slices, in_features, out_features, fft_dropout):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.time_slices = time_slices
        # self.weight = Parameter(torch.empty(time_slices, in_features, out_features, dtype=torch.complex64))
        self.weight = Parameter(torch.empty(in_features, out_features, dtype=torch.complex64))
        self.fft_dropout = nn.Dropout(fft_dropout)
        self.activation = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, tensor):
        tensor = self.fft_dropout(tensor)
        tensor_fft = torch.fft.rfft(tensor, dim=1)
        tensor_fft = torch.matmul(tensor_fft, self.weight)
        tensor_fft_real = self.activation(tensor_fft.real)
        tensor_fft_imag = self.activation(tensor_fft.imag)
        tensor_fft = torch.complex(tensor_fft_real, tensor_fft_imag)
        tensor_ifft = torch.fft.irfft(tensor_fft, tensor.shape[1], dim=1)
        return tensor_ifft


class PredictionLayer(Module):
    def __init__(self, in_features, out_features=1):
        super().__init__()
        self.Linear = nn.Linear(2 * in_features, out_features)
        self.reset_parameters()

    def forward(self, input, edge_src_nodes, edge_trg_nodes):
        src_nodes_features = input.reshape(-1, input.shape[-1])[edge_src_nodes]
        trg_nodes_features = input.reshape(-1, input.shape[-1])[edge_trg_nodes]

        edges_features = torch.cat((src_nodes_features, trg_nodes_features), dim=1)
        output = self.Linear(edges_features)
        return torch.sigmoid(output)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Linear.weight)


class GraphConvolution(Module):
    def __init__(self, time_slices, in_features, out_features, tgc_dropout=0.6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.time_slices = time_slices
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.dropout = nn.Dropout(tgc_dropout)
        self.reset_parameters()

    def compute_AX(self, A, X):
        N = X.size()[1]
        AX = torch.zeros(self.time_slices, N, X.size()[-1], device=X.device)
        for k in range(self.time_slices):
            AX[k] = torch.sparse.mm(A[k], X[k])
        return AX

    def forward(self, adj, input, M):
        h = self.compute_AX(adj, input)
        output = torch.matmul(self.dropout(h), self.weight)
        return output

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)


class FMLPLayer(Module):
    def __init__(self, time_slices, in_features, out_features, fft_dropout):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.time_slices = time_slices
        self.weight = Parameter(torch.empty(in_features, out_features, dtype=torch.float32))
        self.fft_dropout = nn.Dropout(fft_dropout)
        self.activation = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, tensor):
        tensor = self.fft_dropout(tensor)
        tensor = torch.matmul(tensor, self.weight)
        return tensor