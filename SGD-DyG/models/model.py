import torch
import torch.nn as nn

from layers import TensorGraphConvolution, PredictionLayer, FFTLayer, GraphConvolution, FMLPLayer


class SGDDyG(nn.Module):
    def __init__(self, time_slices, N, hidden_features, num_feature, out_features, bandwidth, tgc_dropout=0.6, fft_dropout=0.6,
                 fft=True, tensor_con=True, fft_mlp=True):
        super(SGDDyG, self).__init__()
        self.M = None
        self.N = N
        self.layers = len(hidden_features)
        self.F = [num_feature] + hidden_features
        self.time_slices = time_slices
        self.X = nn.Parameter(torch.empty(time_slices, N, num_feature))
        self.fft = fft

        if fft:
            if fft_mlp:
                self.fft_layer = FFTLayer(time_slices, num_feature, num_feature, fft_dropout)
            else:
                self.fft_layer = FMLPLayer(time_slices, num_feature, num_feature, fft_dropout)

        self.tgcs = nn.ModuleList()
        for layer in range(self.layers):
            if tensor_con:
                self.tgcs.append(TensorGraphConvolution(time_slices, self.F[layer], self.F[layer + 1], bandwidth, tgc_dropout))
            else:
                self.tgcs.append(GraphConvolution(time_slices, self.F[layer], self.F[layer + 1], tgc_dropout))
        self.activation = nn.ReLU()

        self.predict = PredictionLayer(self.F[-1], out_features)

        self.init_weight()

    def forward(self, A, edges_nodes, M, cl=True):
        X = self.X

        if self.fft:
            H = self.fft_layer(X)
        else:
            H = X

        if cl:
            H = X

        if cl and not self.fft:
            H = self.corruption(H)

        for layer, tgc in enumerate(self.tgcs):
            H = tgc(A, H, M)
            if layer is not self.layers - 1:
                H = self.activation(H)

        edge_src_nodes, edge_trg_nodes = edges_nodes
        output = self.predict(H, edge_src_nodes, edge_trg_nodes)

        return torch.squeeze(output), H

    def corruption(self, X):
        neg_X = X.clone()
        for t in range(X.shape[0]):
            perm = torch.randperm(self.N)
            neg_X[t] = X[t, perm]
        return neg_X

    def init_weight(self):
        nn.init.xavier_normal_(self.X)
