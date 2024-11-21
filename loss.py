import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Loss(nn.Module):
    def __init__(self, params=None, lam=0, enable_cl=True, tau: float = 0.1):
        super().__init__()
        self.params = params
        self.lam = lam
        self.enable_cl = enable_cl
        self.tau = tau

    def forward(self, input: Tensor, target: Tensor, h1=None, h2=None) -> Tensor:
        loss = nn.BCELoss()
        final_loss = loss(input, target)

        if self.enable_cl:
            l1 = self.cl_loss(h1, h2)
            ret = l1
            ret = ret.mean()
            final_loss += ret

        if self.params is not None:
            l2_norm = torch.norm(self.params, p=2)
            final_loss = final_loss + self.lam * l2_norm

        return final_loss

    def cl_loss(self, h1: torch.Tensor, h2: torch.Tensor):
        T = h1.shape[0]
        losses = []
        for i in range(T):
            u, v = F.normalize(h1[i]), F.normalize(h2[i])
            s = torch.mean(F.normalize(h1[0:i + 1]), dim=0)

            pos_similarity = torch.sum(u * s, dim=1) / self.tau
            neg_similarity = torch.sum(v * s, dim=1) / self.tau

            similarity = torch.cat((pos_similarity, neg_similarity))

            pos_labels = torch.ones_like(pos_similarity)
            neg_labels = torch.zeros_like(neg_similarity)
            labels = torch.cat((pos_labels, neg_labels))

            cl_loss = nn.BCEWithLogitsLoss()
            loss = cl_loss(similarity, labels)
            losses.append(loss)

        return torch.stack(losses, dim=0)
