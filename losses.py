import torch.nn as nn
import torch


class ResidualLoss(nn.Module):

    def __init__(self):
        super(ResidualLoss, self).__init__()

    def __call__(self, x, Gz):
        return torch.sum(torch.abs(x - Gz))


class DiscriminationLoss(nn.Module):

    def __init__(self, batch_size):
        super(DiscriminationLoss, self).__init__()
        self.CE_ = nn.BCELoss()
        self.real_labels = torch.full((batch_size,), 1.).cuda()

    def __call__(self, Dz):
        return self.CE_(Dz, self.real_labels)
