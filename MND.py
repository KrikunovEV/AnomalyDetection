import torch


class MNDeval:

    def __init__(self, l: int):
        self.l = l
        self.constant = 0.91893854711 * l  # (n / 2) * ln(2pi)

    def __call__(self, e, mu, var):
        a = self.l * torch.log(var) / 2
        b = (e - mu).pow(2.).sum(dim=1) / 2 / var
        return -(self.constant + a + b)
