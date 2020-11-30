import torch.nn as nn


class ECG5000Model(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, l: int, num_layers: int):
        super(ECG5000Model, self).__init__()

        self.linear_1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.LSTM = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=l)

    def __call__(self, x):
        x = self.linear_1(x)
        output, (h, c) = self.LSTM(x)
        x = self.linear_2(output[0])
        return x
