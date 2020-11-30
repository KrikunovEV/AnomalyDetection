import torch.nn as nn
import torch.optim as optim
from config import cfg
from torch.utils.data.dataloader import DataLoader
from datasets import ECG5000Dataset
from models import ECG5000Model
from MND import MNDeval
from LSTMAD import LSTMAD


train_0 = DataLoader(dataset=ECG5000Dataset(cfg.train_0_filename), shuffle=True)
validation1_0 = DataLoader(dataset=ECG5000Dataset(cfg.valid1_0_filename))
validation2_0 = DataLoader(dataset=ECG5000Dataset(cfg.valid2_0_filename))
validation_1 = DataLoader(dataset=ECG5000Dataset(cfg.valid_1_filename))
test_0 = DataLoader(dataset=ECG5000Dataset(cfg.test_0_filename))
test_1 = DataLoader(dataset=ECG5000Dataset(cfg.test_1_filename))

model = ECG5000Model(input_size=1, hidden_size=cfg.hidden_size, l=cfg.l, num_layers=cfg.num_layers)
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
loss_fn = nn.MSELoss()
eval_fn = MNDeval(l=cfg.l)

LSTMAD_ = LSTMAD(train_0, validation1_0, validation2_0, validation_1, test_0, test_1,
                 model, optimizer, loss_fn, eval_fn, cfg)

LSTMAD_.train()
LSTMAD_.validate()
LSTMAD_.test()
