from easydict import EasyDict
import os


cfg = EasyDict()

cfg.data_dir = 'data'
cfg.train_0_filename = os.path.join(cfg.data_dir, 'train_0')
cfg.valid1_0_filename = os.path.join(cfg.data_dir, 'validation1_0')
cfg.valid2_0_filename = os.path.join(cfg.data_dir, 'validation2_0')
cfg.valid_1_filename = os.path.join(cfg.data_dir, 'validation_1')
cfg.test_0_filename = os.path.join(cfg.data_dir, 'test_0')
cfg.test_1_filename = os.path.join(cfg.data_dir, 'test_1')

cfg.l = 3
cfg.hidden_size = 256
cfg.length = 140
cfg.num_layers = 2
cfg.lr = 0.00001
cfg.beta = 0.1
