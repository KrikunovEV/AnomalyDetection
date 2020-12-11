from easydict import EasyDict
import os


cfg = EasyDict()

cfg.data_dir = os.path.join('data', 'capsule')
cfg.train_dir = os.path.join(cfg.data_dir, 'train')
cfg.test_dir = os.path.join(cfg.data_dir, 'test')
cfg.gt_dir = os.path.join(cfg.data_dir, 'ground_truth')
cfg.labels = ('good', 'crack', 'faulty_imprint', 'poke', 'scratch', 'squeeze')

cfg.epochs = 1000
cfg.batch_size = 32
cfg.noise_size = 100
cfg.lr = 0.0002
cfg.beta1 = 0.5
