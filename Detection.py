import torch
import torchvision.utils as utils
import matplotlib.pyplot as plt
import numpy as np
from config import cfg
from torch.utils.data.dataloader import DataLoader
from datasets import MVTecDataset, DatasetType
from models import Generator


test_dataset = MVTecDataset(DatasetType.Test, cfg)
batch_size = len(test_dataset)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

state = torch.load(f'models/1400.pt')
generator = Generator(cfg.noise_size)
generator.load_state_dict(state['generator'])
generator = generator.cuda()
generator.eval()

z = torch.load(f'learnt_latent_vectors/noise.pt')
fake_images = generator(z)
fake_images = fake_images.mean(dim=1)

for real_image, label, gt in test_dataloader:
    real_images = real_image.mean(dim=1).cuda()
    gt_anomaly_map = gt.cuda()
    labels = label

anomaly_map = real_images - fake_images

for label in cfg.labels:
    ind = np.where(np.array(labels) == label)[0]

    ri = (real_images[ind])[:5].unsqueeze(dim=1)
    fi = (fake_images[ind])[:5].unsqueeze(dim=1)
    am = (anomaly_map[ind])[:5]
    gt = (gt_anomaly_map[ind])[:5]

    for c in range(am.shape[0]):
        am[c] -= am[c].min()
        am[c] /= am[c].max()
        ri[c] -= ri[c].min()
        ri[c] /= ri[c].max()
        fi[c] -= fi[c].min()
        fi[c] /= fi[c].max()
    am = am.unsqueeze(dim=1)

    tensor_to_show = utils.make_grid(torch.cat((fi, ri, am, gt)), nrow=5).permute(1, 2, 0).cpu().detach().numpy()

    plt.title(f'Label: {label}')
    plt.imshow(tensor_to_show)
    plt.tight_layout()
    plt.savefig(f'results/{label}.png')
