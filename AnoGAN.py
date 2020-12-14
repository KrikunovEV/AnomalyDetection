import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from config import cfg
from torch.utils.data.dataloader import DataLoader
from datasets import MVTecDataset, DatasetType
from models import Generator, Discriminator
from losses import ResidualLoss, DiscriminationLoss


test_dataset = MVTecDataset(DatasetType.Test, cfg)
batch_size = len(test_dataset)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

state = torch.load(f'models/1400.pt')
generator = Generator(cfg.noise_size)
generator.load_state_dict(state['generator'])
generator = generator.cuda()
generator.eval()
discriminator = Discriminator()
discriminator.load_state_dict(state['discriminator'])
discriminator = discriminator.cuda()
discriminator.eval()

z = nn.Parameter(torch.randn(batch_size, cfg.noise_size, 1, 1).cuda(), requires_grad=True)
optimizer = optim.SGD([z], lr=cfg.ano.lr)

D = DiscriminationLoss(batch_size)
R = ResidualLoss()

step = 0

with mlflow.start_run():
    for epoch in range(cfg.ano.epochs):
        for b, (real_image, _label, _gt) in enumerate(test_dataloader):

            real_image = real_image.cuda()
            fake_image = generator(z)
            fake_prob = discriminator(fake_image)

            Lr = R(real_image, fake_image) * (1 - cfg.ano.lambd)
            Ld = D(fake_prob) * cfg.ano.lambd
            L = Lr + Ld

            optimizer.zero_grad()
            L.backward()
            optimizer.step()

            mlflow.log_metric('residual loss', Lr.item(), step=step)
            mlflow.log_metric('discrimination loss', Ld.item(), step=step)
            mlflow.log_metric('common loss', L.item(), step=step)

            step += 1

            print(f'epoch: {epoch + 1}/100, res loss: {Lr.item()}, dis loss: {Ld.item()}, common loss: {L.item()}')

torch.save(z, 'learnt_latent_vectors/noise.pt')
