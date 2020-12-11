import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from config import cfg
from torch.utils.data.dataloader import DataLoader
from datasets import MVTecDataset, DatasetType
from models import Generator, Discriminator


train_0 = DataLoader(dataset=MVTecDataset(DatasetType.Train, cfg),
                     shuffle=True,
                     batch_size=cfg.batch_size,
                     drop_last=True)

generator = Generator(cfg.noise_size).cuda()
discriminator = Discriminator().cuda()

gen_optimizer = optim.Adam(generator.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
dis_optimizer = optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))

criterion = nn.BCELoss()

real_labels = torch.full((cfg.batch_size,), 1.).cuda()
fake_labels = torch.full((cfg.batch_size,), 0.).cuda()

step = 0

with mlflow.start_run():
    for epoch in range(cfg.epochs):
        for b, real_image in enumerate(train_0):

            real_image = real_image.cuda()
            z = torch.randn(cfg.batch_size, cfg.noise_size, 1, 1).cuda()
            fake_image = generator(z)

            dis_optimizer.zero_grad()
            real_prob = discriminator(real_image)
            D_loss = criterion(real_prob, real_labels) + criterion(discriminator(fake_image.detach()), fake_labels)
            D_loss.backward()
            dis_optimizer.step()

            gen_optimizer.zero_grad()
            fake_prob = discriminator(fake_image)
            G_loss = criterion(fake_prob, real_labels)
            G_loss.backward()
            gen_optimizer.step()

            mlflow.log_metric('generator loss', G_loss.item(), step=step)
            mlflow.log_metric('discriminator loss', D_loss.item(), step=step)
            mlflow.log_metric('common loss', G_loss.item() + D_loss.item(), step=step)
            mlflow.log_metric('real image probability', real_prob.mean().item(), step=step)
            mlflow.log_metric('fake image probability', fake_prob.mean().item(), step=step)

            step += 1

            print(f'epoch: {epoch + 1}/{cfg.epochs}, batch: {b + 1}/{len(train_0)}\n'
                  f'gen loss: {G_loss.item()}, dis loss: {D_loss.item()}, common loss: {G_loss.item()+D_loss.item()}\n'
                  f'real prob: {real_prob.mean().item()}, fake prob: {fake_prob.mean().item()}\n')

        if epoch % 10 == 0:
            state = {
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'gen_optimizer': gen_optimizer.state_dict(),
                'dis_optimizer': dis_optimizer.state_dict(),
                'config': cfg,
                'step': step
            }
            torch.save(state, f'models/{epoch}.pt')
