import torch
import torchvision.utils as utils
import matplotlib.pyplot as plt
from models import Generator


fixed_noise = torch.randn(1, 100, 1, 1)
fake_images = []
for epoch in range(100, 1201, 100):
    state = torch.load(f'models/{epoch}.pt')
    cfg = state['config']

    generator = Generator(cfg.noise_size)
    generator.load_state_dict(state['generator'])
    generator.eval()

    fake_images.append((generator(fixed_noise)[0] + 1.) / 2.)

tensor_to_show = utils.make_grid(torch.stack(fake_images), nrow=4).permute(1, 2, 0).detach().numpy()
plt.imshow(tensor_to_show)
plt.show()
