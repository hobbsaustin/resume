import torch
import torch.optim as optim
import torchvision.utils
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

NC = 3
BATCH_SIZE = 64
IMAGE_SIZE = 64
DATAPATH = 'data'
NZ = 100
NGF = 64
NDF = 64
EPOCHS = 20000
LR = 0.0002
BETA1 = 0.5
writer_fake = SummaryWriter(f'logs/fake')
writer_real = SummaryWriter(f'logs/real')
TRANSFROMS = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = ImageFolder(DATAPATH, transform=TRANSFROMS)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

''' show images in data folder
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis('off')
plt.title('training images')
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()
'''


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.PATH = 'weights/Generator.pt'
        self.main = nn.Sequential(
            nn.ConvTranspose2d(NZ, NGF*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF*8, NGF*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF*4, NGF*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF*2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)

    def save_weights(self):
        torch.save(self.state_dict(), self.PATH)

    def load_weights(self):
        self.load_state_dict(torch.load(self.PATH))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.PATH = 'weights/Discriminator.pt'
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(NDF * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

    def save_weights(self):
        torch.save(self.state_dict(), self.PATH)

    def load_weights(self):
        self.load_state_dict(torch.load(self.PATH))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


netG = Generator().to(device)
netD = Discriminator().to(device)
netG.load_weights()
netD.load_weights()


criterion = nn.BCELoss()
fixed_noise = torch.randn(64, NZ, 1, 1, device=device)
real_label = 1
fake_label = 0
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))


# lists to keep track o progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print('Starting Training Loop...')
for epoch in range(EPOCHS):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, NZ, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch == EPOCHS - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()

                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(real_cpu, normalize=True)
                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=iters
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=iters
                )
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

    if epoch % 50 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
        % (epoch, EPOCHS, i, len(dataloader),
        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))




plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()

netG.save_weights()
netD.save_weights()