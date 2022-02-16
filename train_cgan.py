'''
 FILENAME:      train_cgan.py

 AUTHORS:       Pan shaohua

 START DATE:    2022.02.16/23:52

 CONTACT:       duanxianshi@gmail.com
'''


from multiprocessing import Condition
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from models.cgan import Generator, Discriminator
from config import config


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    dataset = MNIST('data', train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, config['batch_size'], shuffle=True)

    netG = Generator(config['netG']['blocks'], config['netG']['normalize']).to(device)
    print(netG)

    netD = Discriminator(config['netD']['blocks'], config['netD']['normalize'], config['netD']['drop_ratio']).to(device)
    print(netD)

    criterion = nn.BCELoss()

    optimizerG = Adam(netG.parameters(), lr=config['lrG'])
    optimizerD = Adam(netD.parameters(), lr=config['lrD'])

    fixed_noises = torch.rand(80, 100, device=device)
    fixed_conditions = one_hot(torch.tensor([i for i in range(10)] * 8, device=device))

    name = 'exp_cgan'
    writer = SummaryWriter(log_dir='logs/{}'.format(name))

    current_iter = 1
    save_epoch_step = 10
    for epoch in tqdm(range(1, config['n_epochs'] + 1)):
        netG.train()
        netD.train()
        for data in data_loader:
            netD.zero_grad()

            images = data[0].view(-1, 784).to(device)
            conditions = one_hot(data[1], 10).float().to(device)

            batch_size = images.size(dim=0)
            labels = torch.full((batch_size, ), 1., dtype=torch.float, device=device)
            outputs = netD(images, conditions).view(-1)

            errD_real = criterion(outputs, labels)
            errD_real.backward()

            # generator
            noises = torch.rand(batch_size, 100, device=device)
            conditions = one_hot(torch.randint(0, 10, (batch_size, ), device=device)).float()
            fake_img = netG(noises, conditions)
            labels.fill_(0.)
            outputs = netD(fake_img.detach(), conditions).view(-1)

            errD_fake = criterion(outputs, labels)
            errD_fake.backward()

            errD = (errD_real + errD_fake) / 2.

            optimizerD.step()

            writer.add_scalar('Discriminator Loss', errD.item(), current_iter)

            netG.zero_grad()

            labels.fill_(1.)
            outputs = netD(fake_img, conditions).view(-1)
            errG = criterion(outputs, labels)
            errG.backward()

            optimizerG.step()

            writer.add_scalar('Generator Loss', errG.item(), current_iter)

            current_iter += 1

        netG.eval()
        outputs = netG(fixed_noises, fixed_conditions).view(-1, 1, 28, 28)

        writer.add_images('Generated Images', outputs, epoch)

        if epoch % save_epoch_step == 0:
            torch.save({
            'netG': netG.state_dict(),
            'netD': netD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
        }, 'logs/{}/checkpoints/{}_{}.pth'.format(name, name, epoch))


if __name__ == '__main__':
    train(config)