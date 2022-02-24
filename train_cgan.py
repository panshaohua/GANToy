'''
 FILENAME:      train_cgan.py

 AUTHORS:       Pan shaohua

 START DATE:    2022.02.16/23:52

 CONTACT:       duanxianshi@gmail.com
'''


import argparse
import os
from sqlite3 import Row
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.utils  as vutils

from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import LabelBinarizer
import random, numpy.random

from models.cgan import Generator, Discriminator, weights_init
from config.cgan import config
from dataset.data_loader import *
from utils.tools import *

def seed_torch(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_torch()

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description='CGAN model with pytorch')
    parser.add_argument('--exp', type=str, default='exp_cgan_cifar10')
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--download', default=True)
    parser.add_argument('--img_size', default=128, type=int)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--save_img_iter_step', type=int, default=100)
    parser.add_argument('--print_loss_iter', type=int, default=20)
    parser.add_argument('--save_pth_epoch_step', type=int, default=5)


    return parser.parse_args()

def train(args):

    if args.dataset == 'cifar10':
        n_classes = 10
        nc = 3

    lb = LabelBinarizer()
    lb.fit(list(range(0, n_classes)))

    def to_categrical(y: torch.FloatTensor):
        y_one_hot = lb.transform(y.cpu())
        float_tensor = torch.FloatTensor(y_one_hot)
        return float_tensor.to(device)

    datasets = get_data_loader(args)

    train_loader = DataLoader(datasets, config['batch_size'], shuffle=True)

    netG = Generator().to(device)
    netG.apply(weights_init)
    print(netG)

    netD = Discriminator().to(device)
    netD.apply(weights_init)
    print(netD)

    nz = config['z_dim']

    criterion = nn.BCELoss()

    optimizerG = Adam(netG.parameters(), lr=config['lrG'], betas=config['betasG'])
    optimizerD = Adam(netD.parameters(), lr=config['lrD'], betas=config['betasD'])

    name = args.exp
    writer = SummaryWriter(log_dir='logs/{}'.format(name))
    pth_save_path = 'logs/{}/checkpoints'.format(name)
    if not os.path.exists(pth_save_path):
        os.makedirs(pth_save_path)
    fake_images_save_path = 'logs/{}/fake_images'.format(name)
    if not os.path.exists(fake_images_save_path):
        os.makedirs(fake_images_save_path)
    real_images_save_path = 'logs/{}/real_images'.format(name)
    if not os.path.exists(real_images_save_path):
        os.makedirs(real_images_save_path)

    real_label = 1.
    fake_label = 0.
    target_label = 4
    number_of_images = 8

    current_iter = 1
    for epoch in range(config['n_epochs']):
        for batch, (data, target) in enumerate(train_loader):

            data = data.to(device)
            target = target.to(device)

            target1 = to_categrical(target).unsqueeze(2).unsqueeze(3).float()
            target2 = target1.repeat(1, 1, data.size(2), data.size(3))
            data = torch.cat((data, target2), dim=1)
            label = torch.full((data.size(0), 1), real_label).to(device)

            ####################################################################
            # Train discriminator
            netD.zero_grad()
            output = netD(data)

            loss_D1 = criterion(output, label)
            loss_D1.backward()
            D_x = output.mean().item()

            # training fake data
            noise_z = torch.randn(data.size(0), nz, 1, 1).to(device)
            noise_z = torch.cat((noise_z, target1), dim=1)

            fake_data = netG(noise_z)
            fake_data = torch.cat((fake_data, target2), dim=1)
            label = torch.full((data.size(0), 1), fake_label).to(device)
            output = netD(fake_data.detach())

            loss_D2 = criterion(output, label)
            loss_D2.backward()

            D_G_z1 = output.mean().item()

            d_loss = loss_D1.item() + loss_D2.item()

            optimizerD.step()

            writer.add_scalar('Discriminator Loss', d_loss, current_iter)

            ####################################################################
            # training generator
            netG.zero_grad()

            label = torch.full((data.size(0), 1), real_label).to(device)
            # output = netD(fake_data)
            output = netD(fake_data.to(device))
            lossG = criterion(output, label)
            lossG.backward()

            D_G_z2 = output.mean().item()

            optimizerG.step()

            writer.add_scalar('Generator Loss', lossG.item(), current_iter)

            if batch % args.print_loss_iter == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, config['n_epochs'], batch, len(train_loader), d_loss, lossG.item(), D_x, D_G_z1, D_G_z2))

            if current_iter % args.save_img_iter_step == 0:
                vutils.save_image(data[:16, :nc]*0.5+0.5,
                    '{}/real_samples.png'.format(real_images_save_path, current_iter), nrow=4,
                    normalize=True)

                noise_z1 = torch.randn(data.size(0), nz, 1, 1).to(device)
                target3 = to_categrical(torch.full((data.size(0),1), target_label)).unsqueeze(2).unsqueeze(3).float()  #加到噪声上
                noise_z = torch.cat((noise_z1, target3),dim=1)

                fake_data = netG(noise_z.to(device))

                torchvision.utils.save_image(fake_data[:16]*0.5+0.5, '{}/fake_samples_epoch_{}_iter_{}.png'.format(fake_images_save_path ,epoch, current_iter), nrow=4,normalize=True)

                # Log the images while training
                info = {
                    'real_images': get_real_images(data[:16,:nc], nc, number_of_images, data.size(2), data.size(3)),
                    'generated_image': get_generate_images(netG, noise_z, nc, number_of_images, data.size(2), data.size(3))
                }

                for tag, images in info.items():
                    writer.add_images(tag, images, current_iter)

            current_iter += 1

        if epoch % args.save_pth_epoch_step == 0 and epoch != 0:
            torch.save(netG.state_dict(), '{}/netG_epoch_{}.pth'.format(pth_save_path, epoch))
            torch.save(netD.state_dict(), '{}/netD_epoch_{}.pth'.format(pth_save_path, epoch))


if __name__ == '__main__':
    args = parse_args()

    train(args)