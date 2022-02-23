'''
 FILENAME:      train_dcgan.py

 AUTHORS:       Pan shaohua

 START DATE:    2022.02.22/01:18

 CONTACT:       duanxianshi@gmail.com
'''

import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import numpy as np
import random

from models.dcgan import *
from dataset.data_loader import *
from config.dcgan import config
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

random.seed(2022)
torch.manual_seed(2022)

cudnn.benchmark = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

number_of_images = 8

def parse_args():
    parser = argparse.ArgumentParser(description='DCGAN model with pytorch')
    parser.add_argument('--dataroot', default='data', type=str)
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--download', default=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cuda', type=str, default='false')
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--save_img_iter_step', type=int, default=2000)
    parser.add_argument('--print_loss_iter', type=int, default=100)
    parser.add_argument('--save_pth_epoch_step', type=int, default=5)

    return check_args(parser.parse_args())

def check_args(args):

    try:
        assert args.epochs >= 1
    except:
        print('Number of epochs must be larger than or equal to one')

    try:
        assert args.batch_size >= 1
    except:
        print('Batch size must be larger than or equal to one')

    if args.dataset == 'cifar10':
        args.channels = 3
    else:
        args.channels = 1

    args.cuda = True if args.cuda == 'True' else False

    return args


def get_real_images(images, image_channel, number_of_images):
    images = images.mul(0.5).add(0.5)
    if image_channel == 3:
        image_np = images.view(-1, image_channel, config['img_size'], config['img_size']).data.cpu().numpy()
    else:
        image_np = images.view(-1, config['img_size'], config['img_size']).data.cpu().numpy()

    return image_np[:number_of_images]

def get_generate_images(netG, z, image_channel, number_of_images):
    netG.eval()
    output = netG(z)
    samples = output.mul(0.5).add(0.5)
    samples = samples.data.cpu().numpy()[: number_of_images]

    generated_images = []
    for sample in samples:
        if image_channel == 3:
            generated_images.append(sample.reshape(image_channel, config['img_size'], config['img_size']))
        else:
            generated_images.append(sample.reshape(config['img_size'], config['img_size']))

    return np.array(generated_images)

def train(args):

    datasets = get_data_loader(args)

    train_loader = DataLoader(datasets, config['batch_size'], shuffle=True)

    ngpu = config['ngpu']
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    print(netG)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    print(netD)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(config['batch_size'], nz, 1, 1, device=device)
    real_label = 1.0
    fake_label = 0.0

    optimizerG = torch.optim.Adam(netG.parameters(), lr=config['lrG'], betas=config['betasG'])
    optimizerD = torch.optim.Adam(netD.parameters(), lr=config['lrD'], betas=config['betasD'])

    current_iter = 1

    name = 'exp_dcgan'
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

    for epoch in tqdm(range(1, config['n_epochs'] + 1)):

        # netG.train()
        # netD.train()

        for i, images in enumerate(train_loader, 0):
            if i == train_loader.dataset.__len__() // args.batch_size:
                break

            ##################################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ##################################################################
            # Train discriminator
            # Compute BCE_loss using real images
            netD.zero_grad()
            real_imgs = images[0].to(device)
            batch_size = real_imgs.size(0)
            labels = torch.full((batch_size,), real_label, dtype=real_imgs.dtype, device=device)

            outputs = netD(real_imgs)
            d_loss_real = criterion(outputs, labels)
            d_loss_real.backward()
            D_x = outputs.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            labels.fill_(fake_label)
            outputs = netD(fake_images.detach())
            d_loss_fake = criterion(outputs, labels)
            d_loss_fake.backward()

            D_G_z1 = outputs.mean().item()

            # Optimizer discriminator
            d_loss = d_loss_real + d_loss_fake
            optimizerD.step()

            writer.add_scalar('Discriminator Loss', d_loss.item(), current_iter)

            ##################################################################
            # (2) Update G network: maximize log(D(G(z)))
            ##################################################################
            # Train generator
            netG.zero_grad()
            labels.fill_(real_label)
            outputs = netD(fake_images)
            g_loss = criterion(outputs, labels)

            g_loss.backward()

            D_G_z2 = outputs.mean().item()

            optimizerG.step()

            writer.add_scalar('Generator Loss', g_loss.item(), current_iter)

            # print('d_loss, g_loss: ', d_loss.item(), g_loss.item())
            if i % args.print_loss_iter == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, config['n_epochs'], i, len(train_loader), d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))

            if current_iter % args.save_img_iter_step == 0:
                vutils.save_image(real_imgs,
                    '{}/real_samples_iter_{}.png'.format(real_images_save_path, current_iter),
                    normalize=True)

                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                        '{}/fake_samples_epoch_{}_iter_{}.png'.format(fake_images_save_path ,epoch, current_iter),
                        normalize=True)

                # Log values and gradients of the paramaters
                for tag, value in netD.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram(tag, value.data.cpu().numpy(), current_iter)
                    writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), current_iter)

                # Log the images while training
                info = {
                    'real_images': get_real_images(real_imgs, config['in_channel'], number_of_images),
                    'generated_image': get_generate_images(netG, fixed_noise, config['in_channel'], number_of_images)
                }

                for tag, images in info.items():
                    writer.add_images(tag, images, current_iter)

            current_iter += 1

        if epoch % args.save_pth_epoch_step == 0:
            torch.save(netG.state_dict(), '{}/netG_epoch_{}.pth'.format(pth_save_path, epoch))
            torch.save(netD.state_dict(), '{}/netD_epoch_{}.pth'.format(pth_save_path, epoch))

def load_model(model_file_name, model):
    model_path = os.path.join(os.getcwd(), model_file_name)
    model.load_state_dict(torch.load(model_path))
    print('Generator model loaded from {}.'.format(model_path))
    return netG

def evaluate(args, test_loader, G_model_path, netG):
    netG = load_model(G_model_path, netG)
    z = torch.randn(args.batch_size, 100, 1, 1).to(device)
    samples = netG(z)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.data.cpu()
    grid = torchvision.utils.make_grid(samples)
    print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
    torchvision.utils.save_image(grid, 'dgan_model_image.png')
    return netG

def generate_latent_walk(args, number, netG):

    netG.eval()

    if not os.path.exists('interpolated_images/'):
        os.makedirs('interpolated_images/')

    # Interpolate between twe noise(z1, z2) with number_int steps between
    number_int = 10
    z_intp = torch.FloatTensor(1, 100, 1, 1).to(device)
    z1 = torch.randn(1, 100, 1, 1).to(device)
    z2 = torch.randn(1, 100, 1, 1).to(device)

    images = []
    alpha = 1.0 / float(number_int + 1)
    print(alpha)
    for i in range(1, number_int + 1):
        z_intp.data = z1*alpha + z2*(1.0 - alpha)
        alpha += alpha
        fake_im = netG(z_intp)
        fake_im = fake_im.mul(0.5).add(0.5) #denormalize
        images.append(fake_im.view(args.channels, config['img_size'], config['img_size']).data.cpu())

    grid = torchvision.utils.make_grid(images, nrow=number_int )
    torchvision.utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
    print("Saved interpolated images to interpolated_images/interpolated_{}.".format(str(number).zfill(3)))

if __name__ == '__main__':

    args = parse_args()
    if args.is_train:
        train(args)
    # else:
    #     netG = evaluate(args, test_loader, args.netG_pretrained)
    #     for i in range(50):
    #         generate_latent_walk(args, i, netG)



# https://github.com/Ksuryateja/DCGAN-CIFAR10-pytorch/blob/master/gan_cifar.py