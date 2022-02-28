'''
 FILENAME:      train_pix2pix.py

 AUTHORS:       Pan Shaohua

 START DATE:    Friday February 25th 2022

 CONTACT:       shaohua.pan@quvideo.com

 INFO:          
'''

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.pix2pix import create_net
from options.train_options import args_option
from dataset.get_dataset import *
from optimizer.pix2pix_optim import *
from utils.tools import sample_images, get_generate_images

from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    args = args_option()
    print(args)

    netG, netD = create_net(args)
    print(netG)
    print(netD)

    train_dataset = get_data_loader(args, 'train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = get_data_loader(args, 'val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = get_data_loader(args, 'test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizerG, optimizerD = get_optimizers(args, netG, netD)

    criterion_GAN, criterion_pixelwise = get_loss_fn(args)

    # Calculate output of image discriminator (PatchGAN)
    d_out_size = args.image_size // (2**args.n_D_layers) - 2

    # for visual
    writer = SummaryWriter(log_dir='{}/{}'.format(args.log_dir, args.exp))
    pth_save_path = '{}/{}/checkpoints'.format(args.log_dir, args.exp)
    if not os.path.exists(pth_save_path):
        os.makedirs(pth_save_path)
    fake_images_save_path = '{}/{}/fake_images'.format(args.log_dir, args.exp)
    if not os.path.exists(fake_images_save_path):
        os.makedirs(fake_images_save_path)
    real_images_save_path = '{}/{}/real_images'.format(args.log_dir, args.exp)
    if not os.path.exists(real_images_save_path):
        os.makedirs(real_images_save_path)

    total_iters = 0
    for epoch in range(args.epochs):
        for index, item in enumerate(train_loader):

            real_A = item['A'].cuda()
            real_B = item['B'].cuda()

            real_label = torch.FloatTensor(np.ones((real_A.size(0), 1, d_out_size,d_out_size))).cuda()
            fake_label = torch.FloatTensor(np.zeros((real_A.size(0), 1, d_out_size,d_out_size))).cuda()
            ################################################
            # train discriminator
            fake_B = netG(real_A)

            for param in netD.parameters():
                param.requires_grad = True

            optimizerD.zero_grad()

            # Fake; stop backprop to the generator by detaching fake_B
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = netD(fake_AB.detach())
            loss_D_fake = criterion_GAN(pred_fake, fake_label)
            # Real
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = netD(real_AB)
            loss_D_real = criterion_GAN(pred_real, real_label)

            loss_D = (loss_D_fake + loss_D_real) * 0.5

            loss_D.backward()
            optimizerD.step()

            ################################################
            # train generator
            for param in netD.parameters():
                param.requires_grad = False

            optimizerG.zero_grad()

            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = netD(fake_AB)
            loss_G_GAN = criterion_GAN(pred_fake, real_label)
            # Second, G(A) = B
            loss_G_L1 = criterion_pixelwise(fake_B, real_B) * args.lambda_pixel

            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizerG.step()

            writer.add_scalar('Discriminator Loss', loss_D.item(), total_iters)
            writer.add_scalar('Generator Loss', loss_G.item(), total_iters)

            if total_iters % args.print_loss_iter == 0:
                print('[{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f}'.format(epoch, args.epochs, index, len(train_loader), loss_D.item(), loss_G.item()))

            batches_done = epoch * len(train_loader) + index
            if total_iters % args.sample_interval == 0 and total_iters != 0:
                sample_images(netG, val_loader, fake_images_save_path, epoch, batches_done)

                # Log the images while training
                val_image = next(iter(test_loader))
                val_real_A = val_image['A'].cuda()
                info = {
                        'real_images': real_B*0.5+0.5,
                        'generated_image': get_generate_images(netG, real_A, 3, 1, val_real_A.size(2), val_real_A.size(3))
                    }

                for tag, images in info.items():
                    writer.add_images(tag, images, total_iters)

            total_iters += 1

        if epoch % args.save_pth_epoch_step == 0 and epoch != 0:
            torch.save(netG.state_dict(), '{}/netG_epoch_{}.pth'.format(pth_save_path, epoch))
            torch.save(netD.state_dict(), '{}/netD_epoch_{}.pth'.format(pth_save_path, epoch))