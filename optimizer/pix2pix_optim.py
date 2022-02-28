'''
 FILENAME:      pix2pix_optim.py

 AUTHORS:       Pan Shaohua

 START DATE:    Friday February 25th 2022

 CONTACT:       shaohua.pan@quvideo.com

 INFO:          setup pix2pix optimizer
'''

import torch


def get_optimizers(args, netG, netD):
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lrG, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lrD, betas=(0.5, 0.999))

    return optimizerG, optimizerD

def get_loss_fn(args):
    criterion_GAN = torch.nn.BCELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    if torch.cuda.is_available() and args.cuda:
        criterion_GAN = criterion_GAN.cuda()
        criterion_pixelwise = criterion_pixelwise.cuda()

    return criterion_GAN, criterion_pixelwise
