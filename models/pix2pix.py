'''
 FILENAME:      pix2pix.py

 AUTHORS:       Pan Shaohua

 START DATE:    Friday February 25th 2022

 CONTACT:       shaohua.pan@quvideo.com

 INFO:          
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=True):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False, dropout=False)
        self.down2 = UNetDown(64, 128, dropout=False)
        self.down3 = UNetDown(128, 256, dropout=False)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512)
        self.down8 = UNetDown(512, 512, normalize=False)

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512)
        self.up3 = UNetUp(1024, 512)
        self.up4 = UNetUp(1024, 512, dropout=False)
        self.up5 = UNetUp(1024, 256, dropout=False)
        self.up6 = UNetUp(512, 128, dropout=False)
        self.up7 = UNetUp(256, 64, dropout=False)

        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.up8(u7)


class Discriminator_n_layers(nn.Module):
    def __init__(self, args):
        super(Discriminator_n_layers, self).__init__()

        n_layers = args.n_D_layers
        in_channels = args.out_channels
        def discriminator_block(in_filters, out_filters, k=4, s=2, p=1, norm=True, sigmoid=False):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, k, s, p)]
            if norm:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if sigmoid:
                layers.append(nn.Sigmoid())

            return layers

        sequence = [*discriminator_block(in_channels*2, 64, norm=False)]

        assert n_layers <= 5

        if n_layers == 1:
            '''when n_layers==1, the patch size is (16x16)'''
            out_filters = 64 * 2**(n_layers-1)

        elif 1 < n_layers and n_layers <= 4:
            '''
            when n_layers==2, the patch_size is (34x34)
            when n_layers==3, the patch_size is (70x70), this is the size used in the paper
            when n_layers==4, the patch_size is (142x142)
            '''
            for k in range(1, n_layers):
                sequence += [*discriminator_block(2**(5+k), 2**(6+k))]
            out_filters = 64 * 2**(n_layers-1)

        elif 5 == n_layers:
            '''
            when n_layers==5, the patch_size is (286x286), lis larger than the img_size(256),
            so this is the whole img condition
            '''
            for k in range(1, 4):
                sequence += [*discriminator_block(2**(5+k), 2**(6+k))]
                # k=4
            sequence += [*discriminator_block(2**9, 2**9)] #
            out_filters = 2**9

        num_of_filter = min(2*out_filters, 2**9)

        sequence += [*discriminator_block(out_filters, num_of_filter, k=4, s=1, p=1)]
        sequence += [*discriminator_block(num_of_filter, 1, k=4, s=1, p=1, norm=False, sigmoid=True)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


def create_net(args):
    netG = GeneratorUNet()
    netD = Discriminator_n_layers(args)

    if args.resume:
        netG.load_state_dict(torch.load(args.netG_pretrained_pth))
        netD.load_state_dict(torch.load(args.netD_pretrained_pth))
    else:
        netG.apply(weights_init_normal)
        netD.apply(weights_init_normal)

    if torch.cuda.is_available() and args.cuda:
        netG = netG.cuda()
        netD = netD.cuda()

    return netG, netD


if __name__ == '__main__':
    net = GeneratorUNet()

    input_data = torch.randn(4, 3, 256, 256)

    out_data = net(input_data)

    print(out_data.shape)