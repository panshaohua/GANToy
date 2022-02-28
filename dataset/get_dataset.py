'''
 FILENAME:      get_dataset.py

 AUTHORS:       Pan shaohua

 START DATE:    2022.02.22/01:11

 CONTACT:       duanxianshi@gmail.com
'''

import torchvision.datasets as dset
import torchvision.transforms as transforms

from PIL import Image

from .pix2pix_data import Pix2PixDataset


def get_data_loader(args, mode='train'):

    if args.dataset == 'cifar10':
        trans = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        datasets = dset.CIFAR10(root=args.dataroot, download=args.download, train=False, transform=trans)

    elif args.dataset == 'mnist':
        trans = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        datasets = dset.MNIST(root=args.dataroot, download=True, transform=trans)

    elif args.dataset == 'facades':
        trans = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        datasets = Pix2PixDataset(args, trans, mode)

    assert datasets

    return datasets