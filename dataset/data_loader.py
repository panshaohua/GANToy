'''
 FILENAME:      data_loader.py

 AUTHORS:       Pan shaohua

 START DATE:    2022.02.22/01:11

 CONTACT:       duanxianshi@gmail.com
'''

import torchvision.datasets as dset
import torchvision.transforms as transforms


def get_data_loader(args):

    if args.dataset == 'cifar10':
        trans = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = dset.CIFAR10(root=args.dataroot, download=args.download, train=False, transform=trans)

    elif args.dataset == 'mnist':
        trans = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        dataset = dset.MNIST(root=args.dataroot, download=True, transform=trans)

    assert dataset

    return dataset