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
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])

        dataset = dset.CIFAR10(root=args.dataroot, download=args.download, transform=trans)


    assert dataset

    return dataset