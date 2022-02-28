'''
 FILENAME:      pix2pix_data.py

 AUTHORS:       Pan Shaohua

 START DATE:    Friday February 25th 2022

 CONTACT:       shaohua.pan@quvideo.com

 INFO:          preparing data for training with pix2pix model
'''

import sys

from matplotlib.pyplot import bar_label
sys.path.append('/data1/shaohua/code/GANToy')

import glob
import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image

from options.train_options import args_option


class Pix2PixDataset(Dataset):
    def __init__(self, args, transforms_=None, mode='train'):
        self.transform = transforms_
        self.args = args
        self.filelist = sorted(glob.glob(os.path.join(args.dataroot, args.dataset, mode, '*.jpg')))

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):

        img = Image.open(self.filelist[index])
        w, h = img.size

        img_A = img.crop((0, 0, w // 2, h))
        img_B = img.crop((w // 2, 0, w, h))

        if self.args.which_direction != 'AtoB':
            img_A, img_B = img_B, img_A

        # data augmentation
        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    args = args_option()

    trans = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    datasets = Pix2PixDataset(args, trans)
    dataloader = DataLoader(datasets, batch_size=1)

    for index, item in enumerate(dataloader):
        img_A = item['A']
        img_B = item['B']

        torchvision.utils.save_image(img_A[0], 'img_A_{}.png'.format(index))
        if index >= 10:
            break