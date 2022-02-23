'''
 FILENAME:      dist_train_cgan.py

 AUTHORS:       Pan shaohua

 START DATE:    2022.02.16/23:52

 CONTACT:       duanxianshi@gmail.com
'''

import argparse
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
from config.cgan import config
import os

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def train(args, config, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    dataset = MNIST('data', train=True, download=True, transform=transform)

    word_size = dist.get_world_size()
    if args.local_rank != -1:
        dataset_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        dataset_sampler = None
    data_loader = DataLoader(dataset, config['batch_size'], 
                    shuffle=(dataset_sampler is None),
                    sampler=dataset_sampler,
                    drop_last=True)

    netG = Generator(config['netG']['blocks'], config['netG']['normalize']).to(device)
    if args.local_rank != -1:
        netG = DDP(netG, device_ids=[args.local_rank])
    print(netG)

    netD = Discriminator(config['netD']['blocks'], config['netD']['normalize'], config['netD']['drop_ratio']).to(device)
    if args.local_rank != -1:
        netD = DDP(netD, device_ids=[args.local_rank])
    print(netD)

    criterion = nn.BCELoss()

    optimizerG = Adam(netG.parameters(), lr=config['lrG'])
    optimizerD = Adam(netD.parameters(), lr=config['lrD'])

    fixed_noises = torch.rand(80, 100, device=device)
    fixed_conditions = one_hot(torch.tensor([i for i in range(10)] * 8, device=device))

    name = args.exp_name
    writer = SummaryWriter(log_dir='logs/{}'.format(args.exp_name))
    save_path = 'logs/{}/checkpoints'.format(name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    current_iter = 1
    save_epoch_step = 10
    for epoch in tqdm(range(1, config['n_epochs'] + 1)):
        netG.train()
        netD.train()
        data_loader.sampler.set_epoch(epoch)

        for data in data_loader:
            netD.zero_grad()

            images = data[0].view(-1, 784).cuda(device, non_blocking=True)
            conditions = one_hot(data[1], 10).float().cuda(device, non_blocking=True)

            batch_size = images.size(dim=0)
            labels = torch.full((batch_size, ), 1., dtype=torch.float).cuda(device, non_blocking=True)
            outputs = netD(images, conditions).view(-1)

            errD_real = criterion(outputs, labels)
            errD_real.backward()

            # generator
            noises = torch.rand(batch_size, 100).cuda(device, non_blocking=True)
            conditions = one_hot(torch.randint(0, 10, (batch_size, ))).float().cuda(device, non_blocking=True)
            fake_img = netG(noises, conditions)
            labels.fill_(0.)
            outputs = netD(fake_img.detach(), conditions).view(-1)

            errD_fake = criterion(outputs, labels)

            torch.distributed.barrier()

            errD_fake.backward()

            errD = (errD_real + errD_fake) / 2.

            optimizerD.step()

            writer.add_scalar('Discriminator Loss', errD.item(), current_iter)

            netG.zero_grad()

            labels.fill_(1.)
            outputs = netD(fake_img, conditions).view(-1)
            errG = criterion(outputs, labels)

            torch.distributed.barrier()

            errG.backward()

            optimizerG.step()

            writer.add_scalar('Generator Loss', errG.item(), current_iter)

            current_iter += 1

        netG.eval()
        outputs = netG(fixed_noises, fixed_conditions).view(-1, 1, 28, 28)

        writer.add_images('Generated Images', outputs, epoch)

        if epoch % save_epoch_step == 0 and args.local_rank == 0:
            torch.save({
            'netG': netG.module.state_dict(),
            'netD': netD.module.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
        }, '{}/{}_{}.pth'.format(save_path, name, epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--exp_name', default='exp_dist_cgan', type=str)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise Exception('Machine is not supporting DDP training!')

    print("local rank {}".format(args.local_rank))
    assert torch.cuda.device_count() > args.local_rank

    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)

    dist.init_process_group(backend='nccl', init_method='env://')     # nccl是GPU设备上最快、最推荐的后端

    args.world_size = dist.get_world_size()
    print("world size {}".format(args.world_size))
    print("get rank {}".format(dist.get_rank()))

    train(args, config, device)