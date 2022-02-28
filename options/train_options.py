'''
 FILENAME:      options.py

 AUTHORS:       Pan Shaohua

 START DATE:    Friday February 25th 2022

 CONTACT:       shaohua.pan@quvideo.com

 INFO:          
'''

import argparse

def args_option():
    parser = argparse.ArgumentParser(description='Training pix2pix')
    parser.add_argument('--exp', type=str, default='exp_pix2pix_facades0')
    parser.add_argument('--dataset', type=str, default='facades')
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--out_channels', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--print_loss_iter', type=int, default=1)
    parser.add_argument('--sample_interval', type=int, default=200, help='interval between sampling of images from generators')
    parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
    parser.add_argument('--save_pth_epoch_step', type=int, default=5)
    parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--n_D_layers', type=int, default=3)      # patch_size is (70x70)
    parser.add_argument('--lambda_pixel', type=int, default=100, help=' Loss weight of L1 pixel-wise loss between translated image and real image')
    parser.add_argument('--log_dir', type=str, default='logs', help='where to save training log')

    return parser.parse_args()