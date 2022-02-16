'''
 FILENAME:      config.py

 AUTHORS:       Pan shaohua

 START DATE:    2022.02.16/23:40

 CONTACT:       duanxianshi@gmail.com
'''


import torch.nn as nn


netG = {
    'normalize': True,
    'blocks': [
        {'in_features': 110, 'out_features': 256, 'activation_func': nn.ReLU()},
        {'in_features': 256, 'out_features': 512, 'activation_func': nn.ReLU()},
        {'in_features': 512, 'out_features': 1024, 'activation_func': nn.ReLU()},
        {'in_features': 1024, 'out_features': 1024, 'activation_func': nn.ReLU()},
        {'in_features': 1024, 'out_features': 784, 'activation_func': nn.Tanh()},
    ]
}


netD = {
    'normalize': False,
    'drop_ratio': 0.5,
    'blocks': [
        {'in_features': 794, 'out_features': 512, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 512, 'out_features': 256, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 256, 'out_features': 128, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 128, 'out_features': 64, 'activation_func': nn.LeakyReLU(0.2)},
        {'in_features': 64, 'out_features': 1, 'activation_func': nn.Sigmoid()},
    ]
}

config = {
    'netG': netG,
    'netD': netD,
    'lrG': 4e-3,
    'lrD': 4e-3,
    'batch_size': 128,
    'n_epochs': 200,
}