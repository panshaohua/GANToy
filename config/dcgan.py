'''
 FILENAME:      dcgan.py

 AUTHORS:       Pan Shaohua

 START DATE:    Wednesday February 23rd 2022

 CONTACT:       shaohua.pan@quvideo.com

 INFO:          
'''


config = {
    'lrG': 0.0002,
    'betasG': (0.5, 0.999),
    'lrD': 0.0002,
    'betasD': (0.5, 0.999),
    'batch_size': 64,
    'n_epochs': 200,
    'z_dim': 100,
    'in_channel': 3,
    'ngf': 64,
    'ndf': 64,
    'img_size':64,
    'ngpu': 4
}