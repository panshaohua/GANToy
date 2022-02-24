'''
 FILENAME:      tools.py

 AUTHORS:       Shaohua.Pan

 START DATE:    Thursday February 24th 2022

 CONTACT:       duanxianshi@gmail.com
'''

import numpy as np


def get_real_images(images, image_channel, number_of_images, img_h, img_w):
    images = images.mul(0.5).add(0.5)
    if image_channel == 3:
        image_np = images.view(-1, image_channel, img_h, img_w).data.cpu().numpy()
    else:
        image_np = images.view(-1, 1, img_h, img_w).data.cpu().numpy()

    return image_np[:number_of_images]

def get_generate_images(model, z, image_channel, number_of_images, img_h, img_w):
    output = model(z)
    samples = output.mul(0.5).add(0.5)
    samples = samples.data.cpu().numpy()[: number_of_images]

    generated_images = []
    for sample in samples:
        if image_channel == 3:
            generated_images.append(sample.reshape(image_channel, img_h, img_w))
        else:
            generated_images.append(sample.reshape(image_channel, img_h, img_w))

    return np.array(generated_images)