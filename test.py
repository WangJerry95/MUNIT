"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04, get_data_loader_folder, SSIM_PSNR_MSE
from trainer import MUNIT_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torch import nn
import numpy as np
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--content_folder', type=str, help="input image path")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--style_folder', type=str, default='', help="style image path")
parser.add_argument('--metrics', action='store_true', default=False,help="whether to compute metrics or not")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--batch_size',type=int, default=10, help="number of images to be transformed in one batch")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()



torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)
opts.num_style = 1 if opts.style != '' else opts.num_style

# Setup model and data loader
config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    model = MUNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")

if not os.path.exists(opts.output_path):
    os.mkdir(opts.output_path)

data_content_loader = get_data_loader_folder(opts.input_content_folder, opts.batch_size, opts.metrics, new_size=config['new_size'], crop=True)
data_style_loader = get_data_loader_folder(opts.input_style_folder, opts.batch_size, opts.metrics, new_size=config['new_size'], crop=True)

try:
    state_dict = torch.load(opts.checkpoint)
    model.gen_a.load_state_dict(state_dict['a'])
    model.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
    model.gen_a.load_state_dict(state_dict['a'])
    model.gen_b.load_state_dict(state_dict['b'])

model.cuda()
model.eval()
gen_a = nn.DataParallel(model.gen_a)
gen_b = nn.DataParallel(model.gen_b)

with torch.no_grad():
    if opts.metrics:
        ssims = []
        psnrs = []
        mses = []
        fids = []
    # if opts.trainer == 'MUNIT':
    # Start testing
    # style_fixed = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
    for i, (content_images, style_images) in enumerate(zip(data_content_loader, data_style_loader)):
        if i + 1 > opts.num_batch:
            break
        # first encode the content and style images
        content_codes, _ = gen_b(content_images.cuda(), mode='encode')
        _, style_codes = gen_a(style_images.cuda(), mode='encode')
        ###TODO:modify test process, one content is transformed for only one time
        images_transfered = gen_a(content_images.cuda(), mode='decode', content=content_codes, style=style_codes)
        if opts.metrics:
            ssim, psnr, mse, grad, full = SSIM_PSNR_MSE(images_transfered, style_images)
            ssims.append(ssim)
            psnrs.append(psnr)
            mses.append(mse)
        for j in range(images_transfered.size[0]):
            vutils.save_image(images_transfered[i], "%s/test_batch%3d_%03d.jpg" % (opts.output_path, i, j))
            print("image saved at %s/test_%03d_%03d.jpg" % (opts.output_path, i, j))
    mssim = np.mean(np.array(ssims))
    mpsnr = np.mean(np.array(psnrs))
    mmse = np.mean(np.array(mses))
    print("\nssim: %04f, psnr: %04f, mse: %04f" % (mssim, mpsnr, mmse))

