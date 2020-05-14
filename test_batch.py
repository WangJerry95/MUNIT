"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, prepare_sub_folder, get_data_loader_folder, pytorch03_to_pytorch04, load_inception
from model.MUNIT_model import MUNIT_model
from torch import nn
from scipy.stats import entropy
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from data import ImageFolder
import numpy as np
import torchvision.utils as vutils
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import sys
import torch
import os


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder', help='Path to the config file.')
parser.add_argument('--input_content_folder', type=str, help="input image content folder")
parser.add_argument('--input_style_folder', type=str, help="input image style folder")
parser.add_argument('--output_folder', type=str, help="output image folder")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--a2b', type=int, help="1 for a2b and others for b2a", default=1)
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--num_batch',type=int, default=32, help="number of batches to sample")
parser.add_argument('--num_content',type=int, default=4, help="number of contents to sample in one batch")
parser.add_argument('--num_style',type=int, default=8, help="number of styles to sample in one batch")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
parser.add_argument('--compute_IS', action='store_true', help="whether to compute Inception Score or not")
parser.add_argument('--compute_CIS', action='store_true', help="whether to compute Conditional Inception Score or not")
parser.add_argument('--inception_a', type=str, default='.', help="path to the pretrained inception network for domain A")
parser.add_argument('--inception_b', type=str, default='.', help="path to the pretrained inception network for domain B")
parser.add_argument('--gpu', type=str, help='GPUs to be used')

opts = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu
print("Let's use", torch.cuda.device_count(), "GPUs!")

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']
model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory, test_directory = prepare_sub_folder(output_directory)

# Load the inception networks if we need to compute IS or CIIS
if opts.compute_IS or opts.compute_IS:
    inception = load_inception(opts.inception_b) if opts.a2b else load_inception(opts.inception_a)
    # freeze the inception models and set eval mode
    inception.eval()
    for param in inception.parameters():
        param.requires_grad = False
    inception_up = nn.Upsample(size=(299, 299), mode='bilinear')

# Setup model and data loader
# image_names = ImageFolder(opts.input_folder, transform=None, return_paths=True)
data_content_loader = get_data_loader_folder(opts.input_content_folder, opts.num_content, True, new_size=config['new_size'], crop=False)
data_style_loader = get_data_loader_folder(opts.input_style_folder, opts.num_style, True, new_size=config['new_size'], crop=False)

config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    model = MUNIT_model(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
else:
    sys.exit("Only support MUNIT")

try:
    state_dict = torch.load(opts.checkpoint)
    model.gen_a.load_state_dict(state_dict['a'])
    model.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
    model.gen_a.load_state_dict(state_dict['a'])
    model.gen_b.load_state_dict(state_dict['b'])

model.eval()
encoder = nn.DataParallel(model.gen_a) if opts.a2b \
    else nn.DataParallel(model.gen_b) # encode function
decoder = nn.DataParallel(model.gen_b) if opts.a2b \
    else nn.DataParallel(model.gen_a) # decode function

if opts.compute_IS:
    IS = []
    all_preds = []
if opts.compute_CIS:
    CIS = []

# if opts.trainer == 'MUNIT':
    # Start testing
    # style_fixed = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
for i in range(opts.num_batch):
    images_transfered = []
    # first encode the content and style images
    content_images = data_content_loader[i] # (4, 3, 256, 256)
    style_images = data_style_loader[i] # (8, 3, 256,256)
    content_codes, _ = encoder(content_images, mode='encode')
    _, style_codes = encoder(style_images, mode='encode')
    for (content_code, content_image) in zip(content_codes, content_images):
        # apply all style code for each content code
        content_code = content_code.unsqueeze(0).repeat(opts.num_style, 1, 1, 1)
        content_image_tensor = content_image.unsqueeze(0).repeat(opts.num_style, 1, 1, 1)
        image_transfered = decoder(content_image_tensor, mode='decode', content=content_code, style=style_codes)
        image_transfered = torch.stack((content_image.unsqueeze(0), image_transfered), dim=0)
        images_transfered.append(image_transfered)
    images_transfered = torch.stack(tuple(images_transfered), dim=0)
    style_images = torch.stack((torch.ones_like(style_images[0]).unsqueeze(0), style_images), dim=0)
    display_images = torch.stack((style_images, images_transfered), dim=0)
    image_grid = vutils.make_grid(display_images, nrow=opts.num_style+1, normalize=True, pad_value=1)
    vutils.save_image(image_grid, "%s/test_batch_%03d.jpg" % (test_directory, i))

    # for i, (images, names) in enumerate(zip(data_loader, image_names)):
    #     if opts.compute_CIS:
    #         cur_preds = []
    #     print(names[1])
    #     images = Variable(images.cuda(), volatile=True)
    #     content, _ = encode(images)
    #     style = style_fixed if opts.synchronized else Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
    #     for j in range(opts.num_style):
    #         s = style[j].unsqueeze(0)
    #         outputs = decode(content, s)
    #         outputs = (outputs + 1) / 2.
    #         if opts.compute_IS or opts.compute_CIS:
    #             pred = F.softmax(inception(inception_up(outputs)), dim=1).cpu().data.numpy()  # get the predicted class distribution
    #         if opts.compute_IS:
    #             all_preds.append(pred)
    #         if opts.compute_CIS:
    #             cur_preds.append(pred)
    #         # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
    #         basename = os.path.basename(names[1])
    #         path = os.path.join(opts.output_folder+"_%02d"%j,basename)
    #         if not os.path.exists(os.path.dirname(path)):
    #             os.makedirs(os.path.dirname(path))
    #         vutils.save_image(outputs.data, path, padding=0, normalize=True)
    #     if opts.compute_CIS:
    #         cur_preds = np.concatenate(cur_preds, 0)
    #         py = np.sum(cur_preds, axis=0)  # prior is computed from outputs given a specific input
    #         for j in range(cur_preds.shape[0]):
    #             pyx = cur_preds[j, :]
    #             CIS.append(entropy(pyx, py))
    #     if not opts.output_only:
    #         # also save input images
    #         vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)
    # if opts.compute_IS:
    #     all_preds = np.concatenate(all_preds, 0)
    #     py = np.sum(all_preds, axis=0)  # prior is computed from all outputs
    #     for j in range(all_preds.shape[0]):
    #         pyx = all_preds[j, :]
    #         IS.append(entropy(pyx, py))
    #
    # if opts.compute_IS:
    #     print("Inception Score: {}".format(np.exp(np.mean(IS))))
    # if opts.compute_CIS:
    #     print("conditional Inception Score: {}".format(np.exp(np.mean(CIS))))
#
# elif opts.trainer == 'UNIT':
#     # Start testing
#     for i, (images, names) in enumerate(zip(data_loader, image_names)):
#         print(names[1])
#         images = Variable(images.cuda(), volatile=True)
#         content, _ = encode(images)
#
#         outputs = decode(content)
#         outputs = (outputs + 1) / 2.
#         # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
#         basename = os.path.basename(names[1])
#         path = os.path.join(opts.output_folder,basename)
#         if not os.path.exists(os.path.dirname(path)):
#             os.makedirs(os.path.dirname(path))
#         vutils.save_image(outputs.data, path, padding=0, normalize=True)
#         if not opts.output_only:
#             # also save input images
#             vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)
# else:
#     pass
