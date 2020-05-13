"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from model.MUNIT_model import MUNIT_model
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import numpy as np

class MUNIT_Trainer():
    def __init__(self, hyperparameters):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Initiate the networks
        self.MUNIT_model = MUNIT_model(hyperparameters)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.MUNIT_model = nn.DataParallel(self.MUNIT_model)
            self.MUNIT_model_on_one_gpu = self.MUNIT_model.module
        else:
            self.MUNIT_model_on_one_gpu = self.MUNIT_model
        self.MUNIT_model.to(device)

        # Setup the optimizers
        self.dis_opt, self.gen_opt, self.dis_scheduler, self.gen_scheduler = \
            self.MUNIT_model_on_one_gpu.create_optimizers(hyperparameters)

        self.gen_losses_names = ['loss_gen_recon_x_a',
                                 'loss_gen_recon_x_b',
                                 'loss_gen_recon_s_a',
                                 'loss_gen_recon_s_b',
                                 'loss_gen_recon_c_a',
                                 'loss_gen_recon_c_b',
                                 'loss_gen_cycrecon_x_a',
                                 'loss_gen_cycrecon_x_b',
                                 'loss_gen_adv_a',
                                 'loss_gen_adv_b']

        self.dis_losses_names = ['loss_dis_a', 'loss_dis_a_recon', 'loss_dis_b', 'loss_dis_b_recon']

    def gen_update(self, x_a, x_b, hyperparameters, label_a=None, label_b=None):
        self.gen_opt.zero_grad()
        # forward and compute loss
        losses_gen_total, self.gen_losses = \
            self.MUNIT_model(x_a, x_b, hyperparameters, mode='generator', label_a=label_a, label_b=label_b)
        if self.gen_losses.dim() > 1:
            self.gen_losses = self.gen_losses.mean(dim=0)
        if losses_gen_total.dim() > 0:
            self.loss_gen_total = losses_gen_total.mean()
        else:
            self.loss_gen_total = losses_gen_total
        self.loss_gen_total.backward()

        self.gen_opt.step()

    def dis_update(self, x_a, x_b, hyperparameters, label_a=None, label_b=None,):
        self.dis_opt.zero_grad()
        # forward and compute loss
        losses_dis_total, self.dis_losses = \
            self.MUNIT_model(x_a, x_b, hyperparameters, mode='discriminator', label_a=label_a, label_b=label_b)
        if self.dis_losses.dim() > 1:
            self.dis_losses = self.dis_losses.mean(dim=0)
        if losses_dis_total.dim() > 0:
            self.loss_dis_total = losses_dis_total.mean()
        else:
            self.loss_dis_total = losses_dis_total
        self.loss_dis_total.backward()

        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
