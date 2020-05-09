"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from model.networks import AdaINGen, MsImageDis, VAEGen
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import os
import random
import numpy as np

class MUNIT_model(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_model, self).__init__()
        # Initiate the networks
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']
        self.a_attibute = hyperparameters['label_a']
        self.b_attibute = hyperparameters['label_b']

        # fix the noise used in sampling
        #TODO: It is assumed that style code of different domain are of different distribution, so it can not be drawed from the same distribution
        display_size = int(hyperparameters['display_size'])
        # if self.a_attibute == 0:
        #     self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        # else:
        #     self.s_a = torch.randn(display_size, self.style_dim - self.a_attibute, 1, 1).cuda()
        #     s_attribute = [i%self.a_attibute for i in range(display_size)]
        #     s_attribute = torch.tensor(s_attribute, dtype=torch.long).reshape((display_size, 1))
        #     label_a = torch.zeros(display_size, self.a_attibute, dtype=torch.float32).scatter_(1, s_attribute, 1)
        #     label_a = label_a.reshape(display_size, self.a_attibute, 1, 1).cuda()
        #     self.s_a = torch.cat([self.s_a, label_a], 1)
        # if self.b_attibute == 0:
        #     self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        # else:
        #     self.s_b = torch.randn(display_size, self.style_dim - self.b_attibute, 1, 1).cuda()
        #     s_attribute = [i%self.b_attibute for i in range(display_size)]
        #     s_attribute = torch.tensor(s_attribute, dtype=torch.long).reshape((display_size, 1))
        #     label_b = torch.zeros(display_size, self.b_attibute, dtype=torch.float32).scatter_(1, s_attribute, 1)
        #     label_b = label_b.reshape(display_size, self.b_attibute, 1, 1).cuda()
        #     self.s_b = torch.cat([self.s_b, label_b], 1)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def create_optimizers(self, hyperparameters):
        # Setup the optimizers
        lr = hyperparameters['lr']
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        return self.dis_opt, self.gen_opt, dis_scheduler, gen_scheduler

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b, hyperparameters, mode='generator', label_a=None, label_b=None):

        if mode == 'generator':
            # total_losses, losses, x_ab, x_ba, x_a_recon, x_b_recon
            return self.compute_gen_loss(x_a, x_b, hyperparameters, label_a, label_b)
        if mode == 'discriminator':
            # total_losses, losses, x_ab, x_ba, x_a_recon, x_b_recon
            return self.compute_dis_loss(x_a, x_b, hyperparameters, label_a, label_b)
        # if mode == 'sample':
        #     self.eval()
        #     # s_a1 = Variable(self.s_a)
        #     # s_b1 = Variable(self.s_b)
        #     # s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        #     # s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        #     x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        #     for i in range(x_a.size(0)):
        #         c_a, s_a = self.gen_a.encode(x_a[i].unsqueeze(0))
        #         c_b, s_b = self.gen_b.encode(x_b[i].unsqueeze(0))
        #         x_a_recon.append(self.gen_a.decode(x_a, c_a, s_a))
        #         x_b_recon.append(self.gen_b.decode(x_b, c_b, s_b))
        #         x_ba.append(self.gen_a.decode(x_b, c_b, s_a))
        #         x_ab.append(self.gen_b.decode(x_a, c_a, s_b))
        #     x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        #     x_ba, x_ab = torch.cat(x_ba), torch.cat(x_ab)
        #     self.train()
        #     return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def compute_gen_loss(self, x_a, x_b, hyperparameters, label_a=None, label_b=None):

        # if label_a is None:
        #     s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        # else:
        #     style_num = label_a.size(1)
        #     s_a = Variable(torch.randn(x_a.size(0), self.style_dim - style_num, 1, 1).cuda())
        #     label_a = label_a.repeat(x_a.size(0), 1)
        #     label_a = label_a.reshape(x_a.size(0), style_num, 1, 1)
        #     s_a = torch.cat([s_a, label_a], 1)
        # if label_b is None:
        #     s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # else:
        #     style_num = label_b.size(1)
        #     s_b = Variable(torch.randn(x_b.size(0), self.style_dim-style_num, 1, 1).cuda())
        #     label_b = label_b.repeat(x_b.size(0),1)
        #     label_b = label_b.reshape(x_b.size(0), style_num, 1, 1)
        #     s_b = torch.cat([s_b, label_b], 1)
        # encode
        c_a, s_a = self.gen_a.encode(x_a)
        c_b, s_b = self.gen_b.encode(x_b)
        # decode (within domain)
        #TODO: Add geometric transformation within domain
        transform_list = ["transpose", "flip", "rotate","none"]
        transform = random.choice(transform_list)
        if transform == "transpose":
            c_a_t = torch.transpose(c_a, dim0=2, dim1=3)
            x_a_t = torch.transpose(x_a, dim0=2, dim1=3)
            c_b_t = torch.transpose(c_b, dim0=2, dim1=3)
            x_b_t = torch.transpose(x_b, dim0=2, dim1=3)
        elif transform == "flip":
            dim = random.choice([[2, ], [3, ]])
            c_a_t = torch.flip(c_a, dim)
            x_a_t = torch.flip(x_a, dim)
            c_b_t = torch.flip(c_b, dim)
            x_b_t = torch.flip(x_b, dim)
        elif transform == "rotate":
            k = random.choice([1, 2, 3])
            c_a_t = torch.rot90(c_a, k, (2, 3))
            x_a_t = torch.rot90(x_a, k, (2, 3))
            c_b_t = torch.rot90(c_a, k, (2, 3))
            x_b_t = torch.rot90(x_b, k, (2, 3))
        else:
            c_a_t = c_a
            x_a_t = x_a
            c_b_t = c_b
            x_b_t = x_b

        x_a_recon = self.gen_a.decode(x_a_t, c_a_t, s_a)
        x_b_recon = self.gen_b.decode(x_b_t, c_b_t, s_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(x_b, c_b, s_a)
        x_ab = self.gen_b.decode(x_a, c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(x_a, c_a_recon, s_a) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(x_b, c_b_recon, s_b) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a_t)
        loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b_t)
        loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else torch.tensor(0.0, requires_grad=False).cuda()
        loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else torch.tensor(0.0, requires_grad=False).cuda()
        # GAN loss
        loss_gen_adv_a, loss_gen_class_a = self.dis_a.calc_gen_loss(x_ba, label_a)
        loss_gen_adv_b, loss_gen_class_b = self.dis_b.calc_gen_loss(x_ab, label_b)
        # domain-invariant perceptual loss
        loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else torch.tensor(0.0, requires_grad=False).cuda()
        loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else torch.tensor(0.0, requires_grad=False).cuda()
        # total loss
        loss_gen_total = hyperparameters['gan_w'] * loss_gen_adv_a + \
                         hyperparameters['gan_w'] * loss_gen_adv_b + \
                         hyperparameters['recon_x_w'] * loss_gen_recon_x_a + \
                         hyperparameters['recon_s_w'] * loss_gen_recon_s_a + \
                         hyperparameters['recon_c_w'] * loss_gen_recon_c_a + \
                         hyperparameters['recon_x_w'] * loss_gen_recon_x_b + \
                         hyperparameters['recon_s_w'] * loss_gen_recon_s_b + \
                         hyperparameters['recon_c_w'] * loss_gen_recon_c_b + \
                         hyperparameters['recon_x_cyc_w'] * loss_gen_cycrecon_x_a + \
                         hyperparameters['recon_x_cyc_w'] * loss_gen_cycrecon_x_b + \
                         hyperparameters['vgg_w'] * loss_gen_vgg_a + \
                         hyperparameters['vgg_w'] * loss_gen_vgg_b + \
                         hyperparameters['gan_w'] * loss_gen_class_a + \
                         hyperparameters['gan_w'] * loss_gen_class_b

        losses = torch.stack((loss_gen_recon_x_a,
                              loss_gen_recon_x_b,
                              loss_gen_recon_s_a,
                              loss_gen_recon_s_b,
                              loss_gen_recon_c_a,
                              loss_gen_recon_c_b,
                              loss_gen_cycrecon_x_a,
                              loss_gen_cycrecon_x_b,
                              loss_gen_adv_a,
                              loss_gen_adv_b,
                              loss_gen_vgg_a,
                              loss_gen_vgg_b,
                              loss_gen_class_a,
                              loss_gen_class_b))
        losses = losses.unsqueeze(0)

        return loss_gen_total, losses, x_ab, x_ba, x_a_recon, x_b_recon

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        #TODO: Style code should only be extracted from certain images, not drawed from prior distribution
        x_a_t_l, x_a_recon, x_b_recon, x_b_t_l, x_ba, x_ab = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b = self.gen_b.encode(x_b[i].unsqueeze(0))
            transform_list = ["transpose", "flip", "rotate", "none"]
            transform = random.choice(transform_list)
            if transform == "transpose":
                c_a_t = torch.transpose(c_a, dim0=2, dim1=3)
                x_a_t = torch.transpose(x_a[i], dim0=2, dim1=3)
                c_b_t = torch.transpose(c_b, dim0=2, dim1=3)
                x_b_t = torch.transpose(x_b[i], dim0=2, dim1=3)
            elif transform == "flip":
                dim = random.choice([[2, ], [3, ]])
                c_a_t = torch.flip(c_a, dim)
                x_a_t = torch.flip(x_a[i], dim)
                c_b_t = torch.flip(c_b, dim)
                x_b_t = torch.flip(x_b[i], dim)
            elif transform == "rotate":
                k = random.choice([1, 2, 3])
                c_a_t = torch.rot90(c_a, k, (2, 3))
                x_a_t = torch.rot90(x_a[i], k, (2, 3))
                c_b_t = torch.rot90(c_a, k, (2, 3))
                x_b_t = torch.rot90(x_b[i], k, (2, 3))
            else:
                c_a_t = c_a
                x_a_t = x_a
                c_b_t = c_b
                x_b_t = x_b
            x_a_recon.append(self.gen_a.decode(x_a_t, c_a_t, s_a))
            x_b_recon.append(self.gen_b.decode(x_b_t, c_b_t, s_b))
            x_ba.append(self.gen_a.decode(x_b_t, c_b_t, s_a))
            x_ab.append(self.gen_b.decode(x_a_t, c_a_t, s_b))
            x_a_t_l.append(x_a_t), x_b_t_l.append(x_b_t)
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba, x_ab = torch.cat(x_ba), torch.cat(x_ab)
        x_a_t, x_b_t = torch.cat(x_a_t_l), torch.cat(x_b_t_l)
        self.train()
        return x_a, x_a_t, x_a_recon, x_ba, x_b, x_b_t, x_b_recon,  x_ab

    def compute_dis_loss(self, x_a, x_b, hyperparameters, label_a=None, label_b=None,):

        # if label_a is None:
        #     s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        # else:# utilize label in the style code
        #     style_num = label_a.size(1)
        #     s_a = Variable(torch.randn(x_a.size(0), self.style_dim - style_num, 1, 1).cuda())
        #     label_a = label_a.repeat(x_a.size(0), 1)
        #     label_a = label_a.reshape(x_a.size(0), style_num, 1, 1)
        #     s_a = torch.cat([s_a, label_a], 1)
        # if label_b is None:
        #     s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # else:# utilize label in the style code
        #     style_num = label_b.size(1)
        #     s_b = Variable(torch.randn(x_b.size(0), self.style_dim-style_num, 1, 1).cuda())
        #     label_b = label_b.repeat(x_b.size(0),1)
        #     label_b = label_b.reshape(x_b.size(0), style_num, 1, 1)
        #     s_b = torch.cat([s_b, label_b], 1)

        # encode
        c_a, s_a = self.gen_a.encode(x_a)
        c_b, s_b = self.gen_b.encode(x_b)
        #TODO:Add discriminator loss within domain transformation
        transform_list = ["transpose", "flip", "rotate", "none"]
        transform = random.choice(transform_list)
        if transform == "transpose":
            c_a_t = torch.transpose(c_a, dim0=2, dim1=3)
            x_a_t = torch.transpose(x_a, dim0=2, dim1=3)
            c_b_t = torch.transpose(c_b, dim0=2, dim1=3)
            x_b_t = torch.transpose(x_b, dim0=2, dim1=3)
        elif transform == "flip":
            dim = random.choice([[2, ], [3, ]])
            c_a_t = torch.flip(c_a, dim)
            x_a_t = torch.flip(x_a, dim)
            c_b_t = torch.flip(c_b, dim)
            x_b_t = torch.flip(x_b, dim)
        elif transform == "rotate":
            k = random.choice([1, 2, 3])
            c_a_t = torch.rot90(c_a, k, (2, 3))
            x_a_t = torch.rot90(x_a, k, (2, 3))
            c_b_t = torch.rot90(c_a, k, (2, 3))
            x_b_t = torch.rot90(x_b, k, (2, 3))
        else:
            c_a_t = c_a
            x_a_t = x_a
            c_b_t = c_b
            x_b_t = x_b
        # decode(within domain)
        x_a_recon = self.gen_a.decode(x_a_t, c_a_t, s_a)
        x_b_recon = self.gen_b.decode(x_b_t, c_b_t, s_b)
        # decode (cross domain)
        x_ab = self.gen_b.decode(x_a, c_a, s_b)
        x_ba = self.gen_a.decode(x_b, c_b, s_a)
        # D loss
        loss_dis_a, loss_class_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a, label_a)
        loss_dis_b, loss_class_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b, label_b)
        loss_dis_a_recon, loss_class_a_recon = self.dis_a.calc_dis_loss(x_a_recon.detach(), x_a, label_a)
        loss_dis_b_recon, loss_class_b_recon = self.dis_a.calc_dis_loss(x_b_recon.detach(), x_b, label_b)

        loss_dis_total = hyperparameters['gan_w'] * (loss_dis_a + loss_dis_a_recon +  loss_dis_b + loss_dis_b_recon) +\
                         hyperparameters['gan_w'] * loss_class_a + hyperparameters['gan_w'] * loss_class_b

        losses = torch.stack((loss_dis_a,
                              loss_dis_a_recon,
                              loss_dis_b,
                              loss_dis_b_recon,
                              loss_class_a,
                              loss_class_b
                              ))
        losses = losses.unsqueeze(0)

        return loss_dis_total, losses, x_ab, x_ba, x_a_recon, x_b_recon

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)


class UNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(UNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(h_a + n_a)
        x_b_recon = self.gen_b.decode(h_b + n_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(h_a_recon + n_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(h_b_recon + n_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
