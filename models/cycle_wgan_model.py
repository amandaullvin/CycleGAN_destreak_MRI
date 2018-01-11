import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys

# TODO (1) remove CycleLoss?
#       We have feat_loss_ArecA, which computes the feature loss between A and recreated A.
#       It's kind of redundant with CycleLoss, which computes the pixelwise L1 loss between those two.
#       But then again, we might want to keep both, so that we keep both similar 
#       in terms of "style" and "pixelwise resemblence".


# TODO use MSELoss of Pytorch?
def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

class CycleWGANModel(BaseModel):
    def name(self):
        return 'CycleWGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.one = self.Tensor([1])
        self.mone = self.one * -1

        # init G related losses to 0 to print in the first few iterations
        self.loss_G_A = Variable(self.Tensor([0]))
        self.loss_G_B = Variable(self.Tensor([0]))
        self.loss_idt_A = Variable(self.Tensor([0]))
        self.loss_idt_B = Variable(self.Tensor([0]))
        self.loss_cycle_A = Variable(self.Tensor([0]))
        self.loss_cycle_B = Variable(self.Tensor([0]))
        self.feat_loss_AfB = Variable(self.Tensor([0]))
        self.feat_loss_BfA = Variable(self.Tensor([0]))
        self.feat_loss_fArecB = Variable(self.Tensor([0]))
        self.feat_loss_fBrecA = Variable(self.Tensor([0]))
        self.feat_loss_ArecA = Variable(self.Tensor([0]))
        self.feat_loss_BrecB = Variable(self.Tensor([0]))
        self.feat_loss = Variable(self.Tensor([0]))
        # ----------------------------------------------------------------

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)


        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            self.netFeat = networks.define_feature_network(opt.which_model_feat, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            
            # let's remove the pools of fake images.
            # self.fake_A_pool = ImagePool(opt.pool_size)
            # self.fake_B_pool = ImagePool(opt.pool_size)

            # define loss functions
            # Note: use WGAN loss for cases where we use D_A or D_B, otherwise use default loss functions
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionFeat = mse_loss
            # initialize optimizers
            if opt.adam:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                # in https://github.com/martinarjovsky/WassersteinGAN, only LR is provided to RMSProp
                self.optimizer_G = torch.optim.RMSprop(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr)
                self.optimizer_D_A = torch.optim.RMSprop(self.netD_A.parameters(), lr=opt.lr)
                self.optimizer_D_B = torch.optim.RMSprop(self.netD_B.parameters(), lr=opt.lr)


        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B.forward(self.real_B)
        self.rec_B = self.netG_A.forward(self.fake_A)

    def freeze_discriminators(self, freeze=True):
        for p in self.netD_A.parameters(): 
            p.requires_grad = not freeze 
        for p in self.netD_B.parameters(): 
            p.requires_grad = not freeze 

    def freeze_generators(self, freeze=True):
        for p in self.netG_A.parameters(): 
            p.requires_grad = not freeze 
        for p in self.netG_B.parameters(): 
            p.requires_grad = not freeze 

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        realcopy = real.clone()
        errD_real = netD.forward(realcopy) # named it as in WGAN-github
        errD_real = errD_real.mean()  # following DCGAN_D::forward function in WGAN-github
        errD_real = errD_real.view(1)
        # Fake
        errD_fake = netD.forward(fake.detach()) # named it as it WGAN-github
        errD_fake = errD_fake.mean()  # following DCGAN_D::forward function in WGAN-github
        errD_fake = errD_fake.view(1)
        # compute gradients for both
        errD_real.backward(self.one) 
        errD_fake.backward(self.mone)
        import pdb; pdb.set_trace()

        errD = errD_real - errD_fake # it's the approximation of  Wasserstein distance between Preal and Pgenerator
        # errD.backward(self.one)

        # return errD_real, errD_fake 
        return errD

    def backward_D_A(self):
        self.freeze_generators(True)
        self.fake_B = self.netG_A.forward(self.real_A)
        self.freeze_generators(False)
        # self.loss_D_A_real, self.loss_D_A_fake = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)

    def backward_D_B(self):
        self.freeze_generators(True)
        self.fake_A = self.netG_B.forward(self.real_B)
        self.freeze_generators(False)
        # self.loss_D_B_real, self.loss_D_B_fake = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        lambda_feat_AfB = self.opt.lambda_feat_AfB    
        lambda_feat_BfA = self.opt.lambda_feat_BfA

        lambda_feat_fArecB = self.opt.lambda_feat_fArecB
        lambda_feat_fBrecA = self.opt.lambda_feat_fBrecA

        lambda_feat_ArecA = self.opt.lambda_feat_ArecA
        lambda_feat_BrecB = self.opt.lambda_feat_BrecB

        if (self.opt.lambda_feat > 0):
            lambda_feat_AfB = self.opt.lambda_feat
            lambda_feat_BfA = self.opt.lambda_feat
            lambda_feat_fArecB = self.opt.lambda_feat
            lambda_feat_fBrecA = self.opt.lambda_feat
            lambda_feat_ArecA = self.opt.lambda_feat
            lambda_feat_BrecB = self.opt.lambda_feat

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A.forward(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B.forward(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # WGAN loss
        # D_A(G_A(A))
        self.fake_B = self.netG_A.forward(self.real_A)
        self.loss_G_A = self.netD_A.forward(self.fake_B) # as in WGAN-github: errG = netD(fake)
        self.loss_G_A = self.loss_G_A.mean()  # following DCGAN_D::forward function in WGAN-github
        self.loss_G_A = self.loss_G_A.view(1)
        self.loss_G_A.backward(self.one, retain_graph=True) # as in WGAN-github: errG.backward(one)
        # FIXME: Api docs says not to use retain_graph and this can be done efficiently in other ways 

        # D_B(G_B(B))
        self.fake_A = self.netG_B.forward(self.real_B)
        self.loss_G_B = self.netD_B.forward(self.fake_A)
        self.loss_G_B = self.loss_G_B.mean()  # following DCGAN_D::forward function in WGAN-github
        self.loss_G_B = self.loss_G_B.view(1)
        self.loss_G_B.backward(self.one, retain_graph=True)
        # FIXME: Api docs says not to use retain_graph and this can be done efficiently in other ways 

        
        # Forward cycle loss
        self.rec_A = self.netG_B.forward(self.fake_B) 
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        
        # Backward cycle loss
        self.rec_B = self.netG_A.forward(self.fake_A)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # Perceptual losses:
        self.feat_loss_AfB = self.criterionFeat(self.netFeat(self.real_A), self.netFeat(self.fake_B)) * lambda_feat_AfB    
        self.feat_loss_BfA = self.criterionFeat(self.netFeat(self.real_B), self.netFeat(self.fake_A)) * lambda_feat_BfA

        self.feat_loss_fArecB = self.criterionFeat(self.netFeat(self.fake_A), self.netFeat(self.rec_B)) * lambda_feat_fArecB
        self.feat_loss_fBrecA = self.criterionFeat(self.netFeat(self.fake_B), self.netFeat(self.rec_A)) * lambda_feat_fBrecA

        self.feat_loss_ArecA = self.criterionFeat(self.netFeat(self.real_A), self.netFeat(self.rec_A)) * lambda_feat_ArecA 
        self.feat_loss_BrecB = self.criterionFeat(self.netFeat(self.real_B), self.netFeat(self.rec_B)) * lambda_feat_BrecB 

        self.feat_loss = self.feat_loss_AfB + self.feat_loss_BfA + self.feat_loss_fArecB \
                        + self.feat_loss_fBrecA + self.feat_loss_ArecA + self.feat_loss_BrecB

        # combined loss
        self.loss_G = self.loss_cycle_A + self.loss_cycle_B \
                        + self.loss_idt_A + self.loss_idt_B + self.feat_loss
        self.loss_G.backward()

    def optimize_parameters_D(self):
        # call self.forward outside!

        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A() # generates fake_B for the iteration
        self.optimizer_D_A.step()

        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B() # generates fake_B for the iteration
        self.optimizer_D_B.step()

        # clip weights for both discriminators
        for p in self.netD_A.parameters():
            p.data.clamp_(self.opt.clip_lower, self.opt.clip_upper)

        for p in self.netD_B.parameters():
            p.data.clamp_(self.opt.clip_lower, self.opt.clip_upper)

    def optimize_parameters_G(self):
        # call self.forward outside!

        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        # D_A_real, D_A_fake = self.loss_D_A_real.data[0], self.loss_D_A_fake.data[0] 
        D_A = self.loss_D_A.data[0]
        G_A = self.loss_G_A.data[0]
        Cyc_A = self.loss_cycle_A.data[0]
        # D_B_real, D_B_fake = self.loss_D_B_real.data[0], self.loss_D_B_fake.data[0] 
        D_B = self.loss_D_B.data[0]
        G_B = self.loss_G_B.data[0]
        Cyc_B = self.loss_cycle_B.data[0]
        # feat_AfB = self.feat_loss_AfB.data[0]
        # feat_BfA = self.feat_loss_BfA.data[0]
        #feat_fArecB = self.feat_loss_fArecB.data[0]
        #feat_fBrecA = self.feat_loss_fBrecA.data[0]
        #feat_ArecA = self.feat_loss_ArecA.data[0]
        #feat_BrecB = self.feat_loss_BrecB.data[0]
        featL = self.feat_loss.data[0]



        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data[0]
            idt_B = self.loss_idt_B.data[0]
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B),
                                ('featL', featL)]) #, ('feat_fArecB', feat_fArecB), ('feat_fBrecA', feat_fBrecA), ('feat_AfB', feat_AfB), ('feat_BfA', feat_BfA), 
                                #('feat_ArecA', feat_ArecA), ('feat_BrecB', feat_BrecB)]) #, ('featL', featL)])
        else:
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B),
                                ('featL', featL)]) #, ('feat_fArecB', feat_fArecB), ('feat_fBrecA', feat_fBrecA), ('feat_AfB', feat_AfB), ('feat_BfA', feat_BfA), 
                                #('feat_ArecA', feat_ArecA), ('feat_BrecB', feat_BrecB)]) #, ('featL', featL)])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        rec_A = util.tensor2im(self.rec_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        rec_B = util.tensor2im(self.rec_B.data)
        if self.opt.identity > 0.0:
            idt_A = util.tensor2im(self.idt_A.data)
            idt_B = util.tensor2im(self.idt_B.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])




    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.nepoch_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

