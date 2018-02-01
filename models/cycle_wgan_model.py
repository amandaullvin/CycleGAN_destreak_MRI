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

def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Variable. output.data is the Tensor we are interested
    print('')
    print('Inside ' + self.__class__.__name__ + ' forward')    
    # print('input: ', type(input))
    # print('input[0]: ', type(input[0]))
    # print('output: ', type(output))
    # print('')
    # print('input size:', input[0].size())
    # print('output size:', output.data.size())
    print('output norm:', output.data.norm())

def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    #print('Inside class:' + self.__class__.__name__)
    # print('grad_input: ', type(grad_input))
    # print('grad_input[0]: ', type(grad_input[0]))
    # print('grad_output: ', type(grad_output))
    # print('grad_output[0]: ', type(grad_output[0]))
    # print('')
    # print('grad_input size:', grad_input[0].size())
    # print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].data.norm())

class CycleWGANModel(BaseModel):
    def name(self):
        return 'CycleWGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.one = self.Tensor([1])
        self.mone = self.one * -1
        
        if opt.which_model_netD != 'dcgan':
            self.ones = torch.ones(1, 19, 19) # FIXME compute size from input and architecture of netD
            self.ones = self.ones.type(new_type=self.Tensor)
            

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
        #self.disp_sumGA = self.loss_G_A.clone() + self.loss_cycle_A.clone()
        #self.disp_sumGB = self.loss_G_B.clone() + self.loss_cycle_B.clone()
        self.loss_sumGA = Variable(self.Tensor([0]))
        self.loss_sumGB = Variable(self.Tensor([0]))
        self.rec_A = None
        self.rec_B = None
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
            if (self.opt.lambda_feat > 0):
                self.netFeat = networks.define_feature_network(opt.which_model_feat, self.gpu_ids)


            #self.netD_A.model[11].register_forward_hook(printnorm)
            #self.netD_A.model[11].register_backward_hook(printgradnorm)
            #self.netG_A.register_forward_hook(printnorm)
            #self.netG_A.register_backward_hook(printgradnorm)
            #self.netD_B.model[11].register_forward_hook(printnorm)
            #self.netD_B.model[11].register_backward_hook(printgradnorm)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            
            # create pools of fake images, if pool size > 0
            if opt.pool_size > 0:
                self.fake_A_pool = ImagePool(opt.pool_size)
                self.fake_B_pool = ImagePool(opt.pool_size)
                self.fake_A = None
                self.fake_B = None
            else:
                self.fake_A_pool = None
                self.fake_B_pool = None

            # define loss functions
            # Note: use WGAN loss for cases where we use D_A or D_B, otherwise use default loss functions
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionFeat = mse_loss
            self.criterionWGAN = networks.WGANLoss()
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
        # compute outputs for real and fake images
        outD_real = netD(real)
        outD_fake = netD(fake.detach())
        #self.disp_outD_real = outD_real.mean()
        #self.disp_outD_fake = outD_fake.mean()
        wloss = self.criterionWGAN(fake=outD_fake, real=outD_real)
        # import pdb; pdb.set_trace()
        if self.opt.which_model_netD == 'dcgan':
            wloss.backward()
        else:
            wloss.backward(self.ones)


        return outD_real.mean(), outD_fake.mean()

    def backward_D_A(self):
        #if self.fake_B_pool is None or self.fake_B is None:               
            self.fake_B = self.netG_A(self.real_A.detach()) # generate a fake image                  
            self.loss_D_A_real, self.loss_D_A_fake = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)            
            #self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)
        #else:            
        #    fake_B = self.fake_B_pool.query(self.fake_B)            
        #    self.loss_D_A_real, self.loss_D_A_fake = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
            #self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        

    def backward_D_B(self):
        #if self.fake_A_pool is None or self.fake_A is None:
            self.fake_A = self.netG_B(self.real_B.detach())     
            self.loss_D_B_real, self.loss_D_B_fake = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)
            #self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)
        #else:
        #    fake_A = self.fake_A_pool.query(self.fake_A)
        #    self.loss_D_B_real, self.loss_D_B_fake = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
            #self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        

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
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            #self.loss_idt_A = self.criterionWGAN(fake=self.idt_A, real=self.real_B) * lambda_B * lambda_idt
            #self.loss_idt_A = self.criterionWGAN(fake=self.idt_A, real=self.real_B) * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            #self.loss_idt_B = self.criterionWGAN(fake=self.idt_B, real=self.real_A) * lambda_A * lambda_idt
            #self.loss_idt_B = self.criterionWGAN(fake=self.idt_B, real=self.real_A) * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0        
        
        # Freeze discriminators so that they are NOT updated
        self.freeze_discriminators(True)

        # WGAN loss
        # D_A(G_A(A))
        self.fake_B = self.netG_A(self.real_A)                       
        outD_A_fake = self.netD_A(self.fake_B)        
        self.loss_G_A = self.criterionWGAN(real=outD_A_fake) # we give as it was a true sample
        #self.loss_G_A.backward(retain_graph=True)
        
        # FIXME: Api docs says not to use retain_graph and this can be done efficiently in other ways 

        # D_B(G_B(B))        
        self.fake_A = self.netG_B(self.real_B)        
        outD_B_fake = self.netD_B(self.fake_A)        
        self.loss_G_B = self.criterionWGAN(real=outD_B_fake)        
        #self.loss_G_B.backward(retain_graph=True)        

        
        # Forward cycle loss
        if lambda_A != 0:
            self.rec_A = self.netG_B(self.fake_B) 
            #self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
            self.loss_cycle_A = self.criterionWGAN(fake=self.netD_B(self.rec_A), real=self.netD_B(self.real_A)) * lambda_A
        else:
            self.loss_cycle_A = 0
        
        # Backward cycle loss
        if lambda_B != 0:
            self.rec_B = self.netG_A(self.fake_A)
            #self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
            self.loss_cycle_B = self.criterionWGAN(fake=self.netD_A(self.rec_B), real=self.netD_A(self.real_B)) * lambda_B
        else:
            self.loss_cycle_B = 0


        self.loss_sumGA = self.loss_G_A + self.loss_cycle_A + self.loss_idt_A
        self.loss_sumGB = self.loss_G_B + self.loss_cycle_B + self.loss_idt_B         


        #self.disp_sumGA = self.loss_G_A.clone() + self.loss_cycle_A.clone()
        #self.disp_sumGB = self.loss_G_B.clone() + self.loss_cycle_B.clone()

        # Perceptual losses:
        if (self.opt.lambda_feat > 0):
            # self.feat_loss_AfB = self.criterionFeat(self.netFeat(self.real_A), self.netFeat(self.fake_B)) * lambda_feat_AfB    
            # self.feat_loss_BfA = self.criterionFeat(self.netFeat(self.real_B), self.netFeat(self.fake_A)) * lambda_feat_BfA

            # self.feat_loss_fArecB = self.criterionFeat(self.netFeat(self.fake_A), self.netFeat(self.rec_B)) * lambda_feat_fArecB
            # self.feat_loss_fBrecA = self.criterionFeat(self.netFeat(self.fake_B), self.netFeat(self.rec_A)) * lambda_feat_fBrecA

            # self.feat_loss_ArecA = self.criterionFeat(self.netFeat(self.real_A), self.netFeat(self.rec_A)) * lambda_feat_ArecA 
            # self.feat_loss_BrecB = self.criterionFeat(self.netFeat(self.real_B), self.netFeat(self.rec_B)) * lambda_feat_BrecB 

            self.feat_loss_AfB = self.criterionWGAN(real=self.netFeat(self.real_A), fake=self.netFeat(self.fake_B)) * lambda_feat_AfB    
            self.feat_loss_BfA = self.criterionWGAN(real=self.netFeat(self.real_B), fake=self.netFeat(self.fake_A)) * lambda_feat_BfA

            #self.feat_loss_fArecB = self.criterionWGAN(self.netFeat(self.fake_A), self.netFeat(self.rec_B)) * lambda_feat_fArecB
            #self.feat_loss_fBrecA = self.criterionWGAN(self.netFeat(self.fake_B), self.netFeat(self.rec_A)) * lambda_feat_fBrecA
            self.feat_loss_fArecB = 0
            self.feat_loss_fBrecA = 0

            self.feat_loss_ArecA = self.criterionWGAN(real=self.netFeat(self.real_A), fake=self.netFeat(self.rec_A)) * lambda_feat_ArecA 
            self.feat_loss_BrecB = self.criterionWGAN(real=self.netFeat(self.real_B), fake=self.netFeat(self.rec_B)) * lambda_feat_BrecB 

            self.loss_sumGA.backward(retain_graph=True)
            self.loss_sumGB.backward(retain_graph=True)

            self.feat_loss = self.feat_loss_AfB + self.feat_loss_BfA + self.feat_loss_fArecB \
                        + self.feat_loss_fBrecA + self.feat_loss_ArecA + self.feat_loss_BrecB

            self.feat_loss.backward()
        else:
            if self.opt.which_model_netD == 'dcgan':
                self.loss_sumGA.backward()
                self.loss_sumGB.backward()
            else:
                self.loss_sumGA.backward(self.ones)
                self.loss_sumGB.backward(self.ones)

        # Unfreeze them for the next iteration of optimize_parameters_D()
        self.freeze_discriminators(False)  

        del outD_A_fake, outD_B_fake 

        

    def optimize_parameters_D(self):
        # call self.forward outside!

        # D_A
        self.optimizer_D_A.zero_grad()    
        self.backward_D_A() # generates the first fake_B for the iteration
        self.optimizer_D_A.step()                

        # D_B
        self.optimizer_D_B.zero_grad()        
        self.backward_D_B() # generates fake_A for the iteration
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
#        print("GRADS A  : First conv (mean: %.8f) Last Deconv: (mean: %.8f)" % (self.netG_A.model.model[0].weight.grad.mean(), self.netG_A.model.model[3].weight.grad.mean()))
#        print("GRADS B  : First conv (mean: %.8f) Last Deconv: (mean: %.8f)" % (self.netG_B.model.model[0].weight.grad.mean(), self.netG_B.model.model[3].weight.grad.mean()))
        self.optimizer_G.step()
#        print("WEIGHTS A: First conv (mean: %.8f) Last Deconv: (mean: %.8f)" % (self.netG_A.model.model[0].weight.mean(), self.netG_A.model.model[3].weight.mean()))
#        print("WEIGHTS B: First conv (mean: %.8f) Last Deconv: (mean: %.8f)" % (self.netG_B.model.model[0].weight.mean(), self.netG_B.model.model[3].weight.mean()))
        #print("mean(G_A_LastConvLayer): %.9f mean(G_B_LastConvLayer): %.9f" % (self.netG_A.model[26].weight.mean(), self.netG_B.model[26].weight.mean()))

    def get_current_errors(self):        
        #D_A = self.loss_D_A.data[0]        
        #D_B = self.loss_D_B.data[0]
        G_A = self.loss_G_A.data[0]
        G_B = self.loss_G_B.data[0]        
        
        if self.opt.which_model_netD != 'dcgan' and type(G_A) == self.Tensor:
            G_A = G_A.mean()
            G_B = G_B.mean()            

        D_A_real, D_A_fake = self.loss_D_A_real.data[0], self.loss_D_A_fake.data[0]
        D_B_real, D_B_fake = self.loss_D_B_real.data[0], self.loss_D_B_fake.data[0]
        #sumGA = self.loss_sumGA.data[0]
        #sumGB = self.loss_sumGB.data[0]

        
        #currentErrors = OrderedDict([('D_A', D_A), ('D_B', D_B), ('sumGA', sumGA), ('sumGB', sumGB)])
        currentErrors = OrderedDict([('D_A_real', D_A_real), ('D_A_fake', D_A_fake), ('D_B_real', D_B_real), ('D_B_fake', D_B_fake),
                                     ('G_A', G_A), ('G_B', G_B)])


        if self.loss_cycle_A is not 0:
            Cyc_A = self.loss_cycle_A.data[0]   
            if self.opt.which_model_netD != 'dcgan' and type(Cyc_A) == self.Tensor:
                Cyc_A = Cyc_A.mean()
            currentErrors['Cyc_A'] = Cyc_A

        if self.loss_cycle_B is not 0:
            Cyc_B = self.loss_cycle_B.data[0]   
            if self.opt.which_model_netD != 'dcgan' and type(Cyc_B) == self.Tensor:
                Cyc_B = Cyc_B.mean()
            currentErrors['Cyc_B'] = Cyc_B
        
        # feat_AfB = self.feat_loss_AfB.data[0]
        # feat_BfA = self.feat_loss_BfA.data[0]
        #feat_fArecB = self.feat_loss_fArecB.data[0]
        #feat_fBrecA = self.feat_loss_fBrecA.data[0]
        #feat_ArecA = self.feat_loss_ArecA.data[0]
        #feat_BrecB = self.feat_loss_BrecB.data[0]
        #featL = self.feat_loss.data[0]



        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data[0]
            idt_B = self.loss_idt_B.data[0]

            currentErrors['idt_A'] = idt_A
            currentErrors['idt_B'] = idt_B

        return currentErrors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)        
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)

        currentVisuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B), ('fake_A', fake_A)])

        if self.rec_A is not None:
            rec_A = util.tensor2im(self.rec_A.data)
            currentVisuals['rec_A'] = rec_A
        if self.rec_B is not None:
            rec_B = util.tensor2im(self.rec_B.data)
            currentVisuals['rec_B'] = rec_B
        
        if self.opt.identity > 0.0:
            idt_A = util.tensor2im(self.idt_A.data)
            idt_B = util.tensor2im(self.idt_B.data)
            currentVisuals['idt_B'] = idt_B
            currentVisuals['idt_A'] = idt_A

        return currentVisuals




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