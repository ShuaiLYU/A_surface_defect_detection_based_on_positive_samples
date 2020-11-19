import torch
from .base_model import BaseModel
from . import networks
from collections import OrderedDict
import numpy as np
class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='instance', netG='unet_128', dataset_mode='aligned',no_flip=True,
                            input_nc=1,output_nc=1   ,netD="pixel",lr=0 )
        #200326 wslsdx添加
        parser.add_argument('--residual', action='store_true',
                            help='if specified, netG output the residual of A and B')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            # 200327 wslsdx添加
            parser.add_argument('--weight_decay_G', type=float, default=0.0001, help='initial learning rate for adam')
            parser.add_argument('--weight_decay_D', type=float, default=0, help='initial learning rate for adam')
            parser.add_argument('--lr_G', type=float, default=0.0001, help='initial learning rate for adam')
            parser.add_argument('--lr_D', type=float, default=0.0004, help='initial learning rate for adam')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G','D']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        #moide by dy 2020.9.25
        self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                      opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            weight_decay_G=opt.weight_decay_G
            weight_decay_D=opt.weight_decay_D
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), weight_decay=weight_decay_G, lr=opt.lr_G, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),  weight_decay=weight_decay_D,lr=opt.lr_D, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """


        self.real_A = input['patch_defect'].to(self.device)
        self.real_B = input['patch'].to(self.device)
        self.image_paths = input['img_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        b_res=self.opt.residual
        b_res=False
        if b_res:
            x = self.netG(self.real_A)  # G(A)
            self.fake_B = self.real_A + x*2
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pre_score = self.netD(fake_AB)
            # print(torch.sum(pre_score))
            # self.pre_score=torch.max(pre_score)

        else:
            self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        # print(pred_fake.shape)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        visual_ret['real_A'] = self.real_A.detach().cpu().numpy() * 127.5 + 127.5

        visual_ret['fake_B'] = self.real_B.detach().cpu().numpy() * 127.5 + 127.5
        visual_ret['sub'] = (torch.abs(self.real_A-self.real_B)).detach().cpu().numpy() * 255
        arr=(torch.abs(self.real_A - self.real_B)).detach().cpu().numpy()
        MAX,MIN=np.max(arr,axis=(1,2,3),keepdims=True),np.min(arr,axis=(1,2,3),keepdims=True)
        visual_ret["sub2"] = np.array((arr - MIN) / (arr - MAX+(1e-6)) * 255, dtype=np.uint8)
        return visual_ret

    def train(self):
        pass

    def eval(self):
        pass

    def __str__(self):
        print(self.netG)