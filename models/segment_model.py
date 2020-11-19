import torch
from .base_model import BaseModel
from . import networks
from collections import OrderedDict
import  torch
import numpy as np
class BCELossWithWeight(torch.nn.Module):
    def __init__(self, weight0, weight1):
        super(BCELossWithWeight, self).__init__()
        self.weight0 = weight0
        self.weight1 = weight1

    def forward(self, input, label):
        """

        :param input: torch.float32 bx1xhxw
        :param label: torch.long    bx1xhxw
        :return:
        """
        loss = -(self.weight0 * torch.mul(torch.log((1 - input).clamp(1e-6, 1)), (1 - label)) +
                 self.weight1 * torch.mul(torch.log(input.clamp(1e-6, 1)), label))
        return torch.mean(loss)



class SegmentModel(BaseModel):
    """
    Lyu Shuai  add this to train a segmentation netowork.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='instance', netG='unet_256', dataset_mode='aligned',no_flip=True,
                            input_nc=1,output_nc=1 ,lr=0 )

        parser.add_argument('--net_S', type=str, default='unet_256', help='should be the same as net_G')
        parser.add_argument('--weight_decay_S', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--lr_S', type=float, default=0.001, help='initial learning rate for adam')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["bce"]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['img', 'mask', 'pred']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['_seg']
        # define networks (both generator and discriminator)
        self.net_seg = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,opt.net_S,
                                    opt.norm,not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,active="sigmoid")
        if self.isTrain:
            weight_decay=opt.weight_decay_S
            # self.BceLoss = torch.nn.BCELoss().to(self.device)
            self.BceLoss = BCELossWithWeight(weight0=1, weight1=4).to(self.device)
            self.optimizer = torch.optim.Adam(self.net_seg.parameters(), weight_decay=weight_decay,
                                                lr=opt.lr_S, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)



    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        #A y
        self.img = input['patch_defect'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.image_paths = input['img_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.pred = self.net_seg(self.img)  # G(A)

    def backward(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.loss_bce = self.BceLoss(self.pred, self.mask)
        self.loss_bce.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()


    def train(self):
        self.net_seg.train()

    def eval(self):
        self.net_seg.eval()

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        visual_ret['img'] = self.img.detach().cpu().numpy() * 127.5 + 127.5
        visual_ret['mask'] = self.mask.detach().cpu().numpy() *255
        visual_ret['pred'] = self.pred.detach().cpu().numpy() *255
        arr = (self.pred).detach().cpu().numpy()
        MAX, MIN = np.max(arr, axis=(1, 2, 3), keepdims=True), np.min(arr, axis=(1, 2, 3), keepdims=True)
        visual_ret["sub2"] = np.array((arr - MIN) / (arr - MAX+(1e-6)) * 255, dtype=np.uint8)
        return visual_ret