import torch.nn as nn
import torch.nn.functional as F
from vis_utilities import *
from eval_module import *

class ConvEnc(nn.Module): # strided convolutional encoder without skip connections
    def __init__(self, hparam, in_):
        super(ConvEnc, self).__init__()
        enc1, enc2, enc3 = hparam["CONV_ENCS"][0], hparam["CONV_ENCS"][1], hparam["CONV_ENCS"][2]
        n_chin = in_.shape[1]                       # n channels in
        n_filt = hparam["CONV_N_FILT"]              # number of filters
        stride = hparam["CONV_STRIDE"]              # stride
        kernel_size = hparam["CONV_N_KERNEL"]       # filter size
        padding_size = 1                            # padding

        # ffc
        self.conv1 = nn.Conv1d(n_chin, n_filt * enc1, kernel_size=kernel_size, padding=padding_size, padding_mode="reflect", stride=stride, groups=1, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filt * enc1)
        self.conv2 = nn.Conv1d(n_filt * enc1, n_filt * enc2, kernel_size=kernel_size, padding=padding_size, padding_mode="reflect", stride=stride, groups=1, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filt * enc2)
        self.conv3 = nn.Conv1d(n_filt * enc2, n_filt * enc3, kernel_size=kernel_size, padding=padding_size, padding_mode="reflect", stride=stride, groups=1, bias=False)
        self.bn3 = nn.BatchNorm1d(n_filt * enc3)

        ## fb
        self.conv4 = nn.ConvTranspose1d(n_filt * enc3, n_filt * enc2, kernel_size=kernel_size + 1, padding=padding_size, stride=stride, groups=1, bias=False)
        self.bn4 = nn.BatchNorm1d(n_filt * enc2)
        self.conv5 = nn.ConvTranspose1d(n_filt * enc2, n_filt * enc1, kernel_size=kernel_size + 1, padding=padding_size, stride=stride, groups=1, bias=False)
        self.bn5 = nn.BatchNorm1d(n_filt * enc1)
        self.conv6 = nn.ConvTranspose1d(n_filt * enc1, n_chin, kernel_size=kernel_size + 1, padding=padding_size, stride=stride, groups=1, bias=False)
        self.bn6 = nn.BatchNorm1d(n_chin)

        self.act_enc = nn.SELU()
        self.act_dec = nn.SELU()

    def forward(self, x, y, e, fpd="f"):
        x_, y_ = None, None
        if fpd == "f":
            c1 = F.conv1d(x, weight=self.conv1.weight, bias=self.conv1.bias, stride=self.conv1.stride, padding=self.conv1.padding, dilation=self.conv1.dilation, groups=self.conv1.groups)
            b1 = self.bn1(c1)
            b1 = self.act_enc(b1)

            c2 = F.conv1d(b1, weight=self.conv2.weight, bias=self.conv2.bias, stride=self.conv2.stride, padding=self.conv2.padding, dilation=self.conv2.dilation, groups=self.conv2.groups)
            b2 = self.bn2(c2)
            b2 = self.act_enc(b2)

            c3 = F.conv1d(b2, weight=self.conv3.weight, bias=self.conv3.bias, stride=self.conv3.stride, padding=self.conv3.padding, dilation=self.conv3.dilation, groups=self.conv3.groups)
            y_= self.bn3(c3)
            y_ = self.act_enc(y_)

        if fpd == "b":
            c4 = F.conv_transpose1d(y, weight=self.conv4.weight, bias=self.conv4.bias, stride=self.conv4.stride, padding=self.conv4.padding, dilation=self.conv4.dilation, groups=self.conv4.groups)
            b4 = self.bn4(c4)
            b4 = self.act_dec(b4)

            c5 = F.conv_transpose1d(b4, weight=self.conv5.weight, bias=self.conv5.bias, stride=self.conv5.stride, padding=self.conv5.padding, dilation=self.conv5.dilation, groups=self.conv5.groups)
            b5 = self.bn5(c5)
            b5 = self.act_dec(b5)

            c6 = F.conv_transpose1d(b5, weight=self.conv6.weight, bias=self.conv6.bias, stride=self.conv6.stride, padding=self.conv6.padding, dilation=self.conv6.dilation, groups=self.conv6.groups)
            x_ = self.act_dec(c6)
        return x_, y_


class ConvDec(nn.Module): # strided convolutional decoder with skip connections
    def __init__(self, hparam, in_, out_):
        super(ConvDec, self).__init__()
        enc1, enc2, enc3 = hparam["CONV_ENCS"][0], hparam["CONV_ENCS"][1], hparam["CONV_ENCS"][2]
        n_chin = in_.shape[1]                       # n channels in
        n_chout = out_.shape[1]                     # n channels out
        n_filt = hparam["CONV_N_FILT"]              # number of filters
        stride = hparam["CONV_STRIDE"]              # stride
        kernel_size = hparam["CONV_N_KERNEL"]       # filter size
        padding_size = 1                            # padding

        # ff
        self.conv1 = nn.ConvTranspose1d(n_chin, n_filt * enc2, kernel_size=kernel_size + 1, padding=padding_size, stride=stride, groups=1, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filt * enc2)
        self.conv2 = nn.ConvTranspose1d(n_filt * enc2, n_filt * enc1, kernel_size=kernel_size + 1, padding=padding_size, stride=stride, groups=1, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filt * enc1)
        self.conv3 = nn.ConvTranspose1d(n_filt * enc1, n_chout, kernel_size=kernel_size + 1, padding=padding_size, stride=stride, groups=1, bias=False)
        self.bn3 = nn.BatchNorm1d(n_chout)

        ## fb
        self.conv4 = nn.Conv1d(n_chout, n_filt * enc1, kernel_size=kernel_size, padding=padding_size, padding_mode="reflect", stride=stride, groups=1, bias=False)
        self.bn4 = nn.BatchNorm1d(n_filt * enc1)
        self.conv5 = nn.Conv1d(n_filt * enc1, n_filt * enc2, kernel_size=kernel_size, padding=padding_size, padding_mode="reflect", stride=stride, groups=1, bias=False)
        self.bn5 = nn.BatchNorm1d(n_filt * enc2)
        self.conv6 = nn.Conv1d(n_filt * enc2, n_chin, kernel_size=kernel_size, padding=padding_size, padding_mode="reflect", stride=stride, groups=1, bias=False)
        self.bn6 = nn.BatchNorm1d(n_chin)

        self.act_enc = nn.SELU()
        self.act_dec = nn.SELU()

    def forward(self, x, y, e, fpd="f"):
        x_, y_ = None, None
        if fpd == "f":
            c1 = F.conv_transpose1d(x, weight=self.conv1.weight, bias=self.conv1.bias, stride=self.conv1.stride, padding=self.conv1.padding, dilation=self.conv1.dilation, groups=self.conv1.groups)
            b1 = self.bn1(c1)
            b1 = self.act_dec(b1)

            c2 = F.conv_transpose1d(b1, weight=self.conv2.weight, bias=self.conv2.bias, stride=self.conv2.stride, padding=self.conv2.padding, dilation=self.conv2.dilation, groups=self.conv2.groups)
            b2 = self.bn2(c2)
            b2 = self.act_dec(b2)

            c3 = F.conv_transpose1d(b2, weight=self.conv3.weight, bias=self.conv3.bias, stride=self.conv3.stride, padding=self.conv3.padding, dilation=self.conv3.dilation, groups=self.conv3.groups)
            y_ = self.act_dec(c3)

        if fpd == "b":
            c4 = F.conv1d(y, weight=self.conv4.weight, bias=self.conv4.bias, stride=self.conv4.stride, padding=self.conv4.padding, dilation=self.conv4.dilation, groups=self.conv4.groups)
            b4 = self.bn4(c4)
            b4 = self.act_enc(b4)

            c5 = F.conv1d(b4, weight=self.conv5.weight, bias=self.conv5.bias, stride=self.conv5.stride, padding=self.conv5.padding, dilation=self.conv5.dilation, groups=self.conv5.groups)
            b5 = self.bn5(c5)
            b5 = self.act_enc(b5)

            c6 = F.conv1d(b5, weight=self.conv6.weight, bias=self.conv6.bias, stride=self.conv6.stride, padding=self.conv6.padding, dilation=self.conv6.dilation, groups=self.conv6.groups)
            x_ = self.bn6(c6)
            x_ = self.act_enc(x_)
        return x_, y_


class SConvEnc(nn.Module): # strided convolutional encoder with skip connections
    def __init__(self, hparam, in_):
        super(SConvEnc, self).__init__()
        enc1, enc2, enc3 = hparam["CONV_ENCS"][0], hparam["CONV_ENCS"][1], hparam["CONV_ENCS"][2]
        n_chin = in_.shape[1]                       # n channels in
        n_filt = hparam["CONV_N_FILT"]              # number of filters
        stride = hparam["CONV_STRIDE"]              # stride
        kernel_size = hparam["CONV_N_KERNEL"]       # filter size
        padding_size = 1                            # padding

        # ff
        self.conv1 = nn.Conv1d(n_chin, n_filt * enc1, kernel_size=kernel_size, padding=padding_size, padding_mode="reflect", stride=stride, groups=1, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filt * enc1)
        self.conv2 = nn.Conv1d(n_filt * enc1, n_filt * enc2, kernel_size=kernel_size, padding=padding_size, padding_mode="reflect", stride=stride, groups=1, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filt * enc2)
        self.conv3 = nn.Conv1d(n_filt * enc2, n_filt * enc3, kernel_size=kernel_size, padding=padding_size, padding_mode="reflect", stride=stride, groups=1, bias=False)
        self.bn3 = nn.BatchNorm1d(n_filt * enc3)

        ## fb
        self.conv4 = nn.ConvTranspose1d(n_filt * enc3 * 2, n_filt * enc2, kernel_size=kernel_size + 1, padding=padding_size, stride=stride, groups=1, bias=False)
        self.bn4 = nn.BatchNorm1d(n_filt * enc2)
        self.conv5 = nn.ConvTranspose1d(n_filt * enc2 * 2, n_filt * enc1, kernel_size=kernel_size + 1, padding=padding_size, stride=stride, groups=1, bias=False)
        self.bn5 = nn.BatchNorm1d(n_filt * enc1)
        self.conv6 = nn.ConvTranspose1d(n_filt * enc1 * 2, n_chin, kernel_size=kernel_size + 1, padding=padding_size, stride=stride, groups=1, bias=True)
        self.bn6 = nn.BatchNorm1d(n_chin)

        self.act_enc = nn.SELU()
        self.act_dec = nn.SELU()
        #self.act = nn.LeakyReLU()

        # buffers for skip connections for fb
        self.c1 = torch.Tensor
        self.c2 = torch.Tensor
        self.c3 = torch.Tensor

    def forward(self, x, y, e, fpd="f"):
        x_, y_ = None, None
        if fpd == "f":
            self.c1 = F.conv1d(x, weight=self.conv1.weight, bias=self.conv1.bias, stride=self.conv1.stride, padding=self.conv1.padding, dilation=self.conv1.dilation, groups=self.conv1.groups)
            b1 = self.bn1(self.c1)
            b1 = self.act_enc(b1)

            self.c2 = F.conv1d(b1, weight=self.conv2.weight, bias=self.conv2.bias, stride=self.conv2.stride, padding=self.conv2.padding, dilation=self.conv2.dilation, groups=self.conv2.groups)
            b2 = self.bn2(self.c2)
            b2 = self.act_enc(b2)

            self.c3 = F.conv1d(b2, weight=self.conv3.weight, bias=self.conv3.bias, stride=self.conv3.stride, padding=self.conv3.padding, dilation=self.conv3.dilation, groups=self.conv3.groups)
            y_= self.bn3(self.c3)
            y_ = self.act_enc(y_)

        if fpd == "b":
            c4 = F.conv_transpose1d(torch.cat((y, self.c3.detach()), dim=1), weight=self.conv4.weight, bias=self.conv4.bias, stride=self.conv4.stride, padding=self.conv4.padding, dilation=self.conv4.dilation, groups=self.conv4.groups)
            b4 = self.bn4(c4)
            b4 = self.act_dec(b4)

            c5 = F.conv_transpose1d(torch.cat((b4, self.c2.detach()), dim=1), weight=self.conv5.weight, bias=self.conv5.bias, stride=self.conv5.stride, padding=self.conv5.padding, dilation=self.conv5.dilation, groups=self.conv5.groups)
            b5 = self.bn5(c5)
            b5 = self.act_dec(b5)

            c6 = F.conv_transpose1d(torch.cat((b5, self.c1.detach()), dim=1), weight=self.conv6.weight, bias=self.conv6.bias, stride=self.conv6.stride, padding=self.conv6.padding, dilation=self.conv6.dilation, groups=self.conv6.groups)
            x_ = self.act_dec(c6)
        return x_, y_


class SConvDec(nn.Module): # strided convolutional decoder with skip connections
    def __init__(self, hparam, in_, out_):
        super(SConvDec, self).__init__()
        enc1, enc2, enc3 = hparam["CONV_ENCS"][0], hparam["CONV_ENCS"][1], hparam["CONV_ENCS"][2]
        n_chin = in_.shape[1]                       # n channels in
        n_chout = out_.shape[1]                     # n channels out
        n_filt = hparam["CONV_N_FILT"]              # number of filters
        stride = hparam["CONV_STRIDE"]              # stride
        kernel_size = hparam["CONV_N_KERNEL"]       # filter size
        padding_size = 1                            # padding

        # ff
        self.conv1 = nn.ConvTranspose1d(n_chin, n_filt * enc2, kernel_size=kernel_size + 1, padding=padding_size, stride=stride, groups=1, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filt * enc2)
        self.conv2 = nn.ConvTranspose1d(n_filt * enc2, n_filt * enc1, kernel_size=kernel_size + 1, padding=padding_size, stride=stride, groups=1, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filt * enc1)
        self.conv3 = nn.ConvTranspose1d(n_filt * enc1, n_chout, kernel_size=kernel_size + 1, padding=padding_size, stride=stride, groups=1, bias=True)
        self.bn3 = nn.BatchNorm1d(n_chout)

        ## fb
        self.conv4 = nn.Conv1d(n_chout * 2, n_filt * enc1, kernel_size=kernel_size, padding=padding_size, padding_mode="reflect", stride=stride, groups=1, bias=False)
        self.bn4 = nn.BatchNorm1d(n_filt * enc1)
        self.conv5 = nn.Conv1d(n_filt * enc1 * 2, n_filt * enc2, kernel_size=kernel_size, padding=padding_size, padding_mode="reflect", stride=stride, groups=1, bias=False)
        self.bn5 = nn.BatchNorm1d(n_filt * enc2)
        self.conv6 = nn.Conv1d(n_filt * enc2 * 2, n_chin, kernel_size=kernel_size, padding=padding_size, padding_mode="reflect", stride=stride, groups=1, bias=False)
        self.bn6 = nn.BatchNorm1d(n_chin)

        self.act_enc = nn.SELU()
        self.act_dec = nn.SELU()
        #self.act_out = nn.LeakyReLU()

        # buffers for skip connections for fb
        self.c1 = torch.Tensor
        self.c2 = torch.Tensor
        self.c3 = torch.Tensor

    def forward(self, x, y, e, fpd="f"):
        x_, y_ = None, None
        if fpd == "f":
            self.c1 = F.conv_transpose1d(x, weight=self.conv1.weight, bias=self.conv1.bias, stride=self.conv1.stride, padding=self.conv1.padding, dilation=self.conv1.dilation, groups=self.conv1.groups)
            b1 = self.bn1(self.c1)
            b1 = self.act_dec(b1)

            self.c2 = F.conv_transpose1d(b1, weight=self.conv2.weight, bias=self.conv2.bias, stride=self.conv2.stride, padding=self.conv2.padding, dilation=self.conv2.dilation, groups=self.conv2.groups)
            b2 = self.bn2(self.c2)
            b2 = self.act_dec(b2)

            self.c3 = F.conv_transpose1d(b2, weight=self.conv3.weight, bias=self.conv3.bias, stride=self.conv3.stride, padding=self.conv3.padding, dilation=self.conv3.dilation, groups=self.conv3.groups)
            y_ = self.act_dec(self.c3)


        if fpd == "b":
            c4 = F.conv1d(torch.cat((y, self.c3.detach()), dim=1), weight=self.conv4.weight, bias=self.conv4.bias, stride=self.conv4.stride, padding=self.conv4.padding, dilation=self.conv4.dilation, groups=self.conv4.groups)
            b4 = self.bn4(c4)
            b4 = self.act_enc(b4)

            c5 = F.conv1d(torch.cat((b4, self.c2.detach()), dim=1), weight=self.conv5.weight, bias=self.conv5.bias, stride=self.conv5.stride, padding=self.conv5.padding, dilation=self.conv5.dilation, groups=self.conv5.groups)
            b5 = self.bn5(c5)
            b5 = self.act_enc(b5)

            c6 = F.conv1d(torch.cat((b5, self.c1.detach()), dim=1), weight=self.conv6.weight, bias=self.conv6.bias, stride=self.conv6.stride, padding=self.conv6.padding, dilation=self.conv6.dilation, groups=self.conv6.groups)
            x_ = self.bn6(c6)
            x_ = self.act_enc(x_)
        return x_, y_


class CConvEnc(nn.Module): # Causal Convolutional Encoder
    def __init__(self, hparam, in_):
        super(CConvEnc, self).__init__()
        enc1, enc2, enc3 = hparam["CONV_ENCS"][0], hparam["CONV_ENCS"][1], hparam["CONV_ENCS"][2]
        n_chin = in_.shape[1]                       # n channels in
        n_filt = hparam["CONV_N_FILT"]              # number of filters
        dilation = hparam["CONV_DIL"]               # dilation
        kernel_size = hparam["CONV_N_KERNEL"]       # filter size
        stride = (kernel_size - 1) * dilation       # stride
        padding_size = (kernel_size - 1) * dilation # padding
        self.trunc = kernel_size + stride           # truncating the added padding after conv

        # ff
        self.conv1 = nn.Conv1d(n_chin, n_filt * enc1, kernel_size=kernel_size, stride=stride,  padding=0, dilation=dilation, groups=1, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filt * enc1)
        self.pd1 = nn.ConstantPad1d((padding_size, 0), 0)

        self.conv2 = nn.Conv1d(n_filt * enc1, n_filt * enc2, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=1, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filt * enc2)
        self.pd2 = nn.ConstantPad1d((padding_size, 0), 0)

        self.conv3 = nn.Conv1d(n_filt * enc2, n_filt * enc3, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=1, bias=False)
        self.bn3 = nn.BatchNorm1d(n_filt * enc3)
        self.pd3 = nn.ConstantPad1d((padding_size, 0), 0)

        # fb
        self.pd4 = nn.ConstantPad1d((0, padding_size), 0)
        self.conv4 = nn.ConvTranspose1d(n_filt * enc3, n_filt * enc2, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=1, bias=False)
        self.bn4 = nn.BatchNorm1d(n_filt * enc2)

        self.pd5 = nn.ConstantPad1d((0, padding_size), 0)
        self.conv5 = nn.ConvTranspose1d(n_filt * enc2, n_filt * enc1, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=1, bias=False)
        self.bn5 = nn.BatchNorm1d(n_filt * enc1)

        self.pd6 = nn.ConstantPad1d((0, padding_size), 0)
        self.conv6 = nn.ConvTranspose1d(n_filt * enc1, n_chin, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=1, bias=False)
        self.bn6 = nn.BatchNorm1d(n_chin)

        self.act = nn.SELU()

    def forward(self, x, y, e, fpd="f"):
        x_, y_ = None, None
        if fpd == "f":
            x = self.pd1(x)
            c1 = F.conv1d(x, weight=self.conv1.weight, bias=self.conv1.bias, stride=self.conv1.stride, padding=self.conv1.padding, dilation=self.conv1.dilation, groups=self.conv1.groups)
            b1 = self.bn1(c1)
            #b1 = self.act(b1)

            b1 = self.pd2(b1)
            c2 = F.conv1d(b1, weight=self.conv2.weight, bias=self.conv2.bias, stride=self.conv2.stride, padding=self.conv2.padding, dilation=self.conv2.dilation, groups=self.conv2.groups)
            b2 = self.bn2(c2)
            #b2 = self.act(b2)

            b2 = self.pd3(b2)
            c3 = F.conv1d(b2, weight=self.conv3.weight, bias=self.conv3.bias, stride=self.conv3.stride, padding=self.conv3.padding, dilation=self.conv3.dilation, groups=self.conv3.groups)
            y_ = self.bn3(c3)
            #y_ = self.act(b3)

        if fpd == "b":
            c4 = self.pd4(y)
            c4 = self.conv4(c4)[..., self.trunc:]
            b4 = self.bn4(c4)
            #b4 = self.act(b4)

            c5 = self.pd5(b4)
            c5 = self.conv5(c5)[..., self.trunc:]
            b5 = self.bn5(c5)
            #b5 = self.act(b5)

            c6 = self.pd6(b5)
            x_ = self.conv6(c6)[..., self.trunc:]
            # x_ = self.bn6(c6)
            #x_ = self.act(c6)
        return x_, y_


class CConvDec(nn.Module): # Causal Conv Decoder
    def __init__(self, hparam, in_, out_):
        super(CConvDec, self).__init__()
        enc1, enc2, enc3 = hparam["CONV_ENCS"][0], hparam["CONV_ENCS"][1], hparam["CONV_ENCS"][2]
        n_chin = in_.shape[1]                       # n channels in
        n_chout = out_.shape[1]                     # n channels out
        n_filt = hparam["CONV_N_FILT"]              # number of filters
        dilation = hparam["CONV_DIL"]               # dilation
        kernel_size = hparam["CONV_N_KERNEL"]       # filter size
        stride = (kernel_size - 1) * dilation       # stride
        padding_size = (kernel_size - 1) * dilation # padding
        self.trunc = kernel_size + stride           # truncating the added padding after conv

        # ff
        self.conv1 = nn.ConvTranspose1d(n_chin, n_filt * enc2, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=1, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filt * enc2)
        self.pd1 = nn.ConstantPad1d((0, padding_size), 0)

        self.conv2 = nn.ConvTranspose1d(n_filt * enc2, n_filt * enc1, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=1, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filt * enc1)
        self.pd2 = nn.ConstantPad1d((0, padding_size), 0)

        self.conv3 = nn.ConvTranspose1d(n_filt * enc1, n_chout, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=1, bias=False)
        self.bn3 = nn.BatchNorm1d(n_chout)
        self.pd3 = nn.ConstantPad1d((0, padding_size), 0)

        # fb
        self.conv4 = nn.Conv1d(n_chout, n_filt * enc1, kernel_size=kernel_size, stride=stride,  padding=0, dilation=dilation, groups=1, bias=False)
        self.bn4 = nn.BatchNorm1d(n_filt * enc1)
        self.pd4 = nn.ConstantPad1d((padding_size, 0), 0)

        self.conv5 = nn.Conv1d(n_filt * enc1, n_filt * enc2, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=1, bias=False)
        self.bn5 = nn.BatchNorm1d(n_filt * enc2)
        self.pd5 = nn.ConstantPad1d((padding_size, 0), 0)

        self.conv6 = nn.Conv1d(n_filt * enc2, n_chin, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=1, bias=False)
        self.bn6 = nn.BatchNorm1d(n_chin)
        self.pd6 = nn.ConstantPad1d((padding_size, 0), 0)

        self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x, y, e, fpd="f"):
        x_, y_ = None, None
        if fpd == "f":
            x = self.pd1(x)
            c1 = self.conv1(x)[..., self.trunc:]
            b1 = self.bn1(c1)
            #b1 = self.act(b1)

            b1 = self.pd2(b1)
            c2 = self.conv2(b1)[..., self.trunc:]
            b2 = self.bn2(c2)
            #b2 = self.act(b2)

            b2 = self.pd3(b2)
            y_ = self.conv3(b2)[..., self.trunc:]
            #b3 = self.bn3(c3)
            #y_ = self.act(c3)

        if fpd == "b":
            y = self.pd4(y)
            c4 = F.conv1d(y, weight=self.conv4.weight, bias=self.conv4.bias, stride=self.conv4.stride, padding=self.conv4.padding, dilation=self.conv4.dilation, groups=self.conv4.groups)
            b4 = self.bn4(c4)
            #b4 = self.act(b4)

            b4 = self.pd5(b4)
            c5 = F.conv1d(b4, weight=self.conv5.weight, bias=self.conv5.bias, stride=self.conv5.stride, padding=self.conv5.padding, dilation=self.conv5.dilation, groups=self.conv5.groups)
            b5 = self.bn5(c5)
            #b5 = self.act(b5)

            b5 = self.pd6(b5)
            c6 = F.conv1d(b5, weight=self.conv6.weight, bias=self.conv6.bias, stride=self.conv6.stride, padding=self.conv6.padding, dilation=self.conv6.dilation, groups=self.conv6.groups)
            x_ = self.bn6(c6)
            #x_ = self.act(x_)
        return x_, y_


class PepperVar(nn.Module):
    def __init__(self, hparam, in_, out_):
        super(PepperVar, self).__init__()
        # parameters
        self.hparam = hparam
        self.in_, self.out_ = in_, out_
        self.n_in =  len(torch.flatten(in_, start_dim=1)[1])   # n features in
        self.n_out = len(torch.flatten(out_, start_dim=1)[1])  # n features out
        self.n_lat = hparam["VAR_N_LAT"]                       # n features lat

        # init layers
        self.lin_lat_f1 = nn.Linear(in_features=int(self.n_in), out_features=int(self.n_lat), bias=True)
        self.lin_lat_f2 = nn.Linear(in_features=int(self.n_in), out_features=int(self.n_lat), bias=True)
        # sampling here
        self.lin_lat_b1 = nn.Linear(in_features=int(self.n_lat), out_features=int(self.n_out), bias=True)
        self.lin_lat_b2 = nn.Linear(in_features=int(self.n_lat), out_features=int(self.n_out), bias=True)
        self.lat_bias = torch.nn.Parameter(data=torch.Tensor(np.random.uniform(1, -1, self.n_in)))

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl_f = 0
        self.kl_b = 0

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, y, e, fpd="f"):
        x_, y_ = x, y
        # flatten tensor for training stacked model
        if x is not None: x = x.flatten(start_dim=1)
        if y is not None: y = y.flatten(start_dim=1)

        if fpd == "f":  # forward - forward
            mu = self.act(F.linear(x, self.lin_lat_f1.weight, self.lin_lat_f1.bias))
            sig = torch.exp(F.linear(x, self.lin_lat_f2.weight, self.lin_lat_f2.bias))
            lat_f = mu + sig * self.N.sample(mu.shape)
            self.kl_f = (sig ** 2 + mu ** 2 - torch.log(sig) - 0.5).sum()
            y_ = self.act(F.linear(lat_f, self.lin_lat_b1.weight, self.lin_lat_b1.bias))

        if fpd == "b":  # forward - backward
            mu = self.act(F.linear(y, self.lin_lat_b1.weight.transpose(0, 1), self.lin_lat_f1.bias))
            sig = torch.exp(F.linear(y, self.lin_lat_b2.weight.transpose(0, 1), self.lin_lat_f2.bias))
            lat_b = mu + sig * self.N.sample(mu.shape)
            self.kl_b = (sig ** 2 + mu ** 2 - torch.log(sig) - 0.5).sum()
            x_ = self.act(F.linear(lat_b, self.lin_lat_f1.weight.transpose(0, 1), self.lat_bias))
        return torch.reshape(x_, self.in_.shape), torch.reshape(y_, self.out_.shape)


class PepperCore(nn.Module):
    def __init__(self, hparam, in_, out_):
        super(PepperCore, self).__init__()
        # parameters
        self.hparam = hparam
        self.in_, self.out_ = in_, out_
        self.n_in = len(torch.flatten(in_, start_dim=1)[1])    # n features in
        self.n_out = len(torch.flatten(out_, start_dim=1)[1])  # n features out

        self.enc1, self.enc2, self.enc3, self.enc4 = hparam["CORE_ENCS"][0] , hparam["CORE_ENCS"][1], hparam["CORE_ENCS"][2], hparam["CORE_ENCS"][3]   # 1.0, 0.7, 0.5, 0.1
        self.dec1, self.dec2, self.dec3, self.dec4 = hparam["CORE_ENCS"][3] , hparam["CORE_ENCS"][2], hparam["CORE_ENCS"][1], hparam["CORE_ENCS"][0]   # 1.0, 0.7, 0.5, 0.1

        # init layers
        # for each linear layer: lin_*_*.weights == weight tensor, lin_*_*.bias == bias tensor
        self.lin_enc1 = nn.Linear(in_features=int(self.n_in), out_features=int(self.n_in * self.enc1), bias=True)
        self.lin_enc2 = nn.Linear(in_features=int(self.n_in*self.enc1), out_features=int(self.n_in*self.enc2), bias=True)
        self.lin_enc3 = nn.Linear(in_features=int(self.n_in*self.enc2), out_features=int(self.n_in*self.enc3), bias=True)
        self.lin_enc4 = nn.Linear(in_features=int(self.n_in*self.enc3), out_features=int(self.n_in*self.enc4), bias=True)
        self.lin_lat  = nn.Linear(in_features=int(self.n_in*self.enc4), out_features=int(self.n_out*self.dec1), bias=True)
        self.lin_dec1 = nn.Linear(in_features=int(self.n_out*self.dec1), out_features=int(self.n_out*self.dec2), bias=True)
        self.lin_dec2 = nn.Linear(in_features=int(self.n_out*self.dec2), out_features=int(self.n_out*self.dec3), bias=True)
        self.lin_dec3 = nn.Linear(in_features=int(self.n_out*self.dec3), out_features=int(self.n_out*self.dec4), bias=True)
        self.lin_dec4 = nn.Linear(in_features=int(self.n_out*self.dec4), out_features=int(self.n_out), bias=True)

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, y, e, fpd="f"):
        # e = episode, fpd = forward pass direction either "f" for forward "b" for backward
        #if e > 2000: # change activation if training is longer than 200 epochs
        #    self.act = nn.Sigmoid()
        x_, y_ = x, y
        # flatten tensor for training stacked model
        if x is not None: x = x.flatten(start_dim=1)
        if y is not None: y = y.flatten(start_dim=1)

        if fpd == "f":  # forward - forward
            #with torch.no_grad():
            with torch.set_grad_enabled(False):
                h1 = self.act(F.linear(x, self.lin_enc1.weight, self.lin_enc1.bias))
                h2 = self.act(F.linear(h1, self.lin_enc2.weight, self.lin_enc2.bias))
            h3 = self.act(F.linear(h2, self.lin_enc3.weight, self.lin_enc3.bias))
            h4 = self.act(F.linear(h3, self.lin_enc4.weight, self.lin_enc4.bias))
            lat = self.act(F.linear(h4, self.lin_lat.weight, self.lin_lat.bias))
            h4 = self.act(F.linear(lat, self.lin_dec1.weight, self.lin_dec1.bias))
            h5 = self.act(F.linear(h4, self.lin_dec2.weight, self.lin_dec2.bias))
            h6 = self.act(F.linear(h5, self.lin_dec3.weight, self.lin_dec3.bias))
            y_ = self.act(F.linear(h6, self.lin_dec4.weight, self.lin_dec4.bias))
        if fpd == "b":    # forward - backward
            #with torch.no_grad():
            with torch.set_grad_enabled(False):
                h7_b = self.act(F.linear(y, self.lin_dec4.weight.transpose(0, 1), self.lin_dec3.bias))
                h6_b = self.act(F.linear(h7_b, self.lin_dec3.weight.transpose(0, 1), self.lin_dec2.bias))
            h5_b = self.act(F.linear(h6_b, self.lin_dec2.weight.transpose(0, 1), self.lin_dec1.bias))
            h4_b = self.act(F.linear(h5_b, self.lin_dec1.weight.transpose(0, 1), self.lin_lat.bias))
            lat_b = self.act(F.linear(h4_b, self.lin_lat.weight.transpose(0, 1), self.lin_enc4.bias))
            h3_b = self.act(F.linear(lat_b, self.lin_enc4.weight.transpose(0, 1), self.lin_enc3.bias))
            h2_b = self.act(F.linear(h3_b, self.lin_enc3.weight.transpose(0, 1), self.lin_enc2.bias))
            h1_b = self.act(F.linear(h2_b, self.lin_enc2.weight.transpose(0, 1), self.lin_enc1.bias))
            x_ = self.act(F.linear(h1_b, self.lin_enc1.weight.transpose(0, 1), None))
        return torch.reshape(x_, shape=self.in_.shape), torch.reshape(y_, shape=self.out_.shape), None


class PepperVarCore(nn.Module):
    def __init__(self, hparam, in_, out_):
        super(PepperVarCore, self).__init__()
        # parameters
        self.hparam = hparam
        self.in_, self.out_ = in_, out_
        self.n_in =  len(torch.flatten(in_, start_dim=1)[1])   # n features in
        self.n_out = len(torch.flatten(out_, start_dim=1)[1])  # n features out
        self.n_lat = hparam["VAR_N_LAT"]                       # n features lat

        self.enc1, self.enc2, self.enc3, self.enc4 = hparam["CORE_ENCS"][0], hparam["CORE_ENCS"][1], hparam["CORE_ENCS"][2], hparam["CORE_ENCS"][3]   # 1.0, 0.7, 0.5, 0.1
        self.dec1, self.dec2, self.dec3, self.dec4 = hparam["CORE_ENCS"][3], hparam["CORE_ENCS"][2], hparam["CORE_ENCS"][1], hparam["CORE_ENCS"][0]   # 1.0, 0.7, 0.5, 0.1

        # init layers
        # for each linear layer: lin_*_*.weights == weight tensor, lin_*_*.bias == bias tensor
        self.lin_enc1 = nn.Linear(in_features=int(self.n_in), out_features=int(self.n_in * self.enc1), bias=True)
        self.lin_enc2 = nn.Linear(in_features=int(self.n_in*self.enc1), out_features=int(self.n_in*self.enc2), bias=True)
        self.lin_enc3 = nn.Linear(in_features=int(self.n_in*self.enc2), out_features=int(self.n_in*self.enc3), bias=True)
        self.lin_enc4 = nn.Linear(in_features=int(self.n_in*self.enc3), out_features=int(self.n_in*self.enc4), bias=True)

        self.lin_lat_f1 = nn.Linear(in_features=int(self.n_in*self.enc4), out_features=int(self.n_lat), bias=True)
        self.lin_lat_f2 = nn.Linear(in_features=int(self.n_in*self.enc4), out_features=int(self.n_lat), bias=True)
        # sampling
        self.lin_lat_b1 = nn.Linear(in_features=int(self.n_lat), out_features=int(self.n_out*self.dec1), bias=True)
        self.lin_lat_b2 = nn.Linear(in_features=int(self.n_lat), out_features=int(self.n_out*self.dec1), bias=True)

        self.lin_dec1 = nn.Linear(in_features=int(self.n_out*self.dec1), out_features=int(self.n_out*self.dec2), bias=True)
        self.lin_dec2 = nn.Linear(in_features=int(self.n_out*self.dec2), out_features=int(self.n_out*self.dec3), bias=True)
        self.lin_dec3 = nn.Linear(in_features=int(self.n_out*self.dec3), out_features=int(self.n_out*self.dec4), bias=True)
        self.lin_dec4 = nn.Linear(in_features=int(self.n_out*self.dec4), out_features=int(self.n_out), bias=True)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # get sampling on GPU
        self.N.scale = self.N.scale.cuda()
        self.kl_f = 0
        self.kl_b = 0

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, y, e, fpd="f"):
        x_, y_ = x, y
        # flatten tensor for training stacked model
        if x is not None: x = x.flatten(start_dim=1)
        if y is not None: y = y.flatten(start_dim=1)

        if fpd == "f":  # forward - forward
            with torch.set_grad_enabled(False):
                h1 = self.act(F.linear(x, self.lin_enc1.weight, self.lin_enc1.bias))
                h2 = self.act(F.linear(h1, self.lin_enc2.weight, self.lin_enc2.bias))
            h3 = self.act(F.linear(h2, self.lin_enc3.weight, self.lin_enc3.bias))
            h4 = self.act(F.linear(h3, self.lin_enc4.weight, self.lin_enc4.bias))

            mu = self.act(F.linear(h4, self.lin_lat_f1.weight, self.lin_lat_f1.bias))
            sig = torch.exp(F.linear(h4, self.lin_lat_f2.weight, self.lin_lat_f2.bias))
            lat_f = mu + sig * self.N.sample(mu.shape)
            self.kl_f = (sig**2 + mu**2 - torch.log(sig) - 0.5).sum()

            h4 = self.act(F.linear(lat_f, self.lin_lat_b1.weight, self.lin_lat_b1.bias))
            h5 = self.act(F.linear(h4, self.lin_dec1.weight, self.lin_dec1.bias))
            h6 = self.act(F.linear(h5, self.lin_dec2.weight, self.lin_dec2.bias))
            h7 = self.act(F.linear(h6, self.lin_dec3.weight, self.lin_dec3.bias))
            y_ = self.act(F.linear(h7, self.lin_dec4.weight, self.lin_dec4.bias))

        if fpd == "b":    # forward - backward
            with torch.set_grad_enabled(False):
                h8_b = self.act(F.linear(y, self.lin_dec4.weight.transpose(0, 1), self.lin_dec3.bias))
                h7_b = self.act(F.linear(h8_b, self.lin_dec3.weight.transpose(0, 1), self.lin_dec2.bias))
            h6_b = self.act(F.linear(h7_b, self.lin_dec2.weight.transpose(0, 1), self.lin_dec1.bias))
            h5_b = self.act(F.linear(h6_b, self.lin_dec1.weight.transpose(0, 1), self.lin_lat_b1.bias))

            mu = self.act(F.linear(h5_b, self.lin_lat_b1.weight.transpose(0, 1), self.lin_lat_f1.bias))
            sig = torch.exp(F.linear(h5_b, self.lin_lat_b2.weight.transpose(0, 1), self.lin_lat_f2.bias))
            lat_b = mu + sig * self.N.sample(mu.shape)
            self.kl_b = (sig**2 + mu**2 - torch.log(sig) - 0.5).sum()

            h4_b = self.act(F.linear(lat_b, self.lin_lat_f1.weight.transpose(0, 1), self.lin_enc4.bias))
            h3_b = self.act(F.linear(h4_b, self.lin_enc4.weight.transpose(0, 1), self.lin_enc3.bias))
            h2_b = self.act(F.linear(h3_b, self.lin_enc3.weight.transpose(0, 1), self.lin_enc2.bias))
            h1_b = self.act(F.linear(h2_b, self.lin_enc2.weight.transpose(0, 1), self.lin_enc1.bias))
            x_ = self.act(F.linear(h1_b, self.lin_enc1.weight.transpose(0, 1), None))
        return torch.reshape(x_, shape=self.in_.shape), torch.reshape(y_, shape=self.out_.shape), None


class PepperInvCore(nn.Module):
    def __init__(self, hparam, in_, out_):
        super(PepperInvCore, self).__init__()
        # parameters
        self.hparam = hparam
        self.in_, self.out_ = in_, out_
        self.n_in = len(torch.flatten(in_, start_dim=1)[1])    # n features in
        self.n_out = len(torch.flatten(out_, start_dim=1)[1])  # n features out

        self.enc1, self.enc2, self.enc3, self.enc4 = hparam["CORE_ENCS"][0] , hparam["CORE_ENCS"][1], hparam["CORE_ENCS"][2], hparam["CORE_ENCS"][3]   # 1.0, 0.7, 0.5, 0.1
        self.dec1, self.dec2, self.dec3, self.dec4 = hparam["CORE_ENCS"][3] , hparam["CORE_ENCS"][2], hparam["CORE_ENCS"][1], hparam["CORE_ENCS"][0]   # 1.0, 0.7, 0.5, 0.1

        # init layers
        # for each linear layer: lin_*_*.weights == weight tensor, lin_*_*.bias == bias tensor
        self.lin_enc1 = nn.Linear(in_features=int(self.n_in), out_features=int(self.n_in * self.enc1), bias=True)
        self.lin_enc2 = nn.Linear(in_features=int(self.n_in*self.enc1), out_features=int(self.n_in*self.enc2), bias=True)
        self.lin_enc3 = nn.Linear(in_features=int(self.n_in*self.enc2), out_features=int(self.n_in*self.enc3), bias=True)
        self.lin_enc4 = nn.Linear(in_features=int(self.n_in*self.enc3), out_features=int(self.n_in*self.enc4), bias=True)
        self.lin_lat  = nn.Linear(in_features=int(self.n_in*self.enc4), out_features=int(self.n_out*self.dec1), bias=True)
        self.lin_dec1 = nn.Linear(in_features=int(self.n_out*self.dec1), out_features=int(self.n_out*self.dec2), bias=True)
        self.lin_dec2 = nn.Linear(in_features=int(self.n_out*self.dec2), out_features=int(self.n_out*self.dec3), bias=True)
        self.lin_dec3 = nn.Linear(in_features=int(self.n_out*self.dec3), out_features=int(self.n_out*self.dec4), bias=True)
        self.lin_dec4 = nn.Linear(in_features=int(self.n_out*self.dec4), out_features=int(self.n_out * 2), bias=True)

        self.act = nn.SELU()
        self.z_ = torch.Tensor

    def forward(self, x, y, e, fpd="f"):
        x_, y_ = x, y
        # flatten tensor for training stacked model
        if x is not None: x = x.flatten(start_dim=1)
        if y is not None: y = y.flatten(start_dim=1)

        if fpd == "f":  # forward - forward
            with torch.set_grad_enabled(False):
                h1 = self.act(F.linear(x, self.lin_enc1.weight, self.lin_enc1.bias))
                h2 = self.act(F.linear(h1, self.lin_enc2.weight, self.lin_enc2.bias))
            h3 = self.act(F.linear(h2, self.lin_enc3.weight, self.lin_enc3.bias))
            h4 = self.act(F.linear(h3, self.lin_enc4.weight, self.lin_enc4.bias))
            lat = self.act(F.linear(h4, self.lin_lat.weight, self.lin_lat.bias))
            h4 = self.act(F.linear(lat, self.lin_dec1.weight, self.lin_dec1.bias))
            h5 = self.act(F.linear(h4, self.lin_dec2.weight, self.lin_dec2.bias))
            h5 = self.act(F.linear(h4, self.lin_dec2.weight, self.lin_dec2.bias))
            h6 = self.act(F.linear(h5, self.lin_dec3.weight, self.lin_dec3.bias))
            h_out = self.act(F.linear(h6, self.lin_dec4.weight, self.lin_dec4.bias))
            y_, self.z_ = torch.tensor_split(h_out, 2, dim=1)

        if fpd == "b":    # forward - backward
            with torch.set_grad_enabled(False):
                h7_b = self.act(F.linear(torch.cat((y, self.z_), dim=1), self.lin_dec4.weight.transpose(0, 1), self.lin_dec3.bias))
                h6_b = self.act(F.linear(h7_b, self.lin_dec3.weight.transpose(0, 1), self.lin_dec2.bias))
            h5_b = self.act(F.linear(h6_b, self.lin_dec2.weight.transpose(0, 1), self.lin_dec1.bias))
            h4_b = self.act(F.linear(h5_b, self.lin_dec1.weight.transpose(0, 1), self.lin_lat.bias))
            lat_b = self.act(F.linear(h4_b, self.lin_lat.weight.transpose(0, 1), self.lin_enc4.bias))
            h3_b = self.act(F.linear(lat_b, self.lin_enc4.weight.transpose(0, 1), self.lin_enc3.bias))
            h2_b = self.act(F.linear(h3_b, self.lin_enc3.weight.transpose(0, 1), self.lin_enc2.bias))
            h1_b = self.act(F.linear(h2_b, self.lin_enc2.weight.transpose(0, 1), self.lin_enc1.bias))
            x_ = self.act(F.linear(h1_b, self.lin_enc1.weight.transpose(0, 1), None))
        return torch.reshape(x_, shape=self.in_.shape), torch.reshape(y_, shape=self.out_.shape), self.z_



