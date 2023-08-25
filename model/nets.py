import torch
import torch.nn as nn
import modules as m
from utils import *


""" *******************************  core networks  ***********************************************"""

class PepperNetCore(nn.Module):
    def __init__(self, hparam, in_, out_):
        super(PepperNetCore, self).__init__()
        self.hparam = hparam
        self.in_, self.out_ = in_, out_

        # modules
        self.core = m.PepperCore(hparam=hparam, in_=in_, out_=out_)

    def forward(self, x, y, e):
        # f-f
        _, y_, _ = self.core(x, None, e, fpd="f")

        # if y != NaN continue with y.squeeze(dim=2)
        y = y_ if True in torch.isnan(y) else y.squeeze(dim=2)

        x_, _, _ = self.core(None, y, e, fpd="b")
        return x_, y_, _


class PepperNetVAE(nn.Module):
    def __init__(self, hparam, in_, out_):
        super(PepperNetVAE, self).__init__()
        # parameters
        self.hparam = hparam
        self.in_, self.out_ = in_, out_

        # modules
        self.core = m.PepperVarCore(hparam=hparam, in_=in_, out_=out_)

        # kl
        self.kl_f = self.core.kl_f
        self.kl_b = self.core.kl_b

    def forward(self, x, y, e):
        # f-f
        _, y_, _ = self.core(x, None, e, fpd="f")

        y = y_ if True in torch.isnan(y) else y.squeeze(dim=2)

        x_, _, _ = self.core(None, y, e, fpd="b")
        return x_, y_, _


class PepperNetInv(nn.Module):
    def __init__(self, hparam, in_, out_):
        super(PepperNetInv, self).__init__()
        self.hparam = hparam
        self.in_, self.out_ = in_, out_

        # modules
        self.core = m.PepperInvCore(hparam=hparam, in_=in_, out_=out_)

    def forward(self, x, y, e):
        # f-f
        _, y_, z_ = self.core(x, None, e, fpd="f")

        # if y != NaN continue with y.squeeze(dim=2)
        y = y_ if True in torch.isnan(y) else y.squeeze(dim=2)

        x_, _, _ = self.core(None, y, e, fpd="b")
        return x_, y_, z_


""" *******************************  ordinary conv networks  ****************************************"""

class PepperNetConvCore(nn.Module):
    def __init__(self, hparam, in_, out_):
        super(PepperNetConvCore, self).__init__()
        self.hparam = hparam
        self.in_, self.out_ = in_, out_

        # modules
        self.conv_enc = m.ConvEnc(hparam=hparam, in_=in_)
        enc_out = get_out_shape(self.conv_enc, in_, out_)
        self.core = m.PepperCore(hparam=hparam, in_=enc_out, out_=enc_out)
        self.conv_dec = m.ConvDec(hparam=hparam, in_=enc_out, out_=out_)

    def forward(self, x, y, e):
        #ff
        _, x_enc    = self.conv_enc(x, None, e, fpd="f")
        _, x_enc, _ = self.core(x_enc, None, e, fpd="f")
        _, y_       = self.conv_dec(x_enc, None, e, fpd="f")

        y = y_ if True in torch.isnan(y) else y

        #fb
        y_enc, _    = self.conv_dec(None, y, e, fpd="b")
        y_enc, _, _ = self.core(None, y_enc, e, fpd="b")
        x_, _       = self.conv_enc(None, y_enc, e, fpd="b")
        return x_, y_, _


class PepperNetConvVAE(nn.Module):
    def __init__(self, hparam, in_, out_):
        super(PepperNetConvVAE, self).__init__()
        self.hparam = hparam
        self.in_, self.out_ = in_, out_

        # modules
        self.conv_enc = m.ConvEnc(hparam=hparam, in_=in_)
        enc_out = get_out_shape(self.conv_enc, in_, out_)
        self.core = m.PepperVarCore(hparam=hparam, in_=enc_out, out_=enc_out)
        self.conv_dec = m.ConvDec(hparam=hparam, in_=enc_out, out_=out_)

        self.kl_f = self.core.kl_f
        self.kl_b = self.core.kl_b

    def forward(self, x, y, e):
        #ff
        _, x_enc = self.conv_enc(x, None, e, fpd="f")
        _, x_enc, _ = self.core(x_enc, None, e, fpd="f")
        _, y_    = self.conv_dec(x_enc, None, e, fpd="f")

        y = y_ if True in torch.isnan(y) else y

        #fb
        y_enc, _ = self.conv_dec(None, y, e, fpd="b")
        y_enc, _, _ = self.core(None, y_enc, e, fpd="b")
        x_, _    = self.conv_enc(None, y_enc, e, fpd="b")
        return x_, y_, _


class PepperNetConvInv(nn.Module):
    def __init__(self, hparam, in_, out_):
        super(PepperNetConvInv, self).__init__()
        self.hparam = hparam
        self.in_, self.out_ = in_, out_

        # modules
        self.conv_enc = m.ConvEnc(hparam=hparam, in_=in_)
        enc_out = get_out_shape(self.conv_enc, in_, out_)
        self.core = m.PepperInvCore(hparam=hparam, in_=enc_out, out_=enc_out)
        self.conv_dec = m.ConvDec(hparam=hparam, in_=enc_out, out_=out_)

    def forward(self, x, y, e):
        #ff
        _, x_enc = self.conv_enc(x, None, e, fpd="f")
        _, x_enc, z_ = self.core(x_enc, None, e, fpd="f")
        _, y_    = self.conv_dec(x_enc, None, e, fpd="f")

        y = y_ if True in torch.isnan(y) else y

        #fb
        y_enc, _ = self.conv_dec(None, y, e, fpd="b")
        y_enc, _, _ = self.core(None, y_enc, e, fpd="b")
        x_, _    = self.conv_enc(None, y_enc, e, fpd="b")
        return x_, y_, z_


""" *******************************  strided conv networks  ****************************************"""

class PepperNetSConvCore(nn.Module):
    def __init__(self, hparam, in_, out_):
        super(PepperNetSConvCore, self).__init__()
        self.hparam = hparam
        self.in_, self.out_ = in_, out_

        # modules
        self.conv_enc = m.SConvEnc(hparam=hparam, in_=in_)
        enc_out = get_out_shape(self.conv_enc, in_, out_)
        self.core = m.PepperCore(hparam=hparam, in_=enc_out, out_=enc_out)
        self.conv_dec = m.SConvDec(hparam=hparam, in_=enc_out, out_=out_)

    def forward(self, x, y, e):
        #ff
        _, x_enc = self.conv_enc(x, None, e, fpd="f")
        _, x_enc, _ = self.core(x_enc, None, e, fpd="f")
        _, y_    = self.conv_dec(x_enc, None, e, fpd="f")

        y = y_ if True in torch.isnan(y) else y
        #fb
        y_enc, _ = self.conv_dec(None, y, e, fpd="b")
        y_enc, _, _ = self.core(None, y_enc, e, fpd="b")
        x_, _    = self.conv_enc(None, y_enc, e, fpd="b")
        return x_, y_, _


class PepperNetSConvVAE(nn.Module):
    def __init__(self, hparam, in_, out_):
        super(PepperNetSConvVAE, self).__init__()
        self.hparam = hparam
        self.in_, self.out_ = in_, out_

        # modules
        self.conv_enc = m.SConvEnc(hparam=hparam, in_=in_)
        enc_out = get_out_shape(self.conv_enc, in_, out_)
        self.core = m.PepperVarCore(hparam=hparam, in_=enc_out, out_=enc_out)
        self.conv_dec = m.SConvDec(hparam=hparam, in_=enc_out, out_=out_)

        self.kl_f = self.core.kl_f
        self.kl_b = self.core.kl_b

    def forward(self, x, y, e):
        #ff
        _, x_enc = self.conv_enc(x, None, e, fpd="f")
        _, x_enc, _ = self.core(x_enc, None, e, fpd="f")
        _, y_    = self.conv_dec(x_enc, None, e, fpd="f")

        y = y_ if True in torch.isnan(y) else y
        #fb
        y_enc, _ = self.conv_dec(None, y, e, fpd="b")
        y_enc, _, _ = self.core(None, y_enc, e, fpd="b")
        x_, _    = self.conv_enc(None, y_enc, e, fpd="b")
        return x_, y_, _


class PepperNetSConvInv(nn.Module):
    def __init__(self, hparam, in_, out_):
        super(PepperNetSConvInv, self).__init__()
        self.hparam = hparam
        self.in_, self.out_ = in_, out_

        # modules
        self.conv_enc = m.SConvEnc(hparam=hparam, in_=in_)
        enc_out = get_out_shape(self.conv_enc, in_, out_)
        self.core = m.PepperInvCore(hparam=hparam, in_=enc_out, out_=enc_out)
        self.conv_dec = m.SConvDec(hparam=hparam, in_=enc_out, out_=out_)

    def forward(self, x, y, e):
        #ff
        _, x_enc = self.conv_enc(x, None, e, fpd="f")
        _, x_enc, z_ = self.core(x_enc, None, e, fpd="f")
        _, y_    = self.conv_dec(x_enc, None, e, fpd="f")

        y = y_ if True in torch.isnan(y) else y
        #fb
        y_enc, _ = self.conv_dec(None, y, e, fpd="b")
        y_enc, _, _ = self.core(None, y_enc, e, fpd="b")
        x_, _    = self.conv_enc(None, y_enc, e, fpd="b")
        return x_, y_, z_


""" *******************************  causal conv networks  ****************************************"""

class PepperNetCConvCore(nn.Module):
    def __init__(self, hparam, in_, out_):
        super(PepperNetCConvCore, self).__init__()
        self.hparam = hparam
        self.in_, self.out_ = in_, out_

        # modules
        self.conv_enc = m.CConvEnc(hparam=hparam, in_=in_)
        enc_out = get_out_shape(self.conv_enc, in_, out_)
        self.core = m.PepperCore(hparam=hparam, in_=enc_out, out_=enc_out)
        self.conv_dec = m.CConvDec(hparam=hparam, in_=enc_out, out_=out_)

    def forward(self, x, y, e):
        # ff
        _, x_enc = self.conv_enc(x, None, e, fpd="f")
        _, x_enc, _ = self.core(x_enc, None, e, fpd="f")
        _, y_ = self.conv_dec(x_enc, None, e, fpd="f")

        y = y_ if True in torch.isnan(y) else y

        # fb
        y_enc, _ = self.conv_dec(None, y, e, fpd="b")
        y_enc, _, _ = self.core(None, y_enc, e, fpd="b")
        x_, _ = self.conv_enc(None, y_enc, e, fpd="b")
        return x_, y_, _


class PepperNetCConvVAE(nn.Module):
    def __init__(self, hparam, in_, out_):
        super(PepperNetCConvVAE, self).__init__()
        self.hparam = hparam
        self.in_, self.out_ = in_, out_

        # modules
        self.conv_enc = m.CConvEnc(hparam=hparam, in_=in_)
        enc_out = get_out_shape(self.conv_enc, in_, out_)
        self.core = m.PepperVarCore(hparam=hparam, in_=enc_out, out_=enc_out)
        self.conv_dec = m.CConvDec(hparam=hparam, in_=enc_out, out_=out_)

        self.kl_f = self.core.kl_f
        self.kl_b = self.core.kl_b

    def forward(self, x, y, e):
        #ff
        _, x_enc = self.conv_enc(x, None, e, fpd="f")
        _, x_enc, _ = self.core(x_enc, None, e, fpd="f")
        _, y_    = self.conv_dec(x_enc, None, e, fpd="f")

        y = y_ if True in torch.isnan(y) else y

        #fb
        y_enc, _ = self.conv_dec(None, y, e, fpd="b")
        y_enc, _, _ = self.core(None, y_enc, e, fpd="b")
        x_, _    = self.conv_enc(None, y_enc, e, fpd="b")
        return x_, y_, _


class PepperNetCConvInv(nn.Module):
    def __init__(self, hparam, in_, out_):
        super(PepperNetCConvInv, self).__init__()
        self.hparam = hparam
        self.in_, self.out_ = in_, out_

        # modules
        self.conv_enc = m.CConvEnc(hparam=hparam, in_=in_)
        enc_out = get_out_shape(self.conv_enc, in_, out_)
        self.core = m.PepperInvCore(hparam=hparam, in_=enc_out, out_=enc_out)
        self.conv_dec = m.CConvDec(hparam=hparam, in_=enc_out, out_=out_)

    def forward(self, x, y, e):
        #ff
        _, x_enc = self.conv_enc(x, None, e, fpd="f")
        _, x_enc, z_ = self.core(x_enc, None, e, fpd="f")
        _, y_    = self.conv_dec(x_enc, None, e, fpd="f")

        y = y_ if True in torch.isnan(y) else y

        #fb
        y_enc, _ = self.conv_dec(None, y, e, fpd="b")
        y_enc, _, _ = self.core(None, y_enc, e, fpd="b")
        x_, _    = self.conv_enc(None, y_enc, e, fpd="b")
        return x_, y_, z_


""" ********************************  multi net networks  ****************************************"""

class PepperNetMultiCore2(nn.Module):
    def __init__(self, hparam, in_, out_, n_):
        super(PepperNetMultiCore2, self).__init__()
        self.hparam = hparam
        in_, out_, n_ = in_, out_[:, 0], n_

        # modules
        self.conv_enc = m.CConvEnc(hparam=hparam, in_=in_)
        enc_out = get_out_shape(self.conv_enc, in_, out_)
        self.core1 = m.PepperCore(hparam=hparam, in_=enc_out, out_=out_)
        self.core2 = m.PepperCore(hparam=hparam, in_=enc_out, out_=out_)

    def forward(self, x, y, e):
        #ff
        _, x_enc = self.conv_enc(x, None, e, fpd="f")

        _, y1_, _ = self.core1(x_enc, None, e, fpd="f")
        _, y2_, _ = self.core2(x_enc, None, e, fpd="f")
        y_ = torch.stack((y1_, y2_), dim=1)

        if True in torch.isnan(y):
            y1, y2 = y1_, y2_
        else:
            y1, y2 = y[:, 0], y[:, 1]

        #fb
        y1_enc, _, _ = self.core1(None, y1, e, fpd="b")
        y2_enc, _, _ = self.core2(None, y2, e, fpd="b")

        y_enc = (y1_enc + y2_enc) / 2
        x_, _ = self.conv_enc(None, y_enc, e, fpd="b")

        # TODO check network output

        return x_, y_, None


class PepperNetMultiVAE2(nn.Module):
    def __init__(self, hparam, in_, out_, n_):
        super(PepperNetMultiVAE2, self).__init__()
        self.hparam = hparam
        in_, out_, n_ = in_, out_[:, 0], n_

        # modules
        self.conv_enc = m.CConvEnc(hparam=hparam, in_=in_)
        enc_out = get_out_shape(self.conv_enc, in_, out_)
        self.core1 = m.PepperVarCore(hparam=hparam, in_=enc_out, out_=out_)
        self.core2 = m.PepperVarCore(hparam=hparam, in_=enc_out, out_=out_)

        self.kl_f1, self.kl_f2 = self.core1.kl_f, self.core2.kl_f
        self.kl_b1, self.kl_b2 = self.core1.kl_b, self.core2.kl_b
        self.kl_f, self.kl_b = self.kl_f1 + self.kl_f2, self.kl_b1 + self.kl_b2

    def forward(self, x, y, e):
        #ff
        _, x_enc = self.conv_enc(x, None, e, fpd="f")

        _, y1_, _ = self.core1(x_enc, None, e, fpd="f")
        _, y2_, _ = self.core2(x_enc, None, e, fpd="f")
        y_ = torch.stack((y1_, y2_), dim=1)

        if True in torch.isnan(y):
            y1, y2 = y1_, y2_
        else:
            y1, y2 = y[:, 0], y[:, 1]

        #fb
        y1_enc, _, _ = self.core1(None, y1, e, fpd="b")
        y2_enc, _, _ = self.core2(None, y2, e, fpd="b")

        y_enc = (y1_enc + y2_enc) / 2
        x_, _ = self.conv_enc(None, y_enc, e, fpd="b")
        return x_, y_, None


class PepperNetMultiInv2(nn.Module):
    def __init__(self, hparam, in_, out_, n_):
        super(PepperNetMultiInv2, self).__init__()
        self.hparam = hparam
        in_, out_, n_ = in_, out_[:, 0], n_

        # modules
        self.conv_enc = m.CConvEnc(hparam=hparam, in_=in_)
        enc_out = get_out_shape(self.conv_enc, in_, out_)
        self.core1 = m.PepperInvCore(hparam=hparam, in_=enc_out, out_=out_)
        self.core2 = m.PepperInvCore(hparam=hparam, in_=enc_out, out_=out_)

    def forward(self, x, y, e):
        #ff
        _, x_enc = self.conv_enc(x, None, e, fpd="f")

        _, y1_, z1_ = self.core1(x_enc, None, e, fpd="f")
        _, y2_, z2_ = self.core2(x_enc, None, e, fpd="f")
        y_ = torch.stack((y1_, y2_), dim=1)

        """"""
        if True in torch.isnan(y):
            y1, y2 = y1_, y2_
        else:
            y1, y2 = y[:, 0], y[:, 1]

        #fb
        y1_enc, _, _ = self.core1(None, y1, e, fpd="b")
        y2_enc, _, _ = self.core2(None, y2, e, fpd="b")

        y_enc = (y1_enc + y2_enc) / 2
        x_, _ = self.conv_enc(None, y_enc, e, fpd="b")
        return x_, y_, (z1_ + z2_)


class PepperNetMultiCore3(nn.Module):
    def __init__(self, hparam, in_, out_, n_):
        super(PepperNetMultiCore3, self).__init__()
        self.hparam = hparam
        in_, out_, n_ = in_, out_[:, 0], n_

        # modules
        self.conv_enc = m.CConvEnc(hparam=hparam, in_=in_)
        enc_out = get_out_shape(self.conv_enc, in_, out_)
        self.core1 = m.PepperCore(hparam=hparam, in_=enc_out, out_=out_)
        self.core2 = m.PepperCore(hparam=hparam, in_=enc_out, out_=out_)
        self.core3 = m.PepperCore(hparam=hparam, in_=enc_out, out_=out_)

    def forward(self, x, y, e):
        #ff
        _, x_enc = self.conv_enc(x, None, e, fpd="f")

        _, y1_, _ = self.core1(x_enc, None, e, fpd="f")
        _, y2_, _ = self.core2(x_enc, None, e, fpd="f")
        _, y3_, _ = self.core3(x_enc, None, e, fpd="f")
        y_ = torch.stack((y1_, y2_, y3_), dim=1)

        if True in torch.isnan(y):
            y1, y2, y3 = y1_, y2_, y3_
        else:
            y1, y2, y3 = y[:, 0], y[:, 1], y[:, 2]

        #fb
        y1_enc, _, _ = self.core1(None, y1, e, fpd="b")
        y2_enc, _, _ = self.core2(None, y2, e, fpd="b")
        y3_enc, _, _ = self.core3(None, y3, e, fpd="b")

        y_enc = (y1_enc + y2_enc + y3_enc) / 3
        x_, _ = self.conv_enc(None, y_enc, e, fpd="b")
        return x_, y_, None


class PepperNetMultiVAE3(nn.Module):
    def __init__(self, hparam, in_, out_, n_):
        super(PepperNetMultiVAE3, self).__init__()
        self.hparam = hparam
        in_, out_, n_ = in_, out_[:, 0], n_

        # modules
        self.conv_enc = m.CConvEnc(hparam=hparam, in_=in_)
        enc_out = get_out_shape(self.conv_enc, in_, out_)
        self.core1 = m.PepperVarCore(hparam=hparam, in_=enc_out, out_=out_)
        self.core2 = m.PepperVarCore(hparam=hparam, in_=enc_out, out_=out_)
        self.core3 = m.PepperVarCore(hparam=hparam, in_=enc_out, out_=out_)

        self.kl_f1, self.kl_f2, self.kl_f3 = self.core1.kl_f, self.core2.kl_f, self.core3.kl_f
        self.kl_b1, self.kl_b2, self.kl_b3 = self.core1.kl_b, self.core2.kl_b, self.core3.kl_b
        self.kl_f, self.kl_b = self.kl_f1 + self.kl_f2 + self.kl_f3, self.kl_b1 + self.kl_b2 + self.kl_b3

    def forward(self, x, y, e):
        #ff
        _, x_enc = self.conv_enc(x, None, e, fpd="f")

        _, y1_, _ = self.core1(x_enc, None, e, fpd="f")
        _, y2_, _ = self.core2(x_enc, None, e, fpd="f")
        _, y3_, _ = self.core1(x_enc, None, e, fpd="f")
        y_ = torch.stack((y1_, y2_, y3_), dim=1)

        if True in torch.isnan(y):
            y1, y2, y3 = y1_, y2_, y3_
        else:
            y1, y2, y3 = y[:, 0], y[:, 1], y[:, 2]

        #fb
        y1_enc, _, _ = self.core1(None, y1, e, fpd="b")
        y2_enc, _, _ = self.core2(None, y2, e, fpd="b")
        y3_enc, _, _ = self.core3(None, y3, e, fpd="b")

        y_enc = (y1_enc + y2_enc + y3_enc) / 3
        x_, _ = self.conv_enc(None, y_enc, e, fpd="b")
        return x_, y_, None


class PepperNetMultiInv3(nn.Module):
    def __init__(self, hparam, in_, out_, n_):
        super(PepperNetMultiInv3, self).__init__()
        self.hparam = hparam
        in_, out_, n_ = in_, out_[:, 0], n_

        # modules
        self.conv_enc = m.CConvEnc(hparam=hparam, in_=in_)
        enc_out = get_out_shape(self.conv_enc, in_, out_)
        self.core1 = m.PepperInvCore(hparam=hparam, in_=enc_out, out_=out_)
        self.core2 = m.PepperInvCore(hparam=hparam, in_=enc_out, out_=out_)
        self.core3 = m.PepperInvCore(hparam=hparam, in_=enc_out, out_=out_)

    def forward(self, x, y, e):
        #ff
        _, x_enc = self.conv_enc(x, None, e, fpd="f")

        _, y1_, z1_ = self.core1(x_enc, None, e, fpd="f")
        _, y2_, z2_ = self.core2(x_enc, None, e, fpd="f")
        _, y3_, z3_ = self.core1(x_enc, None, e, fpd="f")
        y_ = torch.stack((y1_, y2_, y3_), dim=1)

        """"""
        if True in torch.isnan(y):
            y1, y2, y3 = y1_, y2_, y3_
        else:
            y1, y2, y3 = y[:, 0], y[:, 1], y[:, 2]

        #fb
        y1_enc, _, _ = self.core1(None, y1, e, fpd="b")
        y2_enc, _, _ = self.core2(None, y2, e, fpd="b")
        y3_enc, _, _ = self.core3(None, y3, e, fpd="b")

        y_enc = (y1_enc + y2_enc + y3_enc) / 3
        x_, _ = self.conv_enc(None, y_enc, e, fpd="b")
        return x_, y_, (z1_ + z2_ + z3_)