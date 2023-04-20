import torch
import torch.nn as nn
from utils import utils


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( utils.nz, utils.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(utils.ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(utils.ngf * 8, utils.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(utils.ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( utils.ngf * 4, utils.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(utils.ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( utils.ngf * 2, utils.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(utils.ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( utils.ngf, utils.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
    

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(utils.nc, utils.ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(utils.ndf, utils.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(utils.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(utils.ndf * 2, utils.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(utils.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(utils.ndf * 4, utils.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(utils.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(utils.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)