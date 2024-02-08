import torch.nn as nn
import torch
from models.modules import CA,SRNL,TFM
from models.common import Upsampler_deconv, Downsampler
from models.common import default_conv, BasicBlock, MeanShift
def make_model(args, parent=False):
    return TFMAN(args)

class Fusion1(nn.Module):
    def __init__(self, in_channel, kernel_size = 3,scale=2, conv = default_conv, depth = 12):
        super().__init__() 
        self.tfm = TFM(in_channel,scale = scale, depth=depth)
        self.ca = CA(in_channel)
        self.srnl = SRNL(in_channel)
        pro_up = True if scale in [8,16] else False
        self.upsample = nn.Sequential(Upsampler_deconv(scale, in_channel,kernel_size, pro_up), nn.PReLU())
        self.encoder = nn.Sequential(
            conv(in_channel, in_channel, kernel_size, bias=True),nn.PReLU(), 
            conv(in_channel, in_channel, kernel_size, bias=True) 
        )
    def forward(self, x, i):
        b1 = self.ca(self.tfm(x, i)) 
        b2 = self.upsample(self.srnl(x)) 
        x_fuse = self.encoder(b1-b2) + b1
        return x_fuse 

class FMF(nn.Module):
    def __init__(self, in_channel, kernel_size = 3, scale = 2, conv = default_conv, depth = 12):
        super().__init__()
        self.fusion1 = Fusion1(in_channel,kernel_size=kernel_size,scale = scale, depth=depth)
        self.downsample = nn.Sequential(Downsampler(scale,in_channel,kernel_size), nn.PReLU())
        pro_up = True if scale in [8,16] else False
        self.encoder = nn.Sequential(Upsampler_deconv(scale,in_channel,kernel_size, pro_up), nn.PReLU())
        self.proj_back = nn.Sequential(
            Downsampler(scale,in_channel,kernel_size), nn.PReLU(),
            BasicBlock(conv, in_channel, in_channel, kernel_size, act=nn.PReLU()) 
        ) 
 
    def forward(self, x, i):
        x_fuse = self.fusion1(x, i)  
        x_up = x_fuse + self.encoder(x - self.downsample(x_fuse))
        x_next = self.proj_back(x_up)
        return x_next, x_up 

class TFMAN(nn.Module):
    def __init__(self, args, conv=default_conv):
        super().__init__() 
        n_feats = args.n_feats
        self.depth = args.depth
        kernel_size = 3 
        scale = args.scales[0]       

        rgb_mean = (0.4488, 0.4371, 0.4040) 
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, -1) 
        self.head = nn.Sequential(
            BasicBlock(conv, args.n_colors, n_feats, kernel_size, act=nn.PReLU()),
            BasicBlock(conv, n_feats, n_feats, kernel_size, act=nn.PReLU())
        ) 
        self.fmf = FMF(n_feats, scale = scale, depth = args.depth) 
        self.tail = nn.Sequential(nn.Conv2d(n_feats*self.depth, args.n_colors, kernel_size, padding=kernel_size//2)) 
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, 1)
   
    def forward(self, x):  
        x = self.head(self.sub_mean(x)) 
        bag = []
        for i in range(self.depth):
            x, x_up = self.fmf(x, i)
            bag.append(x_up)  
        return self.add_mean(self.tail(torch.cat(bag,dim=1)))
 