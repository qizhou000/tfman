#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
 

class MShift(nn.Module):
    def __init__(self, color_range, rgb_mean, rgb_std, sign=-1):
        super(MShift, self).__init__()
        self.color_range = color_range
        self.conv = nn.Conv2d(3,3,1)
        std = torch.Tensor(rgb_std)
        self.conv.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1) 
        self.conv.bias.data = sign * torch.Tensor(rgb_mean) / std  
    def forward(self, x):
        x = (x + self.color_range[0])/(self.color_range[1]-self.color_range[0])
        x = self.conv(x)
        return x
  
class VGG(nn.Module):
    def __init__(self, conv_index, color_range, precision):
        super(VGG, self).__init__() 
        if conv_index == '22':
            self.vgg = models.vgg19(pretrained=True).features[:8]
        elif conv_index == '54':
            self.vgg = models.vgg19(pretrained=True).features[:35]  
        else:
            raise ValueError
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229, 0.224, 0.225)
        self.sub_mean = MShift(color_range, vgg_mean, vgg_std)
        self.requires_grad_(False)
        self.to('cuda')
        if precision == 'half':
            self.to(dtype=torch.float16) 
    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x 
        vgg_sr = _forward(sr) 
        vgg_hr = _forward(hr) 
        loss = F.mse_loss(vgg_sr, vgg_hr) 
        return loss 