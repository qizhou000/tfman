#%%
import math 
import torch
import torch.nn as nn 
   
def default_conv(in_c, out_c, k_size, stride = 1, bias = True):
  return nn.Conv2d(in_c, out_c, k_size,
    padding = k_size // 2, stride = stride, bias = bias)

class MeanShift(nn.Module):
  def __init__(self, rgb_range, rgb_mean, sign=-1, device = 'cuda'): 
    super(MeanShift,self).__init__()
    mean = torch.tensor(rgb_mean, device=device).reshape([1,3,1,1])
    mean = mean * (rgb_range[1] - rgb_range[0]) + rgb_range[0] 
    self.mean = sign * mean
  def forward(self, x): 
    return x + self.mean.to(x)
   
class BasicBlock(nn.Sequential):
  def __init__(self, conv, in_c, out_c, k_size, bias=True, act=nn.PReLU()):
    super().__init__(conv(in_c, out_c, k_size, bias=bias), act)

class ResBlock(nn.Module):
  def __init__(self, conv, n_feats, kernel_size, bias=True, act=nn.PReLU(), res_scale=1):
    super().__init__() 
    self.body = nn.Sequential(
      conv(n_feats, n_feats, kernel_size, bias=bias),act, 
      conv(n_feats, n_feats, kernel_size, bias=bias) 
    )
    self.res_scale = res_scale

  def forward(self, x): 
    return self.body(x) * self.res_scale + x

 
class Add_bias(nn.Module):
  def __init__(self, shape):
    super().__init__() 
    self.my_bias = nn.Parameter(torch.zeros(shape))
  def forward(self, x):
    return x + self.my_bias

class Upsampler(nn.Sequential):
  def __init__(self, scale, n_feat,ks=3, pro=True, post_bias=True): 
    m = []
    if pro and (scale & (scale - 1)) == 0:    # Is scale = 2^n? 
      for _ in range(int(math.log(scale, 2))):
        m.append(nn.Conv2d(n_feat, 4*n_feat, ks, 1, 1, bias = not post_bias))
        m.append(nn.PixelShuffle(2))  
        if post_bias:   
          m.append(Add_bias([1,n_feat,1,1]))
    else:
      m.append(nn.Conv2d(n_feat, scale**2*n_feat, ks, 1, 1, bias = not post_bias))
      m.append(nn.PixelShuffle(scale))    
      if post_bias:   
        m.append(Add_bias([1,n_feat,1,1])) 
    super().__init__(*m) 
    
class Upsampler_deconv(nn.Sequential):
  def __init__(self, scale, n_feat, ks=3, pro=False): 
    m = []
    if pro and (scale & (scale - 1)) == 0:    # Is scale = 2^n? 
      for _ in range(int(math.log(scale, 2))):
        m.append(nn.ConvTranspose2d(n_feat, n_feat, 2*ks, 2, 2))
    else:
      m.append(nn.ConvTranspose2d(n_feat, n_feat, scale*ks, scale, scale))
    super().__init__(*m) 
    
class Downsampler(nn.Sequential):
  def __init__(self, scale, n_feat, ks=3, pro=True): 
    m = []
    if pro and (scale & (scale - 1)) == 0:    # Is scale = 2^n? 
      for _ in range(int(math.log(scale, 2))):
        m.append(nn.Conv2d(n_feat, n_feat, 2*ks, 2, 2)) 
    else:
      m.append(nn.Conv2d(n_feat, n_feat, scale*ks, scale, scale))
    super().__init__(*m) 