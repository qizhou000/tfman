import torch.nn as nn  
from loss.vgg import VGG

class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__() 
        self.losses = {} 
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                self.losses[loss_type] = [float(weight), nn.MSELoss()] 
            elif loss_type == 'L1':
                self.losses[loss_type] = [float(weight), nn.L1Loss()]  
            elif loss_type.find('VGG') >= 0:  
                loss_function = VGG(
                    loss_type[3:],
                    color_range = args.color_range,
                    precision = args.precision
                )  
                self.losses[loss_type] = [float(weight), loss_function]   
            self.losses[loss_type][1].to('cuda')
      
    def forward(self, sr, hr):
        loss = 0 
        for k in self.losses.keys():
            loss += self.losses[k][0] * self.losses[k][1](sr, hr)   
        return loss
 