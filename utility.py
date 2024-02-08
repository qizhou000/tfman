#%%  
from torch.utils.tensorboard import SummaryWriter 
from data import SRDataYielder
import os, math, datetime, torch 
import torch.optim.lr_scheduler as lrs
from torch import nn,optim
from torchvision.utils import save_image
import cv2
import numpy as np

class Checker():
    def __init__(self, args, loaders_test:list, record_name='', log_name = ''): 
        self.loaders_test = loaders_test
        self.args = args 
        # Create folders 
        t = datetime.datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
        if record_name == '':
            record_name = t
        self.save_point_path = os.path.join('records',record_name,'save_points',t if log_name == '' else log_name)
        self.logs_path = os.path.join('records',record_name,'logs', t if log_name == '' else log_name)
        self.sr_imgs_path = os.path.join('records',record_name,'sr_Images')
        if not os.path.exists(self.save_point_path):
            os.makedirs(self.save_point_path)
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
        if not os.path.exists(self.sr_imgs_path):
            os.makedirs(self.sr_imgs_path)
        # Record model information
        self.model_info_path = os.path.join('records', record_name, 'parameters.txt')
        self.write_model_info(self.model_info_path)
        # 
        self.log_writer = SummaryWriter(self.logs_path)
 

    def write_model_info(self, path): 
        d = vars(self.args)
        with open(path, 'w') as f:
            for k in d:
                f.write(k+": "+str(d[k])+"\n")
     
    def save_states(self, trainer, psnr, epoch, t):
        save_name = str(t)+'--'+datetime.datetime.now().strftime('%Y.%m.%d-%H.%M.%S')+'--'+str(psnr)
        path = os.path.join(self.save_point_path, save_name)
        torch.save({
            't': t,
            'psnr': psnr,
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(), 
        }, path) 

    def log(self, items:dict): 
        for k in items.keys(): 
            self.log_writer.add_scalar(k, items[k][0], items[k][1]) 

    def save_sr_img(self, sr_img, t, img_name, scale, dataset_name):
        '''
        Color value \in [0, 255]
        '''
        time = datetime.datetime.now().strftime('%Y.%m.%d-%H.%M.%S') 
        img_dir = os.path.join(self.sr_imgs_path, 'x%d'%scale, dataset_name)
        if not os.path.exists(img_dir): os.makedirs(img_dir)
        path = os.path.join(img_dir, str(t)+'-('+str(img_name)+')-'+time+'.png') 
        save_image(
            tensor = sr_img,
            fp = path,
            normalize=True, 
            range=(0,255)) 
   
    def test(self, model, t, test_n = None, save_imgs = False, verbose = False): 
        scale = self.args.scales[0]
        model.eval() 
        with torch.no_grad(): 
            psnrs, ssims = {}, {}
            mean_psnr, mean_ssim = 0, 0
            for loader in self.loaders_test[:test_n]:
                acc_psnr, acc_ssim = 0, 0 
                for i, (hr, lr, hr_size) in enumerate(loader): 
                    hr, lr = hr.to('cuda'), lr[1/scale].to('cuda')

                    sr = model(lr)
                    sr = self.normalize_color(sr) 
                    hr = self.normalize_color(hr)  
                    ps = self.calc_psnr(sr.clone(), hr.clone())  
                    si = self.calc_ssim(sr.clone(), hr.clone())  
                    if verbose:
                        print(i, ps, si)
                    acc_psnr, acc_ssim = acc_psnr+ps, acc_ssim+si
                    if save_imgs: 
                        self.save_sr_img(sr, t, '%s-%.3f'%(i, ps), model.scale, loader.dataset_name)
                acc_psnr, acc_ssim = acc_psnr/len(loader), acc_ssim/len(loader)
                psnrs[loader.dataset_name], ssims[loader.dataset_name] = acc_psnr, acc_ssim
                self.log({
                    'PSNR-'+loader.dataset_name: [acc_psnr, t],
                    'SSIM-'+loader.dataset_name: [acc_ssim, t]}) 
                mean_psnr, mean_ssim = mean_psnr+acc_psnr, mean_ssim+acc_ssim
            mean_psnr /= len(self.loaders_test[:test_n])
            mean_ssim /= len(self.loaders_test[:test_n])
            if test_n != 1: 
                self.log({'PSNR-Mean': [mean_psnr, t], 'SSIM-Mean': [mean_ssim, t]}) 
        return mean_psnr, psnrs, mean_ssim, ssims

    def normalize_color(self, img):
        '''
        Normalize color value into [0, 255]
        '''
        img = (img-self.args.rgb_range[0])/(self.args.rgb_range[1]-self.args.rgb_range[0])*255 
        return img.clip(0, 255).round()

    def transform2Y(self,rgb_img):
        convert = rgb_img.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        rgb_img.mul_(convert).div_(256)
        return rgb_img.sum(dim=1, keepdim=True) 

    def calc_psnr(self, sr, hr):
        '''
        Color value must \in [0,255].
        '''
        scale = self.args.scales[0]
        hr = hr[:,:,:hr.shape[2]//scale*scale,:hr.shape[3]//scale*scale]
        diff = (sr - hr).data.div(255) 
        if diff.size(1) > 1:
            diff = self.transform2Y(diff) 
        valid = diff[:, :, scale:-scale, scale:-scale]
        mse = valid.pow(2).mean()
        psnr = -10 * math.log10(mse)
        return psnr

    def calc_ssim(self, sr, hr):
        '''
        Color value must \in [0,255].
        '''
        scale = self.args.scales[0]
        hr = hr[:,:,:hr.shape[2]//scale*scale,:hr.shape[3]//scale*scale]
        if hr.size(1) > 1:
            hr = self.transform2Y(hr)
        if sr.size(1) > 1:
            sr = self.transform2Y(sr)
        hr = hr[0, 0, scale:-scale, scale:-scale].cpu().numpy()
        sr = sr[0, 0, scale:-scale, scale:-scale].cpu().numpy()
        ssim = calculate_ssim(hr, sr)
        return ssim

def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay
    
    return optimizer_function(trainable, **kwargs)

def make_scheduler(args, my_optimizer):
    if type(args.lr_decay) == int:
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif type(args.lr_decay) == list:  
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=args.lr_decay,
            gamma=args.gamma
        )

    return scheduler

def load_states(model, optimizer, scheduler, path, continue_train, strict=True): 
    if not path:
        return 0,0,0 
    checkpoint = torch.load(path) 
    if type(checkpoint) == dict: 
        psnr = checkpoint['psnr'] 
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        if continue_train:
            t = checkpoint['t']  
            epoch = checkpoint['epoch'] 
            if optimizer != None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])   
            if scheduler != None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            t = epoch = 0
        return t, psnr, epoch
    else:  
        model.model.load_state_dict(checkpoint, strict=strict)  
        return 0,0,0


def ssim(img1, img2):
  C1 = (0.01 * 255)**2
  C2 = (0.03 * 255)**2
  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  kernel = cv2.getGaussianKernel(11, 1.5)
  window = np.outer(kernel, kernel.transpose())
  mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # valid
  mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
  mu1_sq = mu1**2
  mu2_sq = mu2**2
  mu1_mu2 = mu1 * mu2
  sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
  sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
  sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
  ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                              (sigma1_sq + sigma2_sq + C2))
  return ssim_map.mean()
def calculate_ssim(img1, img2):
  '''calculate SSIM
  the same outputs as MATLAB's
  img1, img2: [0, 255]
  '''
  img1 = np.double(img1)
  img2 = np.double(img2)
  if not img1.shape == img2.shape:
    raise ValueError('Input images must have the same dimensions.')
  if img1.ndim == 2:
    return ssim(img1, img2)
  elif img1.ndim == 3:
    if img1.shape[2] == 3:
      ssims = []
      for i in range(3):
        ssims.append(ssim(img1, img2))
      return np.array(ssims).mean()
    elif img1.shape[2] == 1:
      return ssim(np.squeeze(img1), np.squeeze(img2))
  else:
    raise ValueError('Wrong input image dimensions.')
 