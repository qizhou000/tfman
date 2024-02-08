#%% 
import utility
from models import SRModel
from data import SRDataYielder 
from utility import load_states
from time import time
from models.tfman import TFMAN

class args:
    # TFMAN Model settings
    loss = '1*L1'                   # Loss function selection
    depth = 12                      # Depth of TFMAN
    n_feats = 128                   # Number of feature channels
    scales = [2]                    # Scale factor 
    pre_train = 'checkpoints/BIx2'  # Checkpoint path

    # Inference Settings
    rgb_range = (0,1)           # Value range of RGB  
    n_colors = 3                # Number of color channels 
    chop = False                # Chop image into small patches to inference
    self_ensemble = False       # Use self-ensemble augmentation for testing
    patch_size = 48*scales[0]   # HR image patch size during training
    downsample_way = 'bicubic'  # The down-sample way of testing data: bicubic / BD / DN
    test_dirs = {               # Directory pf testing dataset
        'Set5': 'SRTest/Set5',  
        'Set14': 'SRTest/Set14', 
    }
    device = 'cuda:0'

# Testing
if __name__ == '__main__':
    test_yielders = [SRDataYielder(args, False, args.test_dirs[k], k, 
            downsample_way=args.downsample_way) for k in args.test_dirs.keys()] 
    checker = utility.Checker(args, test_yielders, 'test', 'test0')
    model = SRModel(TFMAN(args), args.scales[0], args.chop, args.self_ensemble, args.device) 
    t, psnr, epoch = load_states(model, None, None, args.pre_train, False, True)
    model.requires_grad_(False)
    t = time()
    mean_psnr, psnrs, mean_ssim, ssims = checker.test(model, 9999, save_imgs=True, verbose=False) 
    print("Inference time:", time()-t)
    print('Mean psnr:', mean_psnr)
    for k in psnrs.keys():
        print('  %s: %f'%(k, psnrs[k]) )
    print('Mean ssim:', mean_ssim)
    for k in ssims.keys():
        print('  %s: %f'%(k, ssims[k]) )

