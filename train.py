import utility
from trainer import Trainer
from data import SRDataYielder 
from utility import load_states   
from loss import Loss
from models import SRModel
from models.tfman import TFMAN

class args:
    # TFMAN model setting
    scales = [2]       # Scale factor: 2/3/4/8
    depth = 12         # Depth of TFMAN
    n_feats = 128      # Middle feature dimension
    pre_train = None   # Pre-trained checkpoint path

    # Data setting
    train_dir = 'SRTrain/DIV2K'                 # Directory of training dataset
    downsample_way = 'bicubic'                  # Degradation model: bicubic/BD/DN
    train_data_name = train_dir.split('/')[-1]  # Dataset name, used for processing dataset
    n_train = 800                               # Count of images used for training
    data_aug = 1                                # Data augmentation for training data. None: No, 1: rotation and mirror
    test_dirs = {                               # Datasets for evaluation during training 
        'Set5': 'SRTest/Set5',
    }                               
    rgb_range = (0, 1)                          # Value range of RGB
    n_colors = 3                                # Number of color channels
    patch_size = 48 * scales[0]                 # Size of HR image patch for training 
    data_load2mem = True                        # Load data into memory. Else dynamically load during training

    # Training setting
    optimizer = 'ADAM'      # 'SGD', 'ADAM', 'RMSprop'    
    momentum = 0.9          # Optimizer momentum
    beta1 = 0.9             # ADAM beta1
    beta2 = 0.999           # ADAM beta2
    epsilon = 1e-8          # ADAM epsilon for numerical stability 
    weight_decay = 0        # Weight decay regularization
    loss = '1*L1'           # Loss function configuration   
    gamma = 0.5             # learning rate decay factor for step decay
    lr_decay = [5000, 8500, 10500, 11500, 12500, 13500]  # Epochs when step decay
    epochs = 15000          # Number of epochs to train
    batch_size = 16         # Batch size
    batch_divide_n = 4      # Split a batch of training data into multistep to accumulate gradient
    lr = 1e-4               # learning rate
    epsilon = 5e-8          # Prevent zero division
    device = 'cuda:0'

    # Inference setting
    self_ensemble = False   # Use self-ensemble for testing
    chop = False            # Chop image into small patches before inference

    # Record setting
    save_every = 10         # Every epochs saving checkpoint 
    test_every = 10         # Every epochs testing and recording 

if __name__ == '__main__':
    train_yielder = SRDataYielder(args, True, args.train_dir, args.train_data_name, 
        args.data_load2mem, args.downsample_way)  # Training dataset yielder
    test_yielders = [SRDataYielder(args, False, args.test_dirs[k], k, 
        args.data_load2mem, args.downsample_way) for k in args.test_dirs.keys()] # Testing datasets yielder  
    checker = utility.Checker(args, test_yielders, '',
                    'train_%s_x%d'%(args.downsample_way, args.scales[0]))
    model = SRModel(TFMAN(args), args.scales[0], args.chop, args.self_ensemble, args.device) 
    loss = Loss(args)  
    trainer = Trainer(args, train_yielder, model, loss, checker)  

    # Load checkpoint and train 
    t, psnr, epoch = load_states(model, trainer.optimizer, trainer.scheduler, 
                        args.pre_train, True, True)  
    trainer.train(t, epoch) 

 