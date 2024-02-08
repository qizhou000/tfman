import utility, torch 
from utility import Checker

class Trainer():
    def __init__(self, args, loader_train, my_model, my_loss, checker:Checker):
        self.args = args 

        self.checker = checker
        self.loader_train = loader_train 
        self.loss = my_loss
        self.model = my_model
        self.optimizer = utility.make_optimizer(args, my_model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
     
    def train(self, start_t, start_epoch): 
        t = start_t
        scale = self.args.scales[0]
        save_psnr = {2:35, 3:32, 4:30, 8:26}[scale]   # Only save when psnr higher than this
        max_psnr = 0
        print('Start training...')
        for epoch in range(start_epoch, self.args.epochs): # epochs
            print('epoch:', epoch) 
            self.checker.log({
                'Learning rate': [self.scheduler.get_last_lr()[0], t],
                'Epoch': [epoch, t]
            })  
            self.model.train() 
            for hr, lr, hr_size in self.loader_train:  # one epoch
                t += 1
                l = 0
                hr = torch.split(hr.to('cuda'), self.args.batch_size//self.args.batch_divide_n, 0) 
                lr = torch.split(lr[1/scale].to('cuda'), self.args.batch_size//self.args.batch_divide_n, 0)
                for k in range(self.args.batch_divide_n): # one iteration divide into several
                    sr = self.model(lr[k])
                    loss = self.loss(sr, hr[k]) / self.args.batch_divide_n
                    if loss > 99999:
                        print('Drop too large loss: ', loss)
                        continue
                    l += loss.detach()
                    loss.backward()
                self.checker.log({'Loss': [l, t]})   
                self.optimizer.step()
                self.optimizer.zero_grad() 
            self.scheduler.step() 
            
            if epoch % 2000 == 0 and epoch != 0:# Test all testing dataset
                print('Testing mean PSNR...')
                self.checker.test(self.model, t, None)
            elif epoch % self.args.test_every == 0:# Test the first testing dataset
                print('Testing...')
                psnr,_,_,_ = self.checker.test(self.model, t, 1)
                if psnr > save_psnr and  psnr > max_psnr+0.004:
                    max_psnr = psnr
                    self.checker.save_states(self, psnr, epoch, t)
                elif epoch % self.args.save_every == 0:
                    self.checker.save_states(self, psnr, epoch, t)
 