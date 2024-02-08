import torch
import torch.nn as nn 

class SRModel(nn.Module):
    def __init__(self, model, scale_factor, chop, self_ensemble, device = 'cuda'):
        super(SRModel, self).__init__() 

        self.scale = scale_factor
        self.self_ensemble = self_ensemble 
        self.chop = chop
        self.precision = 'single'  
        self.device = device
        self.model = model.to(self.device) 
        
        if self.precision == 'half': 
            self.model.half()
    
    def forward(self, x):      
        if self.self_ensemble and not self.training: 
            if self.chop:   
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward 
            return self.forward_x8(x, forward_function)
        elif self.chop and not self.training:
            return self.forward_chop(x)
        else: 
            return self.model(x)
 
  
    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output


    def forward_chop(self, x, shave=10, min_size=3600):  
        scale = self.scale 
        n_GPUs = 1
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        h_size += scale-h_size%scale
        w_size += scale-w_size%scale
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

         