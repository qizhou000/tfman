#%%
import enum
import os, cv2, torch, time, threading
import numpy as np  
from fractions import Fraction
from queue import Queue 
from data.preprocess import prepare_binary
 

class SRDataYielder():
  def __init__(self, args, if_train:bool, dataset_path, dataset_name, is_preload2mem=True, downsample_way = 'bicubic'):
    '''
      `hr_patch_size`: the size of HR image patch. If is `None`, output the 
        entire image and the batchsize would be setted as 1 forcibly.
      `is_preload2mem`: Pre-load the dataset into memory first before yield data.  
        If there is enough memory, this can be setted as `True` to speed up 
        the generation of training data.
    '''    
    # It can be an integer or an interval. If it is an interval, each batch randomly obtains the patch size within it
    self.hr_patch_size = args.patch_size if if_train else None 
    self.batch_size = args.batch_size if if_train else 1
    scales = args.scales if if_train else args.scales[:1]
    self.scales = scales = [1/s for s in scales] # Convert to decimals for easy downsampling 
    self.downsample_way = downsample_way  
    self.data_aug = args.data_aug if if_train else None
    self.total_num = total_num = args.n_train if if_train else 999999 # If it's testing, use all datasets
    self.shuffle = shuffle = True if if_train else False
    self.dataset_path = dataset_path  
    self.dataset_name = dataset_name
    self.buffer_size = 512 if if_train else 32
    self.rgb_range = args.rgb_range
    self.precision = 'single'#args.precision 
    self.data_queue = None # The queue to save trimmed HR and LR patches
    self.is_loading_data = False # Determine if there is already a thread loading data
    self.scales_lcm = None
    self.aug_method = None  # Data augmentation methods 
    self.data_paths = prepare_binary(dataset_path, downsample_way, scales, dataset_name)
    # Limit data volume
    for i in range(len(self.data_paths)):
      self.data_paths[i] = self.data_paths[i][:total_num]  
    # If preloaded, load the dataset into memory first. 
    if is_preload2mem:
      self.__preload2mem()
    self.get_img_by_index = self.get_img_by_index_wrap(is_preload2mem)
    # Calculate the minimum common multiple of cropping
    s = []
    for i in range(len(scales)): 
      s.append(Fraction(scales[i]).limit_denominator().denominator)
    self.scales_lcm = np.lcm.reduce(s) if len(s) > 0 else 1
    # Define data augmentation methods  
    def no_aug():
      return lambda x:x
    def rot_and_flip():
      r1, r2 = np.random.randint(0,4), np.random.choice([-1,1]) 
      def trans(img): 
        return np.rot90(img, r1)[::r2].copy()
      return trans 
    if self.data_aug == None:
      self.aug_method = no_aug
    elif self.data_aug == 1:
      self.aug_method = rot_and_flip  
    # Create queues and data reading threads, and initialize relevant parameters 
    self.data_queue = Queue()
    self.buffer_i = 0 # The number of data pairs currently filled in the queue (each pair contains an HR patch and its LR counterpart)
    self.r_indexes = np.array(range(len(self.data_paths[0])))
    if shuffle:
      np.random.shuffle(self.r_indexes)
    self.__fill_buffer() 

  def __preload2mem(self):
    self.datas = [[] for i in self.data_paths]  
    for i, paths in enumerate(self.data_paths):
      for path in paths:
        self.datas[i].append(np.load(path))
  
  def get_img_by_index_wrap(self, is_preload2mem):
    def gibi_mem(index, set_i):
      return self.datas[set_i][index]
    def gibi_disk(index, set_i):
      return np.load(self.data_paths[set_i][index])
    if is_preload2mem:
      return gibi_mem
    return gibi_disk

  def get_batch(self, indexes):  
    '''Process a batch's training data list by indexes and return it'''
    lcm = self.scales_lcm 
    if self.hr_patch_size == None:
      hr_p_size = 999999
    elif type(self.hr_patch_size) == int:
      hr_p_size = self.hr_patch_size//lcm*lcm 
    else:
      hr_p_size = np.random.randint(self.hr_patch_size[0],self.hr_patch_size[1])//lcm*lcm 
    batch = [[], {s:[] for s in self.scales}, hr_p_size] 
    for id in indexes: 
      hr_img = self.get_img_by_index(id, 0)
      if self.hr_patch_size == None:
        ry,rx = 0,0
      else:
        ry = np.random.randint(0, hr_img.shape[0] - hr_p_size)//lcm*lcm
        rx = np.random.randint(0, hr_img.shape[1] - hr_p_size)//lcm*lcm
      data_aug = self.aug_method() # Initialize data augmentation methods
      # Add HR 
      batch[0].append(data_aug(hr_img[ry:ry+hr_p_size, rx:rx+hr_p_size]))
      # Add LR
      for j in range(len(self.data_paths[1:])):
        scale = self.scales[j]
        size = int(np.around(hr_p_size*scale))
        lry, lrx = int(np.around(ry*scale)), int(np.around(rx*scale))
        lr_img = self.get_img_by_index(id, j+1)
        batch[1][scale].append(data_aug(lr_img[lry:lry+size, lrx:lrx+size]))
    # Splicing data into a tensor and mapping it to the corresponding color range
    batch[0] = torch.tensor(np.stack(batch[0])).permute([0,3,1,2])
    batch[0] = batch[0]/255*(self.rgb_range[1]-self.rgb_range[0])+self.rgb_range[0]
    for key in batch[1]:
      batch[1][key] = torch.tensor(np.stack(batch[1][key])).permute([0,3,1,2])
      batch[1][key] = batch[1][key]/255*(self.rgb_range[1]-self.rgb_range[0])+self.rgb_range[0]
    if self.precision == 'half':
      batch[0] = batch[0].half()
      for k in batch[1].keys():
        batch[1][k] = batch[1][k].half()
    return batch
 
  def __fill_buffer(self):  
    if self.is_loading_data:
      return

    def fill_buffer(): 
      self.is_loading_data = True 
      while self.data_queue.qsize() < self.buffer_size:
        self.buffer_i += self.batch_size
        if self.buffer_i <= len(self.data_paths[0]):
          self.data_queue.put(self.get_batch(self.r_indexes[self.buffer_i-self.batch_size:self.buffer_i]))
        else:
          if self.shuffle:
            np.random.shuffle(self.r_indexes)
          self.buffer_i = 0  
      self.is_loading_data = False  
    threading.Thread(target = fill_buffer).start() 

  def __iter__(self):
    self.iter_i = 0
    return self

  def __next__(self):
    self.iter_i += self.batch_size
    if self.iter_i <= len(self.data_paths[0]): 
      if self.data_queue.qsize() <= self.buffer_size/2:
        self.__fill_buffer() 
      t = 0  
      while self.data_queue.qsize() == 0:  
        print('\r', "Waiting data: %d s"%t, end='')
        time.sleep(1) 
        t+=1  
      return self.data_queue.get()
    else:
      raise StopIteration
      
  def __len__(self):
    return len(self.data_paths[0])