#%%
import os
from PIL import Image
import numpy as np


def DIV2K_imgs_paths(dir_path, scales=[2], downsample_way='bicubic'):
  assert downsample_way in ['bicubic'] 
  hr_dir = os.path.join(dir_path, 'DIV2K_train_HR')
  lr_dir = [os.path.join(dir_path, 'DIV2K_train_LR_' + downsample_way, 'x'+str(scale)) for scale in scales]
  hr_paths, lr_paths = [], [[] for _ in scales]
  for name in os.listdir(hr_dir):
    if int(name.split('.')[0])>800:
      continue
    hr_paths.append(os.path.join(hr_dir, name)) 
    for i in range(len(scales)):
      lr_paths[i].append(os.path.join(lr_dir[i], name.split('.')[0]+'x'+str(scales[i])+'.png'))
  return hr_paths, lr_paths

def test_imgs_paths(dir_path, scales=[2], downsample_way='bicubic'):
  assert downsample_way in ['bicubic', 'BD', 'DN']
  hr_dir = os.path.join(dir_path, 'HR')
  lr_dir = [os.path.join(dir_path, 'LR',downsample_way, 'x' + str(scale)) for scale in scales]
  hr_paths, lr_paths = [], [[] for _ in scales]
  for name in os.listdir(hr_dir): 
    hr_paths.append(os.path.join(hr_dir, name)) 
    for i in range(len(scales)):
      lr_paths[i].append(os.path.join(lr_dir[i], name))
  return hr_paths, lr_paths

 
  
def prepare_binary(dir_path, downsample_way='bicubic', scales = [0.5], dataset_name = 'DIV2K'): 
  specifued_datasets = {
    'DIV2K': DIV2K_imgs_paths,
    'SET5': test_imgs_paths,
    'SET14': test_imgs_paths,
    'BSD100': test_imgs_paths,
    'URBAN100': test_imgs_paths,
    'MANGA109': test_imgs_paths,
    'DIV2K_BD': test_imgs_paths,
    'DIV2K_DN': test_imgs_paths,
  }
  # make directory
  data_dirs = [os.path.join(dir_path, 'binary', 'HR')]
  for scale in scales: 
    data_dir = os.path.join(dir_path, 'binary', 'LR', downsample_way, 'x%s'%(1/scale)) 
    data_dirs.append(data_dir)
  for dir in data_dirs:
    if not os.path.exists(dir):
      os.makedirs(dir) 

  data_paths = [[] for _ in range(len(scales)+1)] 
  dataset_name = dataset_name.upper()
  print('Preparing [%s] data...'%dataset_name) 

  if dataset_name in specifued_datasets.keys():
    scales = [int(1/scale) for scale in scales] 
    hr_paths, lr_paths = specifued_datasets[dataset_name](dir_path, scales, downsample_way)

    for hr_path in hr_paths: # HR
      img = Image.open(os.path.join(hr_path))
      imgs_name = os.path.basename(hr_path).split('.')[0] 
      data_path = os.path.join(data_dirs[0], imgs_name + '.npy')
      data_paths[0].append(data_path)
      if not os.path.exists(data_path):  
        np.save(data_path, np.array(img)) 
    for i, data_dir in enumerate(data_dirs[1:]): # LR
      for lr_path in lr_paths[i]:
        img = Image.open(os.path.join(lr_path))
        imgs_name = os.path.basename(lr_path).split('.')[0]  
        data_path = os.path.join(data_dir, imgs_name+'.npy')
        data_paths[i+1].append(data_path)
        if not os.path.exists(data_path):   
          np.save(data_path, np.array(img))
  else:
    raise Exception('Dataset "' + dataset_name + '" does not have a processing function!') 
  return data_paths

   