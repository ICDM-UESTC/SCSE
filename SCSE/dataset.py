import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import torchaudio

import glob
from torch.utils.data.distributed import DistributedSampler
import librosa

from torch.utils.data import SubsetRandomSampler

  
class ConcatDataset(torch.utils.data.Dataset):
  def __init__(self, noisy_root, clean_root):
    super().__init__()
    self.noisy_root = noisy_root
    self.clean_root = clean_root
    self.raw_paths = [x.split('/')[-1] for x in glob.glob(noisy_root + '/*.wav')]
    
    
  def __len__(self):
    return 100
    return len(self.raw_paths)

  def __getitem__(self, index):
    
    raw_paths = self.raw_paths
    noisy, _ = librosa.load(os.path.join(self.noisy_root, raw_paths[index]), sr=16000)  
    clean, _ = librosa.load(os.path.join(self.clean_root, raw_paths[index]), sr=16000) 
    return {
            'noisy_speech': noisy,
            'clean_speech': clean,
        }
  
class TotensorDataset(torch.utils.data.Dataset):
  def __init__(self, noisy_root, clean_root):
    super().__init__()
    self.noisy_root = noisy_root
    self.clean_root = clean_root
    self.raw_paths = [x.split('/')[-1] for x in glob.glob(noisy_root + '/*.wav')]
    
  def __len__(self):
    return 100
    return len(self.raw_paths)

  def __getitem__(self, index):
    
    raw_paths = self.raw_paths
    noisy, _ = librosa.load(os.path.join(self.noisy_root, raw_paths[index]), sr=16000)  
    clean, _ = librosa.load(os.path.join(self.clean_root, raw_paths[index]), sr=16000) 
    return {
          'clean_speech': torch.from_numpy(clean),
          'noisy_speech': torch.from_numpy(noisy),
        }


class Collator:
  def __init__(self, params):
    self.params = params


  def concat_collate(self, minibatch):
    for record in minibatch: 
        if len(record['clean_speech']) < self.params.audio_len:
            start = 0
            end = start + self.params.audio_len
            record['clean_speech'] = np.pad(record['clean_speech'], (0, (end - start) - len(record['clean_speech'])), mode='constant')
            record['noisy_speech'] = np.pad(record['noisy_speech'], (0, (end - start) - len(record['noisy_speech'])), mode='constant')
        start = random.randint(0, record['clean_speech'].shape[-1] - self.params.audio_len)
        end = start + self.params.audio_len
        record['clean_speech'] = record['clean_speech'][start:end]            
        record['noisy_speech'] = record['noisy_speech'][start:end]   
    
    clean_speech = np.stack([record['clean_speech'] for record in minibatch if 'clean_speech' in record])
    noisy_speech = np.stack([record['noisy_speech'] for record in minibatch if 'noisy_speech' in record])
    return {
        'clean_speech': torch.from_numpy(clean_speech),
        'noisy_speech': torch.from_numpy(noisy_speech),
    }
    


def from_path(noisy_root,clean_root, params, is_distributed=False):

    dataset = ConcatDataset(noisy_root,clean_root)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        collate_fn=Collator(params).concat_collate,
        shuffle=True,
        num_workers=1,
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=False)

def from_testpath(noisy_root,clean_root, params, is_distributed=False):

    dataset = TotensorDataset(noisy_root,clean_root)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers= 1,
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=False)

