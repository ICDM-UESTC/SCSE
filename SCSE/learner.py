# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import from_path,from_testpath
from model import SCSE
from params import AttrDict
import wandb
from params import params
from metric import pesq


def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


class SCSELearner:
  def __init__(self, model_dir, model, dataset, valid_dataset, optimizer, params, online_log, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.valid_dataset = valid_dataset
    self.optimizer = optimizer
    self.params = params
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.is_master = True
    self.online_log = online_log
    
    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta)

    
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    self.loss_fn = nn.L1Loss()
    self.summary_writer = None
    self.eta = 1.5

  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'])
    else:
      self.model.load_state_dict(state_dict['model'])
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']

  def save_to_checkpoint(self, filename='weights'):
    save_basename = f'{filename}-{self.step}.pt'
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)
    if os.name == 'nt':
      torch.save(self.state_dict(), link_name)
    else:
      if os.path.islink(link_name):
        os.unlink(link_name)
      os.symlink(save_basename, link_name)

  def restore_from_checkpoint(self, filename='weights'):
    try:
      checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
      self.load_state_dict(checkpoint)
      return True
    except FileNotFoundError:
      return False
    
  def _write_test_summary(self, step, loss, pesq):
    writer = self.summary_writer
    writer.add_scalar('test/loss', loss, step)
    writer.add_scalar('test/pesq', pesq, step)
    writer.flush()
    return

  def train(self, max_steps=None):
    
    device = next(self.model.parameters()).device
    
    while True:
      self.model.train()
      for features in tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:
        if max_steps is not None and self.step >= max_steps:
          return
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
      
        loss = self.train_step(features)
        
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          if self.online_log!=True:
            if self.step % 50 == 0:
              self._write_summary(self.step, loss)
          
          else:
            if self.step % 50 == 0:
              wandb.log(
                {'train_loss': loss.item(),
                 'train_grad_norm':self.grad_norm.item()
                 }
                  )      
          if self.step % len(self.dataset) == 0:
            self.save_to_checkpoint()
        self.step += 1

      if self.step % 5*(len(self.dataset)) == 0:
        mean_valid_loss,pesq_mos = self.inference()
        if self.online_log!=True:
          self._write_test_summary(self.step, mean_valid_loss, pesq_mos)
        else:
          wandb.log(
            {'mean_valid_loss': mean_valid_loss,
              'pesq_mos':pesq_mos
              })


  @torch.no_grad()
  def inference(self):
    device = next(self.model.parameters()).device
    self.model.eval()
    pesq_mos_list=[]
    loss_list = [] 
    for features in tqdm(self.valid_dataset, desc=f'Test') if self.is_master else self.dataset:
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)  
        loss,pesq = self.inference_step(features)
        # loss,pesq = self.common_inference_step(features)
        pesq_mos_list.append(pesq)
        loss_list.append(loss)
    loss = np.mean(loss_list)
    pesq_mos = np.mean(pesq_mos_list)
    return loss,pesq_mos
  
  def inference_step(self, features):
    for param in self.model.parameters():
      param.grad = None

    clean = features['clean_speech']
    noisy = features['noisy_speech']

    N,_= noisy.shape
    device = noisy.device
    self.noise_level = self.noise_level.to(device)

    training_noise_schedule = np.array(params.noise_schedule)
    inference_noise_schedule = np.array(params.noise_schedule)

    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)

    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)

    time_step = 25
    _step = torch.full([1],time_step)
    noise_scale = self.noise_level[_step].unsqueeze(1).to(device)
    noise_scale_sqrt = noise_scale**0.5
    
    noise = torch.randn_like(noisy).to(device)
    audio = noise_scale_sqrt * noisy + (1.0 - noise_scale)**0.5 * noise
    predicted = self.model(audio, torch.tensor([T[time_step]], device=audio.device), noisy)
    loss = self.loss_fn(clean, predicted.squeeze(1))
    pesq_mos = pesq(clean.squeeze(0).cpu().numpy(),predicted.squeeze(0).squeeze(0).cpu().numpy(),16000)

    return loss.item(),pesq_mos

  def common_inference_step(self, features):

    for param in self.model.parameters():
      param.grad = None

    clean = features['clean_speech']
    noisy = features['noisy_speech']

    N,_= noisy.shape
    device = noisy.device
    self.noise_level = self.noise_level.to(device)

    training_noise_schedule = np.array(params.noise_schedule)
    inference_noise_schedule = np.array(params.noise_schedule)

    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)

    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)

    # audio = torch.randn_like(clean)

    time_step = 25
    _step = torch.full([1],time_step)
    noise_scale = self.noise_level[_step].unsqueeze(1).to(device)
    noise_scale_sqrt = noise_scale**0.5
    
    noise = torch.randn_like(noisy).to(device)
    audio = noise_scale_sqrt * noisy + (1.0 - noise_scale)**0.5 * noise

    condition = noisy
    for time_step in range(time_step, -1, -1):
      t = torch.full([1],time_step).to(device)
      c1 = 1 / alpha[time_step]**0.5
      c2 = beta[time_step] / (1 - alpha_cum[time_step])**0.5
      audio = c1 * (audio - c2 * self.model(audio, torch.tensor([T[time_step]], device=audio.device), condition).squeeze(1))
      if time_step > 0:
        noise = torch.randn_like(audio)
        sigma = ((1.0 - alpha_cum[time_step-1]) / (1.0 - alpha_cum[time_step]) * beta[time_step])**0.5
        audio += sigma * noise

    predicted = audio
  
    loss = self.loss_fn(clean, predicted.squeeze(1))
    pesq_mos = pesq(clean.squeeze(0).cpu().numpy(),predicted.squeeze(0).squeeze(0).cpu().numpy(),16000)

    return loss.item(),pesq_mos
  

  def train_step(self, features):
    for param in self.model.parameters():
      param.grad = None

    audio = features['clean_speech']
    noisy = features['noisy_speech']


    N,T= audio.shape
    device = audio.device
    self.noise_level = self.noise_level.to(device)

    with self.autocast:
      t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)      
      noise_scale = self.noise_level[t].unsqueeze(1)
      noise_scale_sqrt = noise_scale ** 0.5
      noise = torch.randn_like(audio)

      noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise
      predicted = self.model(noisy_audio, t, noisy)
      loss = self.loss_fn(audio, predicted.squeeze(1))


    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss

  def _write_summary(self, step, loss):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    
    writer.add_scalar('train/loss', loss, step)
    writer.add_scalar('train/grad_norm', self.grad_norm, step)
    writer.flush()
    self.summary_writer = writer


def _train_impl(replica_id, model, dataset, valid_dataset, args, params):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

  learner = SCSELearner(args.model_dir, model, dataset, valid_dataset, opt, params, args.wandb, fp16=args.fp16)
  learner.is_master = (replica_id == 0)
  learner.restore_from_checkpoint()
  learner.train(max_steps=args.max_steps)


def train(args, params):
  
  dataset = from_path('noisy_train_path',
        'clean_train_path', params)
  
  valid_dataset = from_testpath('noisy_testset_path',
      'clean_testset_path', params)

  device = torch.device('cuda', args.device_num)
  model = SCSE(params).to(device)
  _train_impl(0, model, dataset, valid_dataset, args, params)

