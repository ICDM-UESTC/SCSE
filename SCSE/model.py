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
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

from math import sqrt


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d

def Conv1d(*args, **kwargs):
  layer = nn.Conv1d(*args, **kwargs)
  nn.init.kaiming_normal_(layer.weight)
  return layer

def get_value_from_index(index, min_val=1, max_val=1.2, max_index=30):
    if index < 0 or index > max_index:
        raise ValueError(f"Index should be between 0 and {max_index}")
    
    # 计算线性插值
    value = min_val + (max_val - min_val) * (index / max_index)
    return value

@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
  def __init__(self, max_steps):
    super().__init__()
    self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
    self.projection1 = Linear(128, 512)
    self.projection2 = Linear(512, 512)

  def forward(self, diffusion_step):
    if diffusion_step.dtype in [torch.int32, torch.int64]:
      x = self.embedding[diffusion_step]
    else:
      x = self._lerp_embedding(diffusion_step)
    x = self.projection1(x)
    x = silu(x)
    x = self.projection2(x)
    x = silu(x)
    return x

  def _lerp_embedding(self, t):
    low_idx = torch.floor(t).long()
    high_idx = torch.ceil(t).long()
    low = self.embedding[low_idx]
    high = self.embedding[high_idx]
    return low + (high - low) * (t - low_idx)

  def _build_embedding(self, max_steps):
    steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
    dims = torch.arange(64).unsqueeze(0)          # [1,64]
    table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    return table


class ConditionConvBlock(nn.Module):
  def __init__(self, residual_channels):
    '''
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    '''
    super().__init__()
    self.input_projection = Conv1d(1,2 * residual_channels, 1)
    self.output_projection = Conv1d(2 * residual_channels, 4 * residual_channels, 1)
    self.glu = nn.GLU(dim=1)

  def forward(self, x):
    x = F.relu(self.input_projection(x))
    x = self.glu(self.output_projection(x))
    return x



class ResidualBlock(nn.Module):
  def __init__(self, residual_channels, dilation, uncond=True):
    '''
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable conditional
    '''
    super().__init__() 
    self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
    self.diffusion_projection = Linear(512, residual_channels)
    self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)
    self.uncond = uncond

  def forward(self, x, diffusion_step, condition=None, index=None):

    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
    y = x + diffusion_step
    y = self.dilated_conv(y)
    if self.uncond != True:
      coff = get_value_from_index(index)
      # inference-only 
      # y += coff * condition
      y += condition

    gate, filter = torch.chunk(y, 2, dim=1)
    y = torch.sigmoid(gate) * torch.tanh(filter)

    y = self.output_projection(y)
    residual, skip = torch.chunk(y, 2, dim=1)
    return (x + residual) / sqrt(2.0), skip



class SCSE(nn.Module):
  def __init__(self, params):
    super().__init__()
    self.params = params
    self.input_projection = Conv1d(2, params.residual_channels, 1)
    self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
    
    self.condition_embedding = ConditionConvBlock(params.residual_channels)

    self.residual_layers = nn.ModuleList([
        ResidualBlock(params.residual_channels, 2**(i % params.dilation_cycle_length), uncond=params.unconditional)
        for i in range(params.residual_layers)
    ])
    self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
    self.output_projection = Conv1d(params.residual_channels, 1, 1)
    nn.init.zeros_(self.output_projection.weight)
  
  def forward(self, audio, diffusion_step, condition=None):
    audio = audio.unsqueeze(1)
    condition = condition.unsqueeze(1)
    x = torch.cat([audio, condition], dim=1)
    x = self.input_projection(x)
    x = F.relu(x)

    diffusion_step = self.diffusion_embedding(diffusion_step)
    condition = self.condition_embedding(condition)
    
    skip = None
    # for inference only, can scaling the skip connection by the depth of layers.
    # if u want to train, please delete index to avoid something unexpect.
    index = 1
    for layer in self.residual_layers:
      x, skip_connection = layer(x, diffusion_step, condition , index)
      skip = skip_connection if skip is None else skip_connection + skip
     
    x = skip / sqrt(len(self.residual_layers))
    x = self.skip_projection(x)
    x = F.relu(x)
    x = self.output_projection(x)
    return x