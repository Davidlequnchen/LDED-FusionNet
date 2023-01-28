# -*- coding: utf-8 -*-
"""
Created on Wednesday 11 Jan 2023
---------------------------------------------------------------------
-- Author: Chen Lequn
---------------------------------------------------------------------
FusionNet model
(1) LateFusionCNN: The feature extraction for image and audio signal are on separate branch stream, then fused together
   --> also the so-called "late fusion"
(2) EarlyFusionCNN: Image and Audio are fused in the initial layer. 
"""

#%%
import torch
import torch.nn as nn

from torch import nn, optim
from torch.nn import functional as F
from prettytable import PrettyTable



class LateFusionCNN(nn.Module):
    def __init__(self):
        super(LateFusionCNN, self).__init__()
        self.image_stream = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.audio_stream = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(768, num_classes)

    def forward(self, image, audio):
        # reshape the image data
        image = image.view(-1, 1, 640, 480)
        # reshape the audio data
        audio = audio.view(-1, 1, 1470)

        image_features = self.image_stream(image)
        audio_features = self.audio_stream(audio)
        image_features = image_features.view(image_features.size(0), -1)
        audio_features = audio_features.view(audio_features.size(0), -1)
        features = torch.cat((image_features, audio_features), dim=1)
        out = self.fc(features)
        return out


#%%       
class EarlyFusionCNN(nn.Module):
    def __init__(self):
        super(EarlyFusionCNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
        
        self.fc = nn.Linear(128*6*6, 10)

    def forward(self, image, audio):
        # reshape the image data
        image = image.view(-1, 1, 640, 480) # 640, 480 corresponds to the input dimension of the image
        # reshape the audio data
        audio = audio.view(-1, 1, 1470) # 1470 corresponds to the input dimension of the audio
        # Combine the audio and image data along the channel dimension
        combined = torch.cat((image, audio), dim=1)
        # Pass the combined data through the CNN
        features = self.cnn(combined)
        features = features.view(features.size(0), -1)
        out = self.fc(features)
        return out

#%%


def initialize_weights(m):
  if isinstance(m, nn.Conv1d):
      nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm1d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)


#%%
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        print('Learning rate =')
        print(param_group['lr'])
        return param_group['lr']
    

#%%
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

