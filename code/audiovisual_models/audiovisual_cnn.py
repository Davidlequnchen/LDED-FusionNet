# -*- coding: utf-8 -*-
"""
Created on Wednesday 11 Jan 2023
---------------------------------------------------------------------
-- Author: Chen Lequn
---------------------------------------------------------------------
FusionNet model
(1) AudioVisualFusionCNN: Image and Audio are on separate branch CNN, then fused together
   --> also the so-called "feature-level" fusion
(2) EarlyFusion: Image and Audio are fused directly in the initial layer. 
   --> also known as "data-level" fusion
"""
import os
import sys
from typing import Callable, Dict
from torchviz import make_dot
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import PIL.Image
import torch
import torch.utils.data
import pandas as pd
import torchaudio
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torchsummary import summary
from torchvision.models import resnet18, resnet34, resnet50
import torchvision.models as models
# from ..audio_models.mfcccnn import MFCCCNN

import torch.nn as nn
from torchvision.models import vgg11_bn, vgg16_bn, vgg19_bn, resnet18, resnet34, resnet50



import torch.nn as nn
import torch.nn.functional as F

class MFCCCNN(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(MFCCCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=2*10*32, out_features=640)
        self.fc_bn1 = nn.BatchNorm1d(640)
        self.fc2 = nn.Linear(in_features=640, out_features=256)
        self.fc_bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(in_features=256, out_features=num_classes)
        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_bn1(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc_bn2(F.relu(self.fc2(x)))
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits




def get_visual_backbone(visual_backbone_type, visual_handcrafted_features=None):
    if visual_backbone_type == 'vgg11':
        model = torchvision.models.vgg11_bn(pretrained=True)
        num_features = 512
    elif visual_backbone_type == 'vgg16':
        model = torchvision.models.vgg16_bn(pretrained=True)
        num_features = 512
    elif visual_backbone_type == 'vgg19':
        model = torchvision.models.vgg19_bn(pretrained=True)
        num_features = 512
    elif visual_backbone_type == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        num_features = model.fc.in_features
    elif visual_backbone_type == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
        num_features = model.fc.in_features
    elif visual_backbone_type == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        num_features = model.fc.in_features
    elif visual_backbone_type == 'handcrafted':
        model = None
        num_features = visual_handcrafted_features
    else:
        raise ValueError(f"Unsupported visual backbone type: {visual_backbone_type}")

    # Adjusting for grayscale input
    if visual_backbone_type in ['vgg11', 'vgg16', 'vgg19']:
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    elif visual_backbone_type in ['resnet18', 'resnet34', 'resnet50']:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    return model, num_features


def get_audio_backbone(backbone_type, handcrafted_features=None):
    if backbone_type == 'vgg11':
        model = models.vgg11_bn(pretrained=False)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        num_features = model.classifier[6].in_features
    elif backbone_type == 'vgg16':
        model = models.vgg16_bn(pretrained=False)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        num_features = model.classifier[6].in_features
    elif backbone_type == 'vgg19':
        model = models.vgg19_bn(pretrained=False)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        num_features = model.classifier[6].in_features
    elif backbone_type == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_features = model.fc.in_features
    elif backbone_type == 'resnet34':
        model = models.resnet34(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_features = model.fc.in_features
    elif backbone_type == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_features = model.fc.in_features
    elif backbone_type == 'handcrafted':
        model = None
        num_features = handcrafted_features
    elif backbone_type == 'mfcccnn':
        model = MFCCCNN()
        num_features = 128  # As defined in your MFCCCNN architecture
    else:
        raise ValueError('Invalid audio backbone type')

    return model, num_features



class AudioVisualFusionCNN(nn.Module):
    def __init__(self, visual_backbone_type, audio_backbone_type, num_classes, visual_handcrafted_features=None, audio_handcrafted_features=None):
        super(AudioVisualFusionCNN, self).__init__()

        # Initialize the visual and audio backbones
        self.visual_backbone, visual_num_features = get_visual_backbone(visual_backbone_type, visual_handcrafted_features)
        self.audio_backbone, audio_num_features = get_audio_backbone(audio_backbone_type, audio_handcrafted_features)

        # Fully connected layers
        self.fc1 = nn.Linear(visual_num_features + audio_num_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, visual_input, audio_input):
        # Forward pass for visual backbone
        if self.visual_backbone:
            visual_features = self.visual_backbone(visual_input)
            # If visual backbone is a ResNet, then it will be a direct output. If VGG, we will use the classifier output.
            if isinstance(self.visual_backbone, nn.Sequential):  # VGG variants
                visual_features = visual_features.view(visual_features.size(0), -1)
            else:  # ResNet variants
                visual_features = visual_features.view(visual_features.size(0), -1)
        else:
            visual_features = visual_input

        # Forward pass for audio backbone
        if self.audio_backbone:
            audio_features = self.audio_backbone(audio_input)
            # If audio backbone is MFCCCNN or a ResNet, then it will be a direct output. If VGG, we will use the classifier output.
            if isinstance(self.audio_backbone, MFCCCNN) or not isinstance(self.audio_backbone, nn.Sequential):
                audio_features = audio_features.view(audio_features.size(0), -1)
            else:  # VGG variants
                audio_features = audio_features.view(audio_features.size(0), -1)
        else:
            audio_features = audio_input

        # Concatenate the features from both streams
        fused_features = torch.cat((visual_features, audio_features), dim=1)

        # Fully connected layers
        x = nn.ReLU()(self.fc1(fused_features))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)

        return x



if __name__ == "__main__":
    model = AudioVisualFusionCNN('vgg16', 'resnet34', num_classes=4)
    visual_input = torch.randn(1, 1, 32, 32)
    audio_input = torch.randn(1, 1, 32, 7)
    output = model(visual_input, audio_input)
    print(output.shape)  # Should print torch.Size([1, 4])
