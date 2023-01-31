''' PyTorch training on vision dataset'''

#%%

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler

import os
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import pandas as pd
import numpy as np

from models import VGG, LeNet
from multimodaldataset import MultimodalDataset, LDEDAudioDataset, LDEDVisionDataset
from utils import progress_bar

torch.manual_seed(0)
Multimodal_dataset_PATH = os.path.join("C:\\Users\\Asus\\OneDrive_Chen1470\\OneDrive - Nanyang Technological University\\Dataset\\Multimodal_AM_monitoring\\LDED_Acoustic_Visual_Dataset")
CCD_Image_30Hz_path = os.path.join(Multimodal_dataset_PATH, 'Coaxial_CCD_images_30Hz')
Audio_segmented_30Hz_PATH = os.path.join(Multimodal_dataset_PATH, 'Audio_signal_all_30Hz')
Audio_raw_seg_PATH = os.path.join(Audio_segmented_30Hz_PATH, 'raw')
Audio_equalized_seg_PATH = os.path.join(Audio_segmented_30Hz_PATH, 'equalized')
Audio_bandpassed_seg_PATH = os.path.join(Audio_segmented_30Hz_PATH, 'bandpassed')
Audio_denoised_seg_PATH = os.path.join(Audio_segmented_30Hz_PATH, 'denoised')
AUDIO_DIR = Audio_denoised_seg_PATH
VISON_DIR = CCD_Image_30Hz_path
SAMPLE_RATE = 44100

ANNOTATIONS_FILE = os.path.join(Multimodal_dataset_PATH, "vision_acoustic_label.csv")

classes = ('Defect-free', 'Cracks', 'Keyhole pores', 'Laser-off', 'Laser-start')

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 15

annotations_df = pd.read_csv(ANNOTATIONS_FILE)
annotations_df.head()
# Get the labels and count the number of samples for each class
labels = annotations_df['class_ID'].values
label_counts = np.unique(labels, return_counts=True)[1]
print (label_counts)

#%%


def train(model, epoch, trainloader, loss_fn, optimizer, device):
    '''
    Function for the training single epoch of the training loop
    '''
    print('\nEpoch: %d' % epoch)
    model.train() # training mode
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # for inputs, targets in trainloader:
        print(type(targets))
        inputs, targets = inputs.to(device), targets.to(device)
        # calculate loss
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        # backpropagate error and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print(f"loss: {loss.item()}")
    print("---------------------------")


def test(model, epoch, testloader, loss_fn, device):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch LAAM Vision Dataset Training')
    # parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # instantiating  dataset object and create data loader
    print('==> Preparing dataloader (train and test set)..')
    
    #------ transformation------
    train_transforms=transforms.Compose([
        torchvision.transforms.Resize((32,32)), # original image size: (640,480)
        # data augmentation
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize(mean=[136.20371045013258], std=[61.9731240029325]),
    ])
    
    val_transforms=transforms.Compose([
        torchvision.transforms.Resize((32,32)), # original image size: (640,480)
        transforms.ToTensor(),
        transforms.Normalize(mean=[136.20371045013258], std=[61.9731240029325]),
    ])
 
    #------ data loader (train and test set) ------
    # create the dataset for all samples
    visiondataset = LDEDVisionDataset(ANNOTATIONS_FILE,
                                      VISON_DIR,
                                      train_transforms,
                                      device)
    print ("length of the total dataset:" + str(len(visiondataset)))
  
    train_ratio = 0.8
    # val_ratio = 0.2
    test_ratio = 0.2

    # dealing with dataset imbalance: stratified sampling
    train_idx, val_idx, test_idx = [], [], []
    for class_idx, class_count in enumerate(label_counts):
        indices = np.where(labels == class_idx)[0]
        # Shuffle the indices for this class
        np.random.shuffle(indices)

        train_split = int(train_ratio * class_count)
        test_split = int(test_ratio * class_count) + train_split
        
        train_idx.extend(indices[:train_split])
        # val_idx.extend(indices[train_split:val_split])
        test_idx.extend(indices[test_split:])

    train_sampler = SubsetRandomSampler(train_idx)
    # val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # train_dataset, test_dataset = torch.utils.data.random_split(visiondataset, [train_len, test_len])

    # Create the Dataset objects with the corresponding transforms
    train_dataset = LDEDVisionDataset(ANNOTATIONS_FILE, VISON_DIR, train_transforms, device)
    # val_dataset = LDEDVisionDataset(ANNOTATIONS_FILE, VISON_DIR, val_transforms, device)
    test_dataset = LDEDVisionDataset(ANNOTATIONS_FILE, VISON_DIR, val_transforms, device)

    ## dealing with inbalanced dataset

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print("length of the train dataset:" +  str(len(train_dataset)))
    print("length of the val dataset:" +  str(len(test_dataset)))

  
    # -----Model---------------
    print('==> Building model..')
    # net = VGG('VGG19')
    net = LeNet() 
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE,
    #                     momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)



    for epoch in range(start_epoch, start_epoch+EPOCHS):
        train(net, epoch, trainloader, loss_fn, optimizer, device)
        test(net, epoch, testloader, loss_fn, device)
        scheduler.step()
