'''
Created by: Chen Lequn

Some helper functions for PyTorch model training and testing. 
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchaudio
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle, resample, class_weight
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize

from utils import progress_bar
from datetime import datetime


def train_single_epoch(model, epoch, trainloader, loss_fn, optimizer, device, mode = "single_model"):
    '''
    Function for the training single epoch in the training loop
    '''
    print('\nEpoch: %d' % epoch)
    model.train() # training mode
    running_loss = 0
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if mode == "single_model":
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
        elif mode == "multi_model":
            inputs = [x.to(device) for x in inputs]
            targets = targets.to(device)
            ## forward pass and calculate loss
            outputs = model(*inputs)
            loss = loss_fn(outputs, targets)
            running_loss += loss.item() * inputs[0].size(0)

        # backpropagate error and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # record current progress
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
        #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    # print(f"loss: {loss.item()}")
    epoch_loss = running_loss / len(trainloader.dataset)
    # print("--------------epoch finished---------------")
    return model, optimizer, epoch_loss, acc


def test_single_epoch(model, epoch, testloader, loss_fn, device, mode = "single_model"):
    model.eval() # evaluation mode
    global best_acc # for updating the best accuracy so far
    test_loss = 0
    correct = 0
    total = 0
    running_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if mode == "single_model":
                inputs, targets = inputs.to(device), targets.to(device)
                ## forward pass and calculate loss
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                running_loss += loss.item() *inputs.size(0)
            elif mode == "multi_model":
                inputs = [x.to(device) for x in inputs]
                targets = targets.to(device)
                ## forward pass and calculate loss
                outputs = model(*inputs)
                loss = loss_fn(outputs, targets)
                running_loss += loss.item() *inputs[0].size(0)

            # record current progress
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
            #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'model': model.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.pth')
    #     best_acc = acc
    
    epoch_loss = test_loss / len(testloader.dataset)
    return model, epoch_loss, acc



def training_loop(model, loss_fn, optimizer, train_loader, valid_loader, epochs, scheduler, device, print_every=1, mode = "single_model"):
    '''
    Function defining the entire training loop
    '''
    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []
    train_accuracy = []
    valid_accuracy = []
 
    # Train model
    for epoch in range(0, epochs):
        # training
        model, optimizer, train_loss, train_acc = train_single_epoch(model, epoch, train_loader, loss_fn, optimizer, device, mode = mode)
        train_losses.append(train_loss)
        # validation
        model, valid_loss, valid_acc = test_single_epoch(model, epoch, valid_loader, loss_fn, device, mode = mode)
        valid_losses.append(valid_loss)

        # if epoch % print_every == (print_every - 1):
            
        #     train_acc = get_accuracy(model, train_loader, device=device)
        #     valid_acc = get_accuracy(model, valid_loader, device=device)
                
        print(f'{datetime.now().time().replace(microsecond=0)} --- '
              f'Epoch: {epoch}\t'
              f'Train loss: {train_loss:.4f}\t'
              f'Valid loss: {valid_loss:.4f}\t'
              f'Train accuracy: {train_acc:.2f}\t'
              f'Valid accuracy: {valid_acc:.2f}')
        
        train_accuracy.append(train_acc)
        valid_accuracy.append(valid_acc)
        scheduler.step()
    return model, optimizer, (train_losses, valid_losses, train_accuracy, valid_accuracy)


