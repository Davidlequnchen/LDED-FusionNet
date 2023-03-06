'''
Created by: Chen Lequn

Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
    - loss_acc_plot: plot the loss and accuracy curve (test and val) during the trining process
    - class_report: predict X_test, return y_pred & y_true and print the classification report
    - conf_matrix: compute and plot confusion matrix
    - stratified_k_fold_torch: stratified K fold with Pytorch DataLoaders
    - roc_auc_evaluation: Calculate Micro-Macro ROC-AUC value, plot the ROC curve.
    - cross_validation_model: calculated mean and std of test accuracy on K fold.
    - get_accuracy
'''

import os
import sys
import time
import math
import numpy as np

from prettytable import PrettyTable
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchaudio
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler

## plot
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.pyplot import gca
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import seaborn as sns
from itertools import cycle
import itertools

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle, resample, class_weight
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize


term_width = 30

TOTAL_BAR_LENGTH = 80.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()



def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        print('Learning rate =')
        print(param_group['lr'])
        return param_group['lr']
    

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


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

### Define Loss and Accuracy plot function
def loss_acc_plot(train_losses, valid_losses, train_accuracy, valid_accuracy, epochs_num, title, interval=20, yloss_limit1=0, yloss_limit2=1.5, yacc_limit1=0.4, yacc_limit2=1):
    fig, (ax1,ax2) = plt.subplots(nrows = 2, sharex = True, figsize=(7,8));
    # plt.title(title, fontsize = 20, y=1.05)
    # Loss plot
    ax1.plot(train_losses, 'darkorange', label = 'Train Loss', linewidth=2)
    ax1.plot(valid_losses, 'navy', label = 'Test Loss', linewidth=2)
    ax1.legend(loc =1, fontsize = 16)
    ax1.set_xlabel('Epochs', fontsize = 20)
    ax1.set_xticks(np.arange(0,epochs_num+1,interval))
    ax1.set_ylabel('Crossentropy Loss', fontsize = 20)
    ax1.set_ylim(yloss_limit1,yloss_limit2)
    ax1.set_title('Loss Curve', fontsize = 20, pad=12)
    ax1.xaxis.set_tick_params(labelsize=18)
    ax1.yaxis.set_tick_params(labelsize=18)
    
    # Accuracy plot
    ax2.plot(train_accuracy, 'darkorange', label = 'Train Accuracy', linewidth=2)
    ax2.plot(valid_accuracy, 'navy', label = 'Test Accuracy', linewidth=2)
    ax2.legend(loc =4, fontsize = 16)
    ax2.set_xlabel('Epochs', fontsize = 20)
    ax1.set_xticks(np.arange(0,epochs_num+1,interval))
    ax2.set_ylabel('Accuracy', fontsize =20)
    ax2.set_ylim(yacc_limit1,yacc_limit2)
    ax2.set_title('Accuracy Curve', fontsize =20, pad=12)
    ax2.xaxis.set_tick_params(labelsize=18)
    ax2.yaxis.set_tick_params(labelsize=18)
    ax1.grid(zorder=3, linestyle='--',linewidth=0.8, alpha=0.4, color = "k") #linestyle='--', color='r'
    ax2.grid(zorder=3, linestyle='--',linewidth=0.8, alpha=0.4, color = "k") #linestyle='--', color='r'
    # fig.suptitle(title, fontsize = 22, y=1.001)
    plt.tight_layout()

def get_accuracy(model, data_loader, device, mode = 'single_model'):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    correct_pred = 0 
    total = 0
    with torch.no_grad():
        model.eval()
        for inputs, targets in data_loader:
            if mode == "single_model":
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
            elif mode == "multi_model":
                inputs = [x.to(device) for x in inputs]
                targets = targets.to(device)
                outputs = model(*inputs)
        
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_pred += predicted.eq(targets).sum().item()
            accuracy = correct_pred/total
    return accuracy


### Define function to predict X_test, return y_pred & y_true and print the classification report
def class_report(model, testdataloader, device, classes, mode='single_model'):
    # initialize variables to store true and predicted labels
    y_true, y_pred = [], []
    with torch.no_grad():
        model.eval()
        for inputs, targets in testdataloader:
            if mode == "single_model":
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
            elif mode == "multi_model":
                inputs = [x.to(device) for x in inputs]
                targets = targets.to(device)
                outputs = model(*inputs)

            _, predicted = outputs.max(1)
            # _, predicted = torch.max(outputs, dim=1)
            predicted = predicted.cpu().numpy()     
            # convert true labels to a list
            y_true += [classes[index] for index in targets.cpu().numpy()]
            y_pred += [classes[index] for index in predicted]
    
    print(classification_report(y_true, y_pred,digits=4))
    return y_true, y_pred


### Function to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(7,7))
    im_ratio = cm.shape[1]/cm.shape[0]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20, pad=12)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.3f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 fontsize = 16, 
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Ground Truth', fontsize=20, labelpad =12)
    plt.xlabel('Predicted', fontsize=20, labelpad =12)
    plt.xticks(fontsize=16,  rotation=45, ha='right')
    plt.yticks(fontsize=16)
    cbar = plt.colorbar(orientation="vertical", pad=0.1, ticks=[0.1, 0.4, 0.8], fraction=0.045*im_ratio)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_title('Accuracy',fontsize=16, pad = 12)
    plt.tight_layout()

    # plt.show()
    
def plot_confusion_matrix_sns(y_true, y_pred, classes):
    plt.figure(figsize=(10, 7))
    tick_marks = np.arange(len(classes))
    cm = confusion_matrix(y_true, y_pred)
    # convert to percentage and plot the confusion matrix
    cm_pct = cm.astype(float) / cm.sum(axis =1)[:,np.newaxis]
    sns.heatmap(cm_pct, annot=True, fmt='.3%', cmap='Blues', linewidths=2, linecolor='black') #cmap='Blues'
    plt.xticks(tick_marks, classes, horizontalalignment='center', rotation=70, fontsize=12)
    plt.yticks(tick_marks, classes, horizontalalignment="center", rotation=0, fontsize=12)

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    
    
## Define function to get the confusion matrix and print out the plot as well
def conf_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    
    # convert to percentage and plot the confusion matrix
    cm_pct = cm.astype(float) / cm.sum(axis =1)[:,np.newaxis]

    print(cm)
    plot_confusion_matrix(cm_pct, classes)


def stratified_k_fold_torch(test_dataloader, k=5, seed=0):
    # Create a list to store the k folds of dataloaders
    fold_dataloaders = []
    # Get the dataset from the dataloader
    dataset = test_dataloader.dataset
    # Get the targets from the dataset
    targets = [target for _, target in dataset]
    # Create the StratifiedKFold object
    stratified_k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    # Divide the dataset into k folds
    for train_index, val_index in stratified_k_fold.split(dataset, targets):
        # Create a subdataset for the fold
        fold_dataset = torch.utils.data.Subset(dataset, val_index)
        # Create a dataloader for the fold
        fold_dataloader = DataLoader(fold_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
        # Append the fold dataloader to the list of fold dataloaders
        fold_dataloaders.append(fold_dataloader)
    return fold_dataloaders


def cross_validation_model(model, dataloader, device, mode = 'single_model'):
    tests = []
    for fold_dataloaders in stratified_k_fold_torch(dataloader):
        scores = get_accuracy(model, fold_dataloaders, device, mode = mode)
        tests.append(scores)
    test_accuracy_mean = np.mean(tests)
    test_accuracy_std = np.std(tests)
    return test_accuracy_mean, test_accuracy_std


def model_evaluation(model, dataloader, classes, device, classifier_name = "MFCC-CNN", signal_type = "denoised", seed=0, mode = 'single_model'):
    n_classes = len(classes)
    torch.manual_seed(seed)
    #------------- Perform cross-validation on the model ------------#
    test_accuracy_mean, test_accuracy_std = cross_validation_model(model, dataloader, device, mode = mode)
    #------------ AUC-ROC score measurement ---------#
    auc_mean, auc_std = roc_auc_evaluation(model, dataloader, device, classes, classifier_name, mode = mode)
    #------------ Summary ---------#
    print('Test Accuracy (cross-validation) for' , classifier_name, '= {:.5f} ± {:.5f}'.format(test_accuracy_mean, test_accuracy_std))
    print('micro-averaging AUC for' , classifier_name, '= {:.5f} ± {:.5f}'.format(auc_mean, auc_std))
    # save_fig("ROC_" + classifier_name + "_" + signal_type)
    return test_accuracy_mean, test_accuracy_std, auc_mean, auc_std


def roc_auc_evaluation(model, dataloader, device, classes, classifier_name, mode = 'single_model'):
    '''
    Compute ROC curve and ROC area for each class
    '''
    n_classes = len(classes)
    all_targets = []
    all_targets_ohe = []
    all_scores = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            if mode == "single_model":
                inputs, targets = inputs.to(device), targets.to(device)
                scores = model(inputs)
                targets_ohe = torch.nn.functional.one_hot(targets, num_classes = n_classes)
            elif mode == "multi_model":
                inputs = [x.to(device) for x in inputs]
                targets = targets.to(device)
                scores = model(*inputs)
                targets_ohe = torch.nn.functional.one_hot(targets, num_classes = n_classes)
            all_targets_ohe.append(targets_ohe.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_scores.append(scores.cpu().numpy())
    all_targets = np.concatenate(all_targets, axis=0)
    all_targets_ohe = np.concatenate(all_targets_ohe, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
                                    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_targets_ohe[:, i], all_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # calculate micro-average ROC-AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(all_targets_ohe.ravel(), all_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

     # ----------------------------------Plot all ROC curves-------------------------------
    plt.figure(figsize = (4,3), dpi = 300)
    widths = 2
    ax = gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(widths)

        tick_width = 1.5
    plt.tick_params(direction = 'in', width = tick_width)
    
    #---------------------------(1) micro and macro ROC curve---------------------------
    plt.plot(fpr["micro"], tpr["micro"],
             label=classifier_name + ' - micro-average ROC curve (AUC = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=2, alpha = 0.8) #deeppink, midnightblue

    plt.plot(fpr["macro"], tpr["macro"],
             label=classifier_name + ' - macro-average ROC curve (AUC = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=2, alpha = 0.8) #navy, gold
    
    #---------------------------(2) ROC curve for each class---------------------------
    colors = cycle(["red", "aqua", "darkblue", "darkorange"])
    # colors = cycle(['0.45', 'steelblue',  'olive', 'silver'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i],tpr[i],color=color,
                 lw=1, alpha = 0.8,
                 label=classifier_name + " ROC curve of class \"{0}\" (area = {1:0.2f})".format(classes[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.title("ROC curve")
    plt.legend(loc="lower right",  fontsize = '5', frameon = False)
    # plt.show()

    #----------------------------------Extract the auc score to list-----------------------------------
    auc_score_list = []
    auc_score_list.append(auc(fpr["micro"], tpr["micro"]))
    auc_score_array = np.array(auc_score_list)
    auc_mean = auc_score_array.mean()
    auc_std = auc_score_array.std()
    
    return auc_mean, auc_std



