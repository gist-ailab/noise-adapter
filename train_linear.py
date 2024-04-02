import os
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
import utils

import random

def train():
    device = 'cuda:0'
    data_path = 'data'
    network = 'resnet18'
    max_epoch = 40
    save_path = './checkpoints/dino_linear_0.2'
    filter_path = 'filter.txt'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    lr_decay = [int(0.8*max_epoch), int(0.9*max_epoch)]

    train_loader, valid_loader = utils.get_loader(data_path, noise_rate = 0.2, filter_path=filter_path)

    # model = timm.create_model(network, pretrained=True, num_classes=2) 
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.linear = nn.Linear()
    
    model.to(device)
    
    
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    
    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay = 1e-05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, momentum=0.9, weight_decay = 1e-5)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    saver = timm.utils.CheckpointSaver(model.linear, optimizer, checkpoint_dir= save_path, max_history = 2) 
    print(train_loader.dataset[0][0].shape)

    for epoch in range(max_epoch):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)     
            
            loss = criterion(outputs, targets)
            loss.backward()            
            optimizer.step()

            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()            
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')                       
        train_accuracy = correct/total
        train_avg_loss = total_loss/len(train_loader)
        print()

        ## validation
        model.eval()
        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy = utils.validation_accuracy(model, valid_loader, device)
        scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())

if __name__ =='__main__':
    train()