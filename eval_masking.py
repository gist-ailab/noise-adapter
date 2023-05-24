import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import argparse

from utils import *


import torch
import torch.nn as nn
import numpy as np

import models

def masking_image(x, ratio):
    x_ = x.clone()
    B, _, H, W = x.shape

    # 224 / 16
    mshape = B, 1, round(H / 16), round(W / 16)
    input_mask = torch.rand(mshape, device=x_.device)
    input_mask = (input_mask > ratio).float()
    input_mask = F.interpolate(input_mask, scale_factor=16, mode='nearest')
    masked_x = x_ * input_mask
    return masked_x

def validation_accuracy_masking(model, loader, device):
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs_masking = []
            for i in range(20):
                inputs_ = masking_image(inputs, 0.5)
                outputs = model(inputs_)
                # outputs = torch.softmax(outputs, dim=1)
                outputs_masking.append(outputs.unsqueeze(0))
                # print(outputs.shape)
            outputs_masking = torch.cat(outputs_masking, dim=0)
            # print(outputs_masking.shape)
            outputs_masking = outputs_masking.mean(0)
            # print(outputs_masking.shape)

            #print(outputs.shape)
            total += targets.size(0)
            _, predicted = outputs_masking.max(1)  
            correct += predicted.eq(targets).sum().item()
    valid_accuracy = correct/total
    return valid_accuracy

def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'vit_small_patch16_224', type=str)
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--nr', default = 0.2,  type=float)


    args = parser.parse_args()

    config = read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    dataset_path = config['id_dataset']
    batch_size = config['batch_size']
    save_path = config['save_path'] + args.save_path
    
    num_classes = int(config['num_classes'])

    if 'cifar' in args.data:
        train_loader, valid_loader = get_cifar_noisy(args.data, dataset_path, batch_size, 0.0)
        noise_loader, _ = get_cifar_noisy(args.data, dataset_path, batch_size, args.nr)

    else:
        valid_loader = get_svhn(dataset_path, batch_size)

    model = timm.create_model(args.net, pretrained=True, num_classes=num_classes)
    if args.net == 'resnet18':
        model = models.ResNet18(num_classes=num_classes)
        
    if 'wrn40' == args.net:
        import wrn
        model = wrn.WideResNet(40, num_classes, 2, 0.3)
        
    state_dict = (torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict'])    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    train_accuracy = validation_accuracy(model, train_loader, device)
    train_accuracy_masking = validation_accuracy_masking(model, train_loader, device)
    print('In-distribution accuracy: ', train_accuracy, train_accuracy_masking)

    valid_accuracy = validation_accuracy(model, valid_loader, device)
    valid_accuracy_masking = validation_accuracy_masking(model, train_loader, device)
    print('In-distribution accuracy: ', valid_accuracy, valid_accuracy_masking)



if __name__ =='__main__':
    eval()