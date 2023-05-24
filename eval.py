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

def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'resnet18', type=str)
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--method' ,'-m', default = 'msp', type=str)

    args = parser.parse_args()

    config = read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    dataset_path = config['id_dataset']
    batch_size = config['batch_size']
    save_path = config['save_path'] + args.save_path
    
    num_classes = int(config['num_classes'])

    if 'cifar' in args.data:
        train_loader, valid_loader = get_cifar_noisy(args.data, dataset_path, batch_size, 0.0)
    else:
        valid_loader = get_svhn(dataset_path, batch_size)
        
    if args.net == 'resnet18':
        model = models.ResNet18(num_classes=num_classes)

    if args.net == 'resnet50':
        model = timm.create_model(args.net, pretrained=True, num_classes=num_classes)  

        
    if 'wrn40' == args.net:
        import wrn
        model = wrn.WideResNet(40, num_classes, 2, 0.3)
        
    state_dict = (torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict'])    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    train_accuracy = validation_accuracy(model, train_loader, device)
    print('In-distribution accuracy: ', train_accuracy)

    valid_accuracy = validation_accuracy(model, valid_loader, device)
    print('In-distribution accuracy: ', valid_accuracy)



if __name__ =='__main__':
    eval()