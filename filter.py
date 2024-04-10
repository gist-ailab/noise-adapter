import os
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
import utils

import random

import dino_variant


def evalaute():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    args = parser.parse_args()

    config = utils.read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['id_dataset']
    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])
    noise_rate = args.noise_rate

    if args.data == 'ham10000':
        train_loader, valid_loader = utils.get_noise_dataset(data_path, noise_rate=noise_rate, batch_size = batch_size)
    elif args.data == 'aptos':
        train_loader, valid_loader = utils.get_aptos_noise_dataset(data_path, noise_rate=noise_rate, batch_size = batch_size)
    elif args.data == 'nihchest':
        train_loader, valid_loader = utils.get_nihxray(data_path, batch_size = batch_size)
    elif args.data == 'idrid':
        train_loader, valid_loader = utils.get_idrid_noise_dataset(data_path, noise_rate=noise_rate, batch_size = batch_size)


    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
    elif args.netsize == 'b':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant
    elif args.netsize == 'l':
        model_load = dino_variant._large_dino
        variant = dino_variant._large_variant

    model = torch.hub.load('facebookresearch/dinov2', model_load)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(os.path.join(save_path, 'last.pth.tar'), map_location='cpu')['state_dict'])
    
    total=0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
        outputs = model.linear(outputs)
        total += targets.size(0)
        _, predicted = outputs[:len(targets)].max(1)            
        correct += predicted.eq(targets).sum().item()       

    print(correct/total)    

if __name__ =='__main__':
    evalaute()