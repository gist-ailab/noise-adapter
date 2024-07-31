import os
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
import utils

import random
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

import dino_variant


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--noise_rate', '-n', type=float, default=0.2)
    args = parser.parse_args()

    config = utils.read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['id_dataset']
    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])
    noise_rate = args.noise_rate

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    lr_decay = [int(0.5*max_epoch), int(0.75*max_epoch), int(0.9*max_epoch)]

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
    model.neigh = KNeighborsClassifier(n_neighbors=15)
    model.to(device)
    model.eval()

    features_list = []
    targets_list = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
                
        with torch.no_grad():
            outputs = model(inputs)
        features_list.append(outputs.cpu())
        targets_list.append(targets.cpu())
    features_list = torch.cat(features_list, dim=0)
    targets_list = torch.cat(targets_list, dim=0)

    model.neigh.fit(features_list, targets_list)

    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(valid_loader):
        total += targets.size(0)
        with torch.no_grad():
            features = model(inputs.to(device))
            outputs = model.neigh.predict(features.cpu())
        outputs = torch.tensor(outputs)
        correct += outputs.eq(targets).sum().item()
    print(correct/total)
if __name__ =='__main__':
    train()