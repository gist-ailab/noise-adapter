import os
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
import utils
import random

from PIL import Image

import dino_variant
import rein


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
    noise_rate = 0.4

    if args.data == 'ham10000':
        train_loader, valid_loader = utils.get_noise_dataset(data_path, noise_rate=noise_rate, batch_size = batch_size)
        clean_loader, _ = utils.get_noise_dataset(data_path, noise_rate=0.0, batch_size = batch_size)
    elif args.data == 'aptos':
        train_loader, valid_loader = utils.get_aptos_noise_dataset(data_path, noise_rate=noise_rate, batch_size = batch_size)
        clean_loader, _ = utils.get_aptos_noise_dataset(data_path, noise_rate=0.0, batch_size = batch_size)

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


    if 'linear' in args.save_path:
        model = torch.hub.load('facebookresearch/dinov2', model_load)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.load_state_dict(torch.load(os.path.join(save_path, 'last.pth.tar'), map_location='cpu')['state_dict'], strict=True)
    elif 'rein' in args.save_path:
        model = rein.ReinsDinoVisionTransformer(
            **variant
        )
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])

        model.load_state_dict(torch.load(os.path.join(save_path, 'last.pth.tar'), map_location='cpu')['state_dict'], strict=True) #TEMP
    model.to(device)
    model.eval()
    
    correct_tp=0
    correct_tn=0
    total_tp = 0
    total_tn = 0

    samples = train_loader.dataset.samples
    clean_samples = clean_loader.dataset.samples
    train_transform, _ = utils.get_transform()
    for batch_idx, (inputs, targets) in enumerate(samples):
        img = Image.open(inputs)
        targets = torch.tensor(targets).unsqueeze(0)
        clean_targets = torch.tensor(clean_samples[batch_idx][1]).unsqueeze(0).to(device)

        avg_outputs = []
        for _ in range(1):
            inputs = train_transform(img)
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.unsqueeze(0)
            
            with torch.no_grad():
                if type(model).__name__ == 'ReinsDinoVisionTransformer':
                    outputs = model.forward_features(inputs)
                    outputs = outputs[:, 0, :]
                    norms = outputs.norm(p=2, dim=1, keepdim=True)
                    outputs = outputs / outputs.norm(p=2, dim=1, keepdim=True)
                    outputs = model.linear_rein(outputs)

                    # outputs_ = model.forward_features_no_rein(inputs)
                    # outputs_ = outputs_[:, 0, :]
                    # outputs_ = model.linear(outputs_)              
                    # outputs = outputs
                else:
                    outputs = model(inputs)
                    outputs = model.linear(outputs)
                avg_outputs.append(outputs)
        avg_outputs = torch.stack(avg_outputs).mean(0)
        # print(avg_outputs.shape)
        _, predicted = outputs[:len(targets)].max(1)            
        # correct += predicted.eq(targets).sum().item()       
        if clean_targets == targets:
            # Clean Label
            if norms > 12:
                correct_tp += predicted.eq(clean_targets).sum().item() 
                total_tp += targets.size(0)

        else:
            # Noisy Label
            if norms > 12:
                correct_tn += predicted.eq(clean_targets).sum().item() 
                total_tn += targets.size(0)

    print(correct_tp/total_tp, total_tp)    
    print(correct_tn/total_tn, total_tn)    
    print((correct_tn+correct_tp)/(total_tn+total_tp), total_tn+total_tp)    



if __name__ =='__main__':
    evalaute()