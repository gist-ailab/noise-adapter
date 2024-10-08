import os
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
import utils
import time

from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

import random
import rein

import dino_variant
from sklearn.metrics import f1_score

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
    elif args.data == 'chaoyang':
        train_loader, valid_loader = utils.get_chaoyang_dataset(data_path, batch_size = batch_size)
    elif 'mnist' in args.data:
        train_loader, valid_loader = utils.get_mnist_noise_dataset(args.data, noise_rate=noise_rate, batch_size = batch_size)
    elif args.data == 'dr':
        train_loader, valid_loader = utils.get_dr(data_path, batch_size = batch_size)
    elif 'cifar' in args.data:
        train_loader, valid_loader = utils.get_cifar_noise_dataset(args.data, data_path, batch_size = batch_size,  noise_rate=noise_rate)
    elif args.data == 'clothing':
        train_loader, valid_loader = utils.get_clothing1m_dataset(data_path, batch_size=batch_size)
        lr_decay = [5, 10]
    elif args.data == 'webvision':
        train_loader, valid_loader = utils.get_webvision(data_path, batch_size=batch_size)   
    elif args.data == 'animal10n':
        train_loader, valid_loader = utils.get_animal10n(data_path, batch_size=batch_size)   
        
    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
    elif args.netsize == 'b':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant
    elif args.netsize == 'l':
        model_load = dino_variant._large_dino
        variant = dino_variant._large_variant


    dino = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = dino.state_dict()
    new_state_dict = dict()
    for k in dino_state_dict.keys():
        new_k = k.replace("attn.qkv", "attn.qkv.qkv")
        new_state_dict[new_k] = dino_state_dict[k]
    model = rein.LoRADinoVisionTransformer(dino)
    model.dino.load_state_dict(new_state_dict, strict=False)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.to(device)
    
    # print(model)
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 1) 
    model.train()
    print('model: ', sum(p.numel() for p in model.parameters()))
    print('model_adapter: ', sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'linear' in n))
    
    # f = open(os.path.join(save_path, 'epoch_acc.txt'), 'w')
    scaler = GradScaler()
    avg_accuracy = 0.0

    for epoch in range(max_epoch):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with autocast(enabled=True):
                features = model.forward_features(inputs)
                outputs = model.linear(features)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()            
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')
        train_accuracy = correct/total
        end_time = time.time()
        train_avg_loss = total_loss/len(train_loader)
        print()
        print(end_time-start_time)

        ## validation
        model.eval()
        total_loss = 0
        total = 0
        correct = 0

        valid_accuracy = utils.validation_accuracy_lora(model, valid_loader, device)
        if epoch >= max_epoch-10:
            avg_accuracy += valid_accuracy 
        scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())

    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy/10))
    
if __name__ =='__main__':
    train()