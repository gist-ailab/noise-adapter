import os
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
import utils

import random
import rein

import dino_variant

def get_clean_noisy_loss(model, train_loader, clean_loader, device):
    model.eval()
    train_set = train_loader.dataset
    clean_set = clean_loader.dataset
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    loss_clean = 0
    loss_noisy = 0
    total_clean = 0
    total_noise = 0


    inputs = []
    targets = []
    true_targets = []

    with torch.no_grad():
        for batch_idx, (traindata, cleandata) in enumerate(zip(train_set, clean_set)):
            input, target = traindata
            _, true_target = cleandata

            inputs.append(input.unsqueeze(0))
            targets.append(torch.tensor(target).unsqueeze(0))
            true_targets.append(torch.tensor(true_target).unsqueeze(0))

            if batch_idx % 128 == 127:
                inputs = torch.cat(inputs)
                targets = torch.cat(targets)
                true_targets = torch.cat(true_targets)

                # print(inputs.shape, targets.shape, true_targets.shape)
                
                inputs, targets, true_targets = inputs.to(device), targets.to(device), true_targets.to(device)

                features_rein = model.forward_features(inputs)
                features_rein = features_rein[:, 0, :]
                # features_rein = features_rein / features_rein.norm(p=2, dim=1, keepdim=True)
                
                outputs = model.linear_rein(features_rein)
            
                loss = criterion(outputs, targets)
                # print(loss)
                # Clean
                total_clean += (targets == true_targets).sum()
                loss_clean += loss[targets == true_targets].sum()
                total_noise += (targets != true_targets).sum()
                loss_noisy += loss[targets != true_targets].sum()
                inputs = []
                targets = []
                true_targets = []

    print(loss_clean, total_clean, loss_clean/total_clean)
    print(loss_noisy, total_noise, loss_noisy/total_noise)
    return loss_clean/total_clean, loss_noisy/total_noise

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
        clean_loader, _ = utils.get_noise_dataset(data_path, noise_rate=0.0, batch_size = batch_size)
        
    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
    model = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = model.state_dict()

    model = rein.ReinsDinoVisionTransformer(
        **variant
    )
    model.load_state_dict(dino_state_dict, strict=False)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    model.eval()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    print(train_loader.dataset[0][0].shape)

    avg_accuracy = 0.0
    f1 = open(os.path.join(save_path, 'loss_clean.txt'), 'w')
    f2 = open(os.path.join(save_path, 'loss_noisy.txt'), 'w')

    for epoch in range(max_epoch):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        correct_linear = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            features_rein = model.forward_features(inputs)
            features_rein = features_rein[:, 0, :]
            # features_rein = features_rein / features_rein.norm(p=2, dim=1, keepdim=True)
            
            outputs = model.linear_rein(features_rein)

            with torch.no_grad():
                features_ = model.forward_features_no_rein(inputs)
                features_ = features_[:, 0, :]
            outputs_ = model.linear(features_)

            with torch.no_grad():
                pred = outputs_.max(1).indices
                linear_accurate = (pred==targets)

            loss_rein = linear_accurate*criterion(outputs, targets)
            loss_linear = criterion(outputs_, targets)
            loss = loss_rein.mean() + loss_linear.mean()
            loss.backward()            
            optimizer.step() # + outputs_


            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()       

            _, predicted = outputs_[:len(targets)].max(1)            
            correct_linear += predicted.eq(targets).sum().item()   
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% | LinearAcc: %.3f%% | (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, 100.*correct_linear/total, correct, total), end = '')                       
        train_accuracy = correct/total
        train_avg_loss = total_loss/len(train_loader)
        print()

        ## validation
        model.eval()
        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy = utils.validation_accuracy_ours(model, valid_loader, device)
        valid_accuracy_linear = utils.validation_accuracy_linear(model, valid_loader, device)
        scheduler.step()
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}], VALID(linear) [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy, valid_accuracy_linear))
        print(scheduler.get_last_lr())
        avg_clean_loss, avg_noise_loss = get_clean_noisy_loss(model, train_loader, clean_loader, device)
        f1.write('{},'.format(avg_clean_loss))
        f2.write('{},'.format(avg_noise_loss))

    f1.close()
    f2.close()

if __name__ =='__main__':
    train()