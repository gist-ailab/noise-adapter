import os
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
import utils

import random
import rein
import others.jocor

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
    num_gradual = 10

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
        train_loader, valid_loader, _  = utils.get_dr(data_path, batch_size = batch_size)
    elif 'cifar' in args.data:
        train_loader, valid_loader = utils.get_cifar_noise_dataset(args.data, data_path, batch_size = batch_size,  noise_rate=noise_rate)
    elif args.data == 'clothing':
        train_loader, valid_loader = utils.get_clothing1m_dataset(data_path, batch_size=batch_size)
        lr_decay = [5, 10]
        num_gradual = 1
    elif args.data == 'webvision':
        train_loader, valid_loader = utils.get_webvision(data_path, batch_size=batch_size)   
        
    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
    elif args.netsize == 'b':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant
    elif args.netsize == 'l':
        model_load = dino_variant._large_dino
        variant = dino_variant._large_variant


    model= torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = model.state_dict()

    model1 = rein.ReinsDinoVisionTransformer(
        **variant
    )
    model1.load_state_dict(dino_state_dict, strict=False)
    model1.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model1.to(device)
    
    model2 = rein.ReinsDinoVisionTransformer(
        **variant
    )
    model2.load_state_dict(dino_state_dict, strict=False)
    model2.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model2.to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay = 1e-05)
    optimizer = torch.optim.Adam(list(model1.parameters())+ list(model2.parameters()), lr=1e-3, weight_decay = 1e-5)
    criterion = others.jocor.loss_jocor
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    saver = timm.utils.CheckpointSaver(model1, optimizer, checkpoint_dir= save_path, max_history = 1) 
    print(train_loader.dataset[0][0].shape)

    if args.data == 'dr':
        num_samples = {0: 25810, 1: 2443, 2: 5292, 3: 873, 4: 708}
        class_weight = torch.tensor([1-num_samples[x]/sum(num_samples.values()) for x in num_samples]).to(device)
        print(class_weight)
    else:
        class_weight = None

    exponent = 1 # 0.5, 1 or 2; This parameter is equal to c in Tc for R(T) in Co-teaching paper.
    rate_schedule = np.ones(max_epoch) * noise_rate
    rate_schedule[:num_gradual] = np.linspace(0, noise_rate ** exponent, num_gradual)
    # print(ra)

    avg_accuracy = 0.0
    avg_kappa = 0.0
    for epoch in range(max_epoch):
        ## training
        model1.train()
        model2.train()

        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # Forward + Backward + Optimize
            features1 = model1.forward_features(inputs)
            features1 = features1[:, 0, :]
            outputs1 = model1.linear(features1)

            features2 = model2.forward_features(inputs)
            features2 = features2[:, 0, :]
            outputs2 = model2.linear(features2)


            loss_1, loss_2 = criterion(outputs1, outputs2, targets, rate_schedule[epoch], class_weight=class_weight)
            loss_1.backward()            
            optimizer.step()

            total_loss += loss_1
            total += targets.size(0)
            _, predicted = outputs1[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()            
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')
            train_accuracy = correct/total
                  
        train_avg_loss = total_loss/len(train_loader)
        print()

        ## validation
        model1.eval()
        model2.eval()
        total_loss = 0
        total = 0
        correct = 0

        valid_accuracy = utils.validation_accuracy_rein(model1, valid_loader, device)
        if epoch >= max_epoch-10:
            avg_accuracy += valid_accuracy
            kappa =  utils.validation_kohen_kappa(model1, valid_loader, device)
            avg_kappa += kappa
        scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())

    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy/10))
        f.write('|')
        f.write(str(avg_kappa/10))
    
if __name__ =='__main__':
    train()