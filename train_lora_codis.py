import os
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
import utils

import random
import rein
import others.codis as codis
import time

import dino_variant
from sklearn.metrics import f1_score

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--noise_rate', '-n', type=float, default=0.35)
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
        train_loader, valid_loader, _ = utils.get_dr(data_path, batch_size = batch_size)
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


    dino = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = dino.state_dict()

    model1 = rein.LoRADinoVisionTransformer(dino)
    model1.load_state_dict(dino_state_dict, strict=False)
    model1.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model1.to(device)

    dino = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = dino.state_dict()

    model2 = rein.LoRADinoVisionTransformer(dino)
    model2.load_state_dict(dino_state_dict, strict=False)
    model2.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model2.to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay = 1e-05)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

    criterion = codis.loss_codis
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, lr_decay)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, lr_decay)

    saver = timm.utils.CheckpointSaver(model1, optimizer1, checkpoint_dir= save_path, max_history = 1) 
    print(train_loader.dataset[0][0].shape)

    rate_schedule = np.ones(max_epoch) * noise_rate
    rate_schedule[:num_gradual] = np.linspace(0, noise_rate, num_gradual)

    if args.data == 'dr':
        num_samples = {0: 25810, 1: 2443, 2: 5292, 3: 873, 4: 708}
        class_weight = torch.tensor([1-num_samples[x]/sum(num_samples.values()) for x in num_samples]).to(device)
        print(class_weight)
    else:
        class_weight = None
    avg_accuracy = 0.0
    for epoch in range(max_epoch):
        ## training
        model1.train()
        model2.train()

        total_loss = 0
        total = 0
        correct = 0
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward + Backward + Optimize
            features1 = model1.forward_features(inputs)
            outputs1 = model1.linear(features1)

            features2 = model2.forward_features(inputs)
            outputs2 = model2.linear(features2)

            
            loss_1, loss_2 = criterion(outputs1, outputs2, targets, rate_schedule[epoch], class_weight=class_weight)
            optimizer1.zero_grad()
            loss_1.backward(retain_graph=True)
            optimizer1.step()
            
            optimizer2.zero_grad()
            loss_2.backward(retain_graph=True)
            optimizer2.step()

            total_loss += loss_1.item()
            total += targets.size(0)
            _, predicted = outputs1[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()            
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')
        train_accuracy = correct/total

        end_time = time.time()
        print(end_time - start_time)    
        train_avg_loss = total_loss/len(train_loader)
        print()

        ## validation
        model1.eval()
        model2.eval()
        total_loss = 0
        total = 0
        correct = 0

        valid_accuracy = utils.validation_accuracy_lora(model1, valid_loader, device)
        if epoch >= max_epoch-10:
            avg_accuracy += valid_accuracy 
        scheduler1.step()
        scheduler2.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler1.get_last_lr())

    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy/10))
    
if __name__ =='__main__':
    train()