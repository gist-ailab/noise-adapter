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
    elif args.data == 'idrid':
        train_loader, valid_loader = utils.get_idrid_noise_dataset(data_path, noise_rate=noise_rate, batch_size = batch_size)
    elif args.data == 'chaoyang':
        train_loader, valid_loader = utils.get_chaoyang_dataset(data_path, batch_size = batch_size)
    elif 'mnist' in args.data:
        train_loader, valid_loader = utils.get_mnist_noise_dataset(args.data, noise_rate=noise_rate, batch_size = batch_size)
    elif args.data == 'dr':
        train_loader, valid_loader, _, _ = utils.get_dr(data_path, batch_size = batch_size)
    elif 'cifar' in args.data:
        train_loader, valid_loader = utils.get_cifar_noise_dataset(args.data, data_path, batch_size = batch_size,  noise_rate=noise_rate)
    elif args.data == 'clothing':
        train_loader, valid_loader = utils.get_clothing1m_dataset(data_path, batch_size=batch_size)   
        lr_decay = [5, 10]
    elif args.data == 'webvision':
        train_loader, valid_loader = utils.get_webvision(data_path, batch_size=batch_size)   
        
    num_samples = {}
    for i in range(config['num_classes']):
        num_samples[i] = 0
    for sample in train_loader.dataset:
        num_samples[sample[1]]+=1
    print(num_samples)
    
    class_weight = torch.tensor([1-num_samples[x]/sum(num_samples.values()) for x in num_samples])
    print(class_weight)
        
    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
    elif args.netsize == 'b':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant
    elif args.netsize == 'l':
        model_load = dino_variant._large_dino
        variant = dino_variant._large_variant
    # model = timm.create_model(network, pretrained=True, num_classes=2) 
    model = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = model.state_dict()

    model = rein.ReinsDinoVisionTransformer_3_head(
        **variant
    )
    model.load_state_dict(dino_state_dict, strict=False)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.linear_rein1 = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.linear_rein2 = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_linear = torch.nn.CrossEntropyLoss(reduction='none', weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-5)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 1) 
    print(train_loader.dataset[0][0].shape)

    avg_accuracy = 0.0
    avg_kappa = 0.0
    remember_rate = 1 - args.noise_rate
    for epoch in range(max_epoch):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        correct2 = 0
        correct_linear = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            features_rein = model.forward_features1(inputs)
            features_rein = features_rein[:, 0, :]
            outputs = model.linear_rein1(features_rein)

            features_rein2 = model.forward_features2(inputs)
            features_rein2 = features_rein2[:, 0, :]
            outputs2 = model.linear_rein2(features_rein2)

            with torch.no_grad():
                # features_ = model.forward_features_no_rein(inputs)
                features_ = model.forward_features_no_rein(inputs)
                features_ = features_[:, 0, :]
            outputs_ = model.linear(features_)
            # print(outputs.shape, outputs_.shape)

            with torch.no_grad():
                pred = (outputs_).max(1).indices
                linear_accurate = (pred==targets)

                loss_ = criterion(outputs, targets)
                ind_sorted = np.argsort(loss_.cpu().data).cuda()
                num_remember = int(remember_rate * len(ind_sorted))
                ind_update=ind_sorted[:num_remember]

            loss_rein = linear_accurate*criterion(outputs, targets)
            loss_rein2 = criterion(outputs2, targets)[ind_update]
            loss_linear = criterion_linear(outputs_, targets)
            loss = loss_linear.mean() +loss_rein.mean() + loss_rein2.mean()
            loss.backward()            
            optimizer.step() # + outputs_


            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()       

            _, predicted = outputs2[:len(targets)].max(1)            
            correct2 += predicted.eq(targets).sum().item()   

            _, predicted = outputs_[:len(targets)].max(1)            
            correct_linear += predicted.eq(targets).sum().item()   
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc2: %.3f%% | Acc1: %.3f%% | LinearAcc: %.3f%% | (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct2/total, 100.*correct/total, 100.*correct_linear/total, correct, total), end = '')                       
        train_accuracy = correct/total
        train_avg_loss = total_loss/len(train_loader)
        print()

        ## validation
        model.eval()
        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy = utils.validation_accuracy_ours_head3(model, valid_loader, device)
        valid_accuracy_ = utils.validation_accuracy_ours_head3(model, valid_loader, device, use_rein1 = True)
        valid_accuracy_linear = utils.validation_accuracy_linear(model, valid_loader, device)
        
        scheduler.step()
        if epoch >= max_epoch-10:
            avg_accuracy += valid_accuracy 
            # kappa =  utils.validation_kohen_kappa_ours(model, valid_loader, device)
            # avg_kappa += kappa
        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID_2 [acc - {:.4f}], VALID_1 [acc - {:.4f}], VALID(linear) [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy, valid_accuracy_, valid_accuracy_linear))
        print(scheduler.get_last_lr())
    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy/10))
        f.write('|')
        # f.write(str(avg_kappa/10))
if __name__ =='__main__':
    train()