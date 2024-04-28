import os
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
import utils

import random
import rein

from sklearn.metrics import f1_score

import dino_variant
import others.codis as codis


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
        train_loader, valid_loader = utils.get_noise_dataset_with_cleanlabel(data_path, noise_rate=noise_rate, batch_size = batch_size)

    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant

    model = torch.hub.load('facebookresearch/dinov2', model_load)
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
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3, weight_decay = 1e-5)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3, weight_decay = 1e-5)

    criterion = codis.loss_codis_clean
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, lr_decay)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, lr_decay)

    num_gradual = 10
    rate_schedule = np.ones(max_epoch) * noise_rate
    rate_schedule[:num_gradual] = np.linspace(0, noise_rate, num_gradual)

    print(train_loader.dataset[0][0].shape)

    f_pre_codis = open(os.path.join(save_path, 'train_pre_codis.txt'), 'w')
    f_rec_codis = open(os.path.join(save_path, 'train_rec_codis.txt'), 'w')

    f_acc_codis = open(os.path.join(save_path, 'test_codis.txt'), 'w')

    for epoch in range(max_epoch):
        ## training
        model1.train()
        model2.train()
        total_loss = 0

        total = 0
        correct = 0

        tpfp = 0
        tp = 0
        tpfn = 0

        for batch_idx, (inputs, targets, cleans) in enumerate(train_loader):
            inputs, targets, cleans = inputs.to(device), targets.to(device), cleans.to(device)
                        
            # Forward + Backward + Optimize
            features1 = model1.forward_features(inputs)
            features1_ = features1[:, 0, :]
            outputs1 = model1.linear(features1_)

            features2 = model2.forward_features(inputs)
            features2_ = features2[:, 0, :]
            outputs2 = model2.linear(features2_)

            
            loss_1, loss_2, clean_idx = criterion(outputs1, outputs2, targets, rate_schedule[epoch])
            optimizer1.zero_grad()
            loss_1.backward(retain_graph=True)
            optimizer1.step()
            
            optimizer2.zero_grad()
            loss_2.backward(retain_graph=True)
            optimizer2.step()

            with torch.no_grad():
                # linear_accurate = (clean_idx) # Prediction on training set (TP+FP) (1 for Clean)
                true_accurate = (cleans==targets) # Clean or Noise () (1 for Clean)
                correct_accurate = true_accurate[clean_idx] # TP
                tpfp += clean_idx.size(0)
                tp += correct_accurate.sum().item()
                tpfn += true_accurate.sum().item()
                # print(tpfp, tp, tpfn, clean_idx, true_accurate)
            

            total_loss += loss_1
            total += targets.size(0)
            _, predicted = outputs1[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()            
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')
        print()  
        train_accuracy = correct/total
        train_avg_loss = total_loss/len(train_loader)

        # Save precision
        print(tp/tpfp ,tp/tpfn, tp , tpfp)
        f_pre_codis.write('{:.4f},'.format(tp/tpfp))
        f_rec_codis.write('{:.4f},'.format(tp/tpfn))


        ## validation
        model1.eval()
        model2.eval()

        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy = utils.validation_accuracy_rein(model1, valid_loader, device)
        f_acc_codis.write('{:.4f},'.format(valid_accuracy))
        scheduler1.step()
        scheduler2.step()
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID_2 [acc - {:.4f}], VALID_1 [acc - {:.4f}], VALID(linear) [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy, valid_accuracy_, valid_accuracy_linear))
        print(scheduler1.get_last_lr())
    f_pre_codis.close()   
    f_rec_codis.close()   
    f_acc_codis.close()  


if __name__ =='__main__':
    train()