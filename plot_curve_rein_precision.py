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

    model = rein.ReinsDinoVisionTransformer(
        **variant
    )
    model.load_state_dict(dino_state_dict, strict=False)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    model.eval()

    model2 = rein.ReinsDinoVisionTransformer(
        **variant
    )
    model2.load_state_dict(dino_state_dict, strict=False)
    model2.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])
    model2.to(device)
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    model.eval()
    model2.eval()

    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay = 1e-05)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-5)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3, weight_decay = 1e-5)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, lr_decay)

    print(train_loader.dataset[0][0].shape)

    f_pre_a = open(os.path.join(save_path, 'train_pre_adapter.txt'), 'w')
    f_pre_l = open(os.path.join(save_path, 'train_pre_linear.txt'), 'w')

    f_rec_a = open(os.path.join(save_path, 'train_rec_adapter.txt'), 'w')
    f_rec_l = open(os.path.join(save_path, 'train_rec_linear.txt'), 'w')

    f_acc_lp = open(os.path.join(save_path, 'test_lp.txt'), 'w')
    f_acc_ia = open(os.path.join(save_path, 'test_ia.txt'), 'w')
    f_acc_la = open(os.path.join(save_path, 'test_la.txt'), 'w')

    for epoch in range(max_epoch):
        ## training
        model.train()
        model2.train()
        total_loss = 0

        total = 0
        correct = 0

        tpfp = 0
        tp = 0
        tpfn = 0

        tpfp2 = 0
        tp2 = 0
        tpfn2 = 0

        for batch_idx, (inputs, targets, cleans) in enumerate(train_loader):
            inputs, targets, cleans = inputs.to(device), targets.to(device), cleans.to(device)
            optimizer.zero_grad()
            
            
            features_rein = model.forward_features(inputs)
            features_rein = features_rein[:, 0, :]
            outputs = model.linear_rein(features_rein)

            features_rein2 = model2.forward_features(inputs)
            features_rein2 = features_rein2[:, 0, :]
            outputs2 = model2.linear_rein(features_rein2)

            with torch.no_grad():
                # features_ = model.forward_features_no_rein(inputs)
                features_ = model.forward_features_no_rein(inputs)
                features_ = features_[:, 0, :]
            outputs_ = model.linear(features_)

            with torch.no_grad():
                pred = (outputs_).max(1).indices
                linear_accurate = (pred==targets) # Prediction on training set (TP+FP) (1 for Clean)
                true_accurate = (cleans==targets) # Clean or Noise () (1 for Clean)
                correct_accurate = (linear_accurate & true_accurate) # TP
                tpfp += linear_accurate.sum()
                tp += correct_accurate.sum()
                tpfn += true_accurate.sum()

                pred2 = outputs.max(1).indices
                linear_accurate2 = (pred2==targets) # Prediction on training set (TP2+FP2)
                correct_accurate2 = (linear_accurate2 & true_accurate) # TP2
                tpfp2 += linear_accurate2.sum()
                tp2 += correct_accurate2.sum()
                tpfn2 += true_accurate.sum()
            


            loss_rein = linear_accurate*criterion(outputs, targets)
            loss_rein2 = linear_accurate2*criterion(outputs2, targets)
            loss_linear = criterion(outputs_, targets)
            loss = loss_linear.mean()+loss_rein.mean()#+ loss_rein2.mean()
            loss.backward()            
            optimizer.step() # + outputs_

            optimizer2.zero_grad()
            loss_rein2.mean().backward()
            optimizer2.step()

            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()            
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')
        print()  
        train_accuracy = correct/total
        train_avg_loss = total_loss/len(train_loader)

        # Save precision
        print(tp/tpfp ,tp/tpfn, tp , tpfp)
        print(tp2/ tpfp2, tp2/tpfn2, tp2, tpfp2, total)
        f_pre_l.write('{:.4f},'.format(tp/tpfp))
        f_pre_a.write('{:.4f},'.format(tp2/tpfp2))

        f_rec_l.write('{:.4f},'.format(tp/tpfn))
        f_rec_a.write('{:.4f},'.format(tp2/tpfn2))


        ## validation
        model.eval()
        model2.eval()

        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy = utils.validation_accuracy_ours(model2, valid_loader, device)
        valid_accuracy_ = utils.validation_accuracy_ours(model, valid_loader, device)
        valid_accuracy_linear = utils.validation_accuracy_linear(model, valid_loader, device)
        f_acc_lp.write('{:.4f},'.format(valid_accuracy_linear))
        f_acc_ia.write('{:.4f},'.format(valid_accuracy_))
        f_acc_la.write('{:.4f},'.format(valid_accuracy))
        scheduler.step()
        scheduler2.step()
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID_2 [acc - {:.4f}], VALID_1 [acc - {:.4f}], VALID(linear) [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy, valid_accuracy_, valid_accuracy_linear))
        print(scheduler.get_last_lr())
    f_pre_l.close()   
    f_pre_a.close()   
    f_rec_l.close()   
    f_rec_a.close()   
    f_acc_lp.close()   
    f_acc_ia.close()   
    f_acc_la.close()   


if __name__ =='__main__':
    train()