import os
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
import utils

import random


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
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
    elif 'mnist' in args.data:
        train_loader, valid_loader = utils.get_mnist_noise_dataset(args.data, noise_rate=noise_rate, batch_size = batch_size)
        
    model = timm.create_model('resnet50', pretrained = True, num_classes = config['num_classes'])
    model.to(device)
    
    model2 = timm.create_model('resnet50', pretrained = True, num_classes = config['num_classes'])
    model2.to(device)

    model3 = timm.create_model('resnet50', pretrained = True, num_classes = config['num_classes'])
    model3.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    
    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay = 1e-05)
    params = model.fc.parameters()
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay = 1e-5)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3, weight_decay = 1e-5)
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=1e-3, weight_decay = 1e-5)


    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    scheduler3 = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)

    saver = timm.utils.CheckpointSaver(model3, optimizer, checkpoint_dir= save_path, max_history = 1) 
    print(train_loader.dataset[0][0].shape)
    avg_accuracy = 0.0
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
            
            # outputs = model(inputs)
            with torch.no_grad():
                features = model.forward_features(inputs)
                features = model.global_pool(features)
            outputs = model.fc(features)
            # outputs = model.linear(outputs)
            outputs2 = model2(inputs)

            outputs3 = model3(inputs)

            with torch.no_grad():
                pred = (outputs).max(1).indices
                linear_accurate = (pred==targets)

                pred2 = outputs2.max(1).indices
                linear_accurate2 = (pred2==targets)

            loss_rein = linear_accurate*criterion(outputs2, targets)
            loss_rein2 = linear_accurate2*criterion(outputs3, targets)
            loss_linear = criterion(outputs, targets)

            optimizer.zero_grad()            
            loss_linear.mean().backward()
            optimizer.step()

            optimizer2.zero_grad()
            loss_rein.mean().backward()            
            optimizer2.step() # + outputs_

            optimizer3.zero_grad()
            loss_rein2.mean().backward()
            optimizer3.step()

            total_loss += loss_rein2.mean()
            total += targets.size(0)
            _, predicted = outputs3[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()       

            _, predicted = outputs2[:len(targets)].max(1)            
            correct2 += predicted.eq(targets).sum().item()   

            _, predicted = outputs[:len(targets)].max(1)            
            correct_linear += predicted.eq(targets).sum().item()   

            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc2: %.3f%% | Acc1: %.3f%% | LinearAcc: %.3f%% | (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, 100.*correct2/total, 100.*correct_linear/total, correct, total), end = '')                       
        train_accuracy = correct/total
        train_avg_loss = total_loss/len(train_loader)

        ## validation
        model.eval()
        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy_linear = utils.validation_accuracy_resnet(model, valid_loader, device)
        valid_accuracy_ = utils.validation_accuracy_resnet(model2, valid_loader, device)
        valid_accuracy = utils.validation_accuracy_resnet(model3, valid_loader, device)

        scheduler.step()
        scheduler2.step()
        scheduler3.step()

        if epoch >= max_epoch-10:
            avg_accuracy += valid_accuracy 
        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID_2 [acc - {:.4f}], VALID_1 [acc - {:.4f}], VALID(linear) [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy, valid_accuracy_, valid_accuracy_linear))
        print(scheduler.get_last_lr())

    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy/10))

if __name__ =='__main__':
    train()