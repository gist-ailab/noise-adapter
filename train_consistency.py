import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import timm
import numpy as np
import utils

from kmeans_pytorch import kmeans


def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'vit_tiny_patch16_224', type=str)
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--nr', default = 0.2, type=float)

    args = parser.parse_args()
    config = utils.read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu

    model_name = args.net
    dataset_path = config['id_dataset']
    save_path = config['save_path'] + args.save_path
    num_classes = int(config['num_classes'])
    class_range = list(range(0, num_classes))

    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])
    lrde = [50, 75]

    print(model_name, dataset_path.split('/')[-2], batch_size, class_range)
    
    if not os.path.exists(config['save_path']):
        os.mkdir(config['save_path'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        raise ValueError('save_path already exists')
    
    if 'cifar' in args.data:
        train_loader, valid_loader = utils.get_cifar_noisy(args.data, dataset_path, batch_size, args.nr)
    print(args.net)

    model = timm.create_model(args.net, pretrained=True, num_classes=num_classes)  
    model.to(device)

    ema_model = timm.utils.ModelEmaV2(model, decay = 0.999, device = device)
    
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    print(utils.validation_accuracy(model, valid_loader, device))
    
    if 'vit' in args.net:
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.003, momentum=0.9, weight_decay = 1e-04)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9, weight_decay = 1e-04)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lrde)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 2)   
    print(train_loader.dataset[0][0].shape)
    ce_lambda = 1.0
    for epoch in range(max_epoch):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            outputs_ema = ema_model.module(inputs)
            
            ce_loss = criterion(outputs, targets)
            consistency_loss = softmax_entropy(outputs, outputs_ema).mean(0)
            
            loss = ce_lambda * ce_loss + consistency_loss
            loss.backward()            
            optimizer.step()

            ema_model.update(model)
            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()            
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')                       
        train_accuracy = correct/total
        train_avg_loss = total_loss/len(train_loader)
        print()

        ## validation
        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy = utils.validation_accuracy(model, valid_loader, device)
        scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        # print(utils.validation_accuracy(ema_model.module, train_loader, device), utils.validation_accuracy(ema_model.module, valid_loader, device))
        ema_accuracy = utils.validation_accuracy(ema_model.module, train_loader, device)
        if ema_accuracy < (train_accuracy + 0.002) and ema_accuracy > (train_accuracy - 0.002):
            ce_lambda = 0.0
            
        print(ema_accuracy, train_accuracy, ce_lambda)
        print(utils.validation_accuracy(ema_model.module, valid_loader, device))
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())
if __name__ =='__main__':
    train()