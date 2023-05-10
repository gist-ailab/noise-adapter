import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import timm
import numpy as np

import utils
import models

def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def train_adaptation(model, train_loader, epochs, device):
    model.train()
    for e in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

def masking_image(x, ratio):
    x_ = x.clone()
    B, _, H, W = x.shape

    mshape = B, 1, round(H / 4), round(W / 4)
    input_mask = torch.rand(mshape, device=x_.device)
    input_mask = (input_mask > ratio).float()
    input_mask = F.interpolate(input_mask, scale_factor=4, mode='nearest')
    masked_x = x_ * input_mask
    return masked_x
            
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'vit_tiny_patch16_224', type=str)
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--nr', default = 0.2, type=float)
    parser.add_argument('--asym', action='store_true')
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
    

    print(model_name, dataset_path.split('/')[-2], batch_size, class_range)
    
    if not os.path.exists(config['save_path']):
        os.mkdir(config['save_path'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        raise ValueError('save_path already exists')
    
    if 'cifar' in args.data:
        print('asym:', args.asym)
        train_loader, valid_loader = utils.get_cifar_noisy(args.data, dataset_path, batch_size, args.nr, args.asym)
        lrde = [80, 120]
    elif 'food101n' in args.data:
        train_loader, valid_loader = utils.get_food101n(dataset_path, batch_size)
        lrde = [50, 75]
    elif 'clothing1m' in args.data:
        train_loader, valid_loader = utils.get_clothing1m(dataset_path, batch_size)
        lrde = [40]
    elif 'animal10n' in args.data:
        train_loader, valid_loader = utils.get_animal10n(dataset_path, batch_size)
        lrde = [50, 75]
    print(args.net)

    if args.net == 'resnet18':
        model = models.ResNet18(num_classes=num_classes)
        # model.load_state_dict(torch.load('/SSDe/yyg/RR/pretrained_resnet18/last.pth.tar', map_location=device)['state_dict'])
        # model.fc = torch.nn.Linear(512, num_classes)
    elif args.net == 'resnet50':
        model = timm.create_model(args.net, pretrained=False, num_classes=num_classes)  
        model.load_state_dict(torch.load('/SSDe/yyg/RR/dino_resnet50_pretrain.pth', map_location=device), strict=False)
        model.fc = torch.nn.Linear(2048, num_classes)

    else:
        model = timm.create_model(args.net, pretrained=True, num_classes=num_classes)  
    
    model.to(device)

    # train_adaptation(model, train_loader, 5, device)

    ema_model = timm.utils.ModelEmaV2(model, decay = 0.99, device = device)
    

    criterion = torch.nn.CrossEntropyLoss()
    criterion_noreduction = torch.nn.CrossEntropyLoss(reduction='none')

    model.eval()
    print(utils.validation_accuracy(model, valid_loader, device))

    if 'vit' in args.net:
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9, weight_decay = 1e-03)
    elif 'resnet34' in args.net:
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9, weight_decay = 1e-03)
    else:
        if args.data == 'clothing1m':
            lr = 0.002
        else:
            lr = 0.003
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9, weight_decay = 5e-04)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.0002)

    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 2)   
    print(train_loader.dataset[0][0].shape)

    f = open(save_path + '/record.txt', 'w')
    ce_lambda = 1.0
    check = False
    for epoch in range(max_epoch):
        ## training
        model.train()
        ema_model.eval()
        total_loss = 0
        total = 0
        correct = 0
        correct_ema = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            outputs_masking = model(masking_image(inputs, 0.2))
            with torch.no_grad():
                outputs_ema = ema_model.module(inputs)
                outputs_ema = torch.softmax(outputs_ema, dim=1)

                outputs_ema_maskings = []
                for i in range(10):
                    outputs_ema_masking = ema_model.module(masking_image(inputs, 0.2))
                    outputs_ema_maskings.append(outputs_ema_masking.unsqueeze(0))
                outputs_ema_maskings = torch.cat(outputs_ema_maskings, dim=0)
            outputs_ema_maskings = outputs_ema_maskings.mean(0)
            outputs_ema_maskings = torch.softmax(outputs_ema_maskings, dim=1)

            pseudo_label_weight_masking, pseudo_label_ema_masking = outputs_ema_maskings.max(dim=1)
            pseudo_label_weight, pseudo_label_ema = outputs_ema.max(dim=1)
            # print(pseudo_label_weight_masking)
            if check:
                ce_loss = criterion_noreduction(outputs, pseudo_label_ema)* pseudo_label_weight_masking#[pseudo_label_ema_masking == pseudo_label_ema]
                consistency_loss = criterion_noreduction(outputs, pseudo_label_ema_masking) * pseudo_label_weight_masking
                consistency_loss += criterion_noreduction(outputs_masking, pseudo_label_ema) * pseudo_label_weight
                loss = ce_loss.mean() + consistency_loss.mean()
            else:
                ce_loss = criterion(outputs, targets)
                consistency_loss = criterion_noreduction(outputs_masking, pseudo_label_ema) * pseudo_label_weight
                loss = ce_loss + consistency_loss.mean()
            # consistency_loss = softmax_entropy(outputs, outputs_ema)

            loss.backward()            
            optimizer.step()

            ema_model.update(model)
            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()       

            _, predicted_ema = outputs_ema[:len(targets)].max(1)    
            correct_ema += predicted_ema.eq(targets).sum().item()       

            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')                       
        train_accuracy = correct/total
        ema_accuracy = correct_ema/total

        train_avg_loss = total_loss/len(train_loader)
        print()

        valid_accuracy = utils.validation_accuracy(model, valid_loader, device)
        scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        
        if ema_accuracy > train_accuracy and not check:
            check = True

        print(ema_accuracy, train_accuracy, check)
        # print()
        valid_accuracy_ema = utils.validation_accuracy(ema_model.module, valid_loader, device)
        print(valid_accuracy_ema)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())
        f.write('{}\t{}\t{}\t{}\t{}\n'.format(epoch, ema_accuracy, train_accuracy, valid_accuracy_ema, valid_accuracy))
    f.close()
if __name__ =='__main__':
    train()