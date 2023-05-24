import os
import torch
import torch.nn.functional as F
import argparse
import timm
import numpy as np
import utils

import models
import random
random.seed(0)
np.random.seed(0)

def masking_image(x, ratio):
    x_ = x.clone()
    B, _, H, W = x.shape

    mshape = B, 1, round(H / 16), round(W / 16)
    input_mask = torch.rand(mshape, device=x_.device)
    input_mask = (input_mask > ratio).float()
    input_mask = F.interpolate(input_mask, scale_factor=16, mode='nearest')

    # mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
    masked_x = x_ * input_mask
    return masked_x, input_mask

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
    lrde = [50, 75]

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
        lrde = [50, 75]
    elif 'food101n' in args.data:
        train_loader, valid_loader = utils.get_food101n(dataset_path, batch_size)
    elif 'clothing1m' in args.data:
        train_loader, valid_loader = utils.get_clothing1m(dataset_path, batch_size)
        lrde=[40]
    elif 'animal10n' in args.data:
        train_loader, valid_loader = utils.get_animal10n(dataset_path, batch_size)

    if args.net == 'resnet18':
        model = models.ResNet18(num_classes=1000)
        model.load_state_dict(torch.load('/SSDb/yyg/RR/pretrained_resnet18/last.pth.tar', map_location=device)['state_dict'])
        model.fc = torch.nn.Linear(512, num_classes)
    else:
        model = timm.create_model(args.net, pretrained=False, num_classes=num_classes)  
        # import vit
        # model= vit.return_vit_small(num_classes)
        model.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=192,
                out_channels=16*16*3, kernel_size=1),
            torch.nn.PixelShuffle(16),
        )
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    print(utils.validation_accuracy(model, valid_loader, device))
    
    if 'vit' in args.net:
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9, weight_decay = 1e-04)
    else:
        if args.data == 'clothing1m':
            lr = 0.01
        else:
            lr = 0.02
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9, weight_decay = 5e-04)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0002)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lrde)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 2) 
    f = open(save_path + '/record.txt', 'w')
    print(train_loader.dataset[0][0].shape)
    for epoch in range(max_epoch):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx == 1000:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs) 
            
            masked_inputs, mask= masking_image(inputs, 0.6)
            tokens = model.forward_features(masked_inputs) 
            # print(tokens.shape)
            cls_token = tokens[:, 0]
            
            msk_token = tokens[:, 1:]
            B, L, C = msk_token.shape
            H = W = 14
            msk_token = msk_token.permute(0, 2, 1).reshape(B, C, H, W)
            # print(msk_token.shape)
            inputs_rec = model.decoder(msk_token)
            # print(inputs_rec.shape)

            # mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
            loss_rec = F.l1_loss(inputs, inputs_rec, reduction='none')
            loss_rec = (loss_rec * (1-mask)).sum() / ((1-mask).sum() + 1e-5)
            # print(loss_rec.shape)
            loss = loss_rec
            loss.backward()            
            optimizer.step()

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
        model.eval()
        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy = utils.validation_accuracy(model, valid_loader, device)
        scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())
        f.write('{}\t{}\t{}\n'.format(epoch, train_accuracy, valid_accuracy))
    f.close()
if __name__ =='__main__':
    train()