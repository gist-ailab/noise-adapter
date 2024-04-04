import os
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
import utils

import random
import rein

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--noise_rate', '-n', type=float, default=0.2)
    parser.add_argument('--teacher', '-t', type=str)
    args = parser.parse_args()

    config = utils.read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['id_dataset']
    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])
    noise_rate = args.noise_rate
    teacher_path = os.path.join(config['save_path'], args.teacher, 'last.pth.tar')

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    lr_decay = [int(0.5*max_epoch), int(0.75*max_epoch), int(0.9*max_epoch)]

    train_loader, valid_loader = utils.get_noise_dataset(data_path, noise_rate=noise_rate)

    # model = timm.create_model(network, pretrained=True, num_classes=2) 
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dino_state_dict = model.state_dict()
    model = rein.ReinsDinoVisionTransformer(
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        img_size=518,
        ffn_layer="mlp",
        init_values=1e-05,
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True
    )
    model.load_state_dict(dino_state_dict, strict=False)
    model.requires_grad_(False)
    model.linear_rein = nn.Linear(384, config['num_classes'])
    model.to(device)
    
    # print(model.state_dict()['blocks.11.mlp.fc2.weight'])
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    model.eval()
    
    teacher = rein.ReinsDinoVisionTransformer(
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        img_size=518,
        ffn_layer="mlp",
        init_values=1e-05,
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True
    )
    teacher.linear_rein = nn.Linear(384, config['num_classes'])
    teacher.load_state_dict(torch.load(teacher_path, map_location='cpu')['state_dict'], strict=False)
    teacher.to(device)
    teacher.eval()
    valid_accuracy = utils.validation_accuracy_ours(teacher, valid_loader, device)
    print('Teacher accuracy on test set: ', valid_accuracy)
    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay = 1e-05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay = 1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 1) 
    print(train_loader.dataset[0][0].shape)


    avg_accuracy = 0.0
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
            outputs = model.linear_rein(features_rein)
            
            # print(outputs.shape, outputs_.shape)

            with torch.no_grad():
                features_rein = teacher.forward_features(inputs)
                features_rein = features_rein[:, 0, :]
                outputs_ = teacher.linear_rein(features_rein)
            
                pred = outputs_.max(1).indices
                linear_accurate = (pred==targets)
                # print(pred.shape, targets.shape)
            # print(model.reins.state_dict())

            loss_rein = linear_accurate*criterion(outputs, targets)
            loss = loss_rein.mean()
            loss.backward()            
            optimizer.step()


            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item() 
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% | (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')                       
        train_accuracy = correct/total
        train_avg_loss = total_loss/len(train_loader)
        print()

        ## validation
        model.eval()
        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy = utils.validation_accuracy_ours(model, valid_loader, device)
        scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())

if __name__ =='__main__':
    train()