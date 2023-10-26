import os
import torch
import torch.nn.functional as F
import argparse
import timm
import numpy as np
import utils

import random

def train():
    device = 'cuda:0'
    data_path = 'data'
    network = 'resnet18'
    max_epoch = 40
    save_path = 'checkpoints/normalization_0.2_1'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    lr_decay = [int(0.5*max_epoch), int(0.75*max_epoch)]

    train_loader, valid_loader = utils.get_loader(data_path, noise_rate = 0.2, shuffle = False)

    model = timm.create_model(network, pretrained=True, num_classes=2) 
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(os.path.join(save_path, 'last.pth.tar'))['state_dict'])

    valid_accuracy = utils.validation_accuracy(model, valid_loader, device)


    train_accuracy = utils.validation_accuracy(model, train_loader, device)

    print(train_accuracy, valid_accuracy)


    gt = dict()

    with open('gt.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            gt[line[0]] = int(line[1])
    
    gt_list = list(gt.values())
    print(gt_list)


    conf_list = []
    pred_list = []
    target_list = [] 
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        features = model.forward_features(inputs)
        features = model.global_pool(features).view(-1, 512)
        features = F.normalize(features)
        outputs = model.fc(features)
        outputs = outputs.softmax(1)
        _, pred = outputs.max(1)
        conf = torch.gather(outputs, dim=1, index= targets.view(-1, 1))
        # print(conf.shape, outputs.shape)

        conf_list.append(conf)
        pred_list.append(pred)
        target_list.append(targets)
    conf_list = torch.cat(conf_list, dim=0)
    pred_list = torch.cat(pred_list, dim=0)
    target_list = torch.cat(target_list, dim=0)


    print(conf_list, conf_list.shape)
    print(pred_list)

    # sort = torch.sort(conf_list, descending=True).values
    # thr = sort[int(len(sort)*0.95)]

    # print(thr)

    correct = 0
    total = 0

    # noise_filtered = 0
    filtered_noise = 0
    filtered_total = 0

    unfiltered_noise = 0
    unfiltered_total = 0

    f = open('filter.txt', 'w')
    for i, data in enumerate(zip(conf_list, pred_list, target_list)):
        if data[0] < 0.5:
            # Filter
            # print('wrong', data[0])

            pred = data[1]
            filtered_total += 1
            if not data[2] == gt_list[i]: #This is a Noise data
                filtered_noise += 1
        else:
            # Pass            
            # print('correct', data[0])

            pred = data[2]
            unfiltered_total += 1
            
            if not data[2] == gt_list[i]:
                unfiltered_noise += 1

        f.write('{}\n'.format(pred))
        if pred == gt_list[i]:
            correct +=1
        total+=1
    f.close()
    # print(total, correct)
