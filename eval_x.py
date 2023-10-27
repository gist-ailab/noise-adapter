import os
import torch
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
    save_path = 'checkpoints/filtered_baseline_0.2_1'
    filter_path = 'filter.txt'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    lr_decay = [int(0.5*max_epoch), int(0.75*max_epoch)]

    # train_loader, valid_loader = utils.get_loader(data_path, noise_rate = 0.2, filter_path=filter_path)

    # model = timm.create_model(network, pretrained=True, num_classes=2) 
    # model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    
    model.load_state_dict(torch.load(save_path))

    print(validation_accuracy(model, valid_loader, device))

if __name__ =='__main__':
    train()