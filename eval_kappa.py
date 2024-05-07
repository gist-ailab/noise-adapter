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
    elif args.data == 'nihxray':
        train_loader, valid_loader = utils.get_nihxray(batch_size = batch_size)
    elif args.data == 'dr':
        train_loader, valid_loader, fgadr_loader = utils.get_dr(data_path, batch_size = batch_size)
        
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

    if 'rein' in args.save_path:
        model = rein.ReinsDinoVisionTransformer(
            **variant
        )
    model.load_state_dict(dino_state_dict, strict=False)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.to(device)
    
    model.load_state_dict(torch.load(os.path.join(save_path, 'last.pth.tar'))['state_dict'], strict=False)
    # print(model.state_dict()['blocks.11.mlp.fc2.weight'])
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    model.eval()
    
    if 'ours' in args.save_path:
        # kappa_score = utils.validation_kohen_kappa_ours(model, valid_loader, device)
        accuracy = utils.validation_accuracy_ours(model, valid_loader, device)
        accuracy_fgadr = utils.validation_accuracy_ours(model, fgadr_loader, device)
        
        accuracy_b = utils.validation_balnced_accuracy(model, valid_loader, 'ours', device)
        accuracy_b_fgadr = utils.validation_balnced_accuracy(model, fgadr_loader, 'ours', device)


    elif 'rein' in args.save_path:
        accuracy = utils.validation_accuracy_rein(model, valid_loader, device)
        accuracy_fgadr = utils.validation_accuracy_rein(model, fgadr_loader, device)

        accuracy_b = utils.validation_balnced_accuracy(model, valid_loader, 'rein', device)
        accuracy_b_fgadr = utils.validation_balnced_accuracy(model, fgadr_loader, 'rein', device)
    else:
        # Linear or full
        accuracy = utils.validation_accuracy(model, valid_loader, device)
        accuracy_fgadr = utils.validation_accuracy(model, fgadr_loader, device)

        accuracy_b = utils.validation_balnced_accuracy(model, valid_loader, 'linear', device)
        accuracy_b_fgadr = utils.validation_balnced_accuracy(model, fgadr_loader, 'linear', device)

    print(accuracy, accuracy_fgadr, accuracy_b, accuracy_b_fgadr)

        
if __name__ =='__main__':
    train()