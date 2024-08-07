import os
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
import utils

import random
import rein
import open_clip

import dino_variant
from sklearn.metrics import f1_score

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--net', default='dinov2', type=str)
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
    elif args.data == 'nihchest':
        train_loader, valid_loader = utils.get_nihxray(data_path, batch_size = batch_size)
    elif args.data == 'idrid':
        train_loader, valid_loader = utils.get_idrid_noise_dataset(data_path, noise_rate=noise_rate, batch_size = batch_size)
    elif args.data == 'chaoyang':
        train_loader, valid_loader = utils.get_chaoyang_dataset(data_path, batch_size = batch_size)
    elif 'mnist' in args.data:
        train_loader, valid_loader = utils.get_mnist_noise_dataset(args.data, noise_rate=noise_rate, batch_size = batch_size)
    elif args.data == 'dr':
        train_loader, valid_loader = utils.get_dr(data_path, batch_size = batch_size)


    if args.net == 'dinov2':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant

        model = torch.hub.load('facebookresearch/dinov2', model_load)
        dino_state_dict = model.state_dict()
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.to(device)  

    elif args.net == 'dinov1':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        variant = dino_variant._dinov1_variant
        dino_state_dict = model.state_dict()
        print(dino_state_dict.keys())
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.to(device)  

    elif args.net == 'clip':
        # print(open_clip.list_pretrained())
        variant = dino_variant._clip_variant
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
        clip_state_dict = model.state_dict()

        state_dict = {}

        for k in clip_state_dict.keys():
            if k.startswith("visual."):
                new_k = k.replace("visual.transformer.res", "")
                state_dict[new_k] = clip_state_dict[k]

        # if "positional_embedding" in clip_state_dict.keys():
        #     if (
        #         self.positional_embedding.shape
        #         != state_dict["positional_embedding"].shape
        #     ):
        #         print(
        #             f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}'
        #         )
        #         cls_pos = state_dict["positional_embedding"][0:1, :]
        #         leng = int(state_dict["positional_embedding"][1:,].shape[-2] ** 0.5)
        #         spatial_pos = F.interpolate(
        #             state_dict["positional_embedding"][1:,]
        #             .reshape(1, leng, leng, self.width)
        #             .permute(0, 3, 1, 2),
        #             size=(self.spatial_size, self.spatial_size),
        #             mode="bilinear",
        #         )
        #         spatial_pos = spatial_pos.reshape(
        #             self.width, self.spatial_size * self.spatial_size
        #         ).permute(1, 0)
        #         positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
        #         assert (
        #             self.positional_embedding.shape
        #             == state_dict["positional_embedding"].shape
        #         )
        state_dict["pos_embed"] = clip_state_dict["positional_embedding"]

        # conv1 = clip_state_dict["conv1.weight"]
        # state_dict["conv1.weight"] = conv1

        model = rein.ReinsDinoVisionTransformer(
            **variant
        )
        u, w = model.load_state_dict(state_dict, True)
        print(u, w, "are misaligned params in vision transformer")

        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.to(device)  

    elif args.net == 'mae':
        variant = dino_variant._dinov1_variant
        dino_state_dict = torch.load('mae_pretrain_vit_base.pth')['model']
        print(dino_state_dict.keys())
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        model.load_state_dict(dino_state_dict, strict=True)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.to(device)  

    elif args.net == 'bioCLIP':
        from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        # print(model)
        model = model.visual.trunk
        bioCLIP_state_dict = model.state_dict()
        # print(state_dict)
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        model.load_state_dict(bioCLIP_state_dict, strict=True)
        model.linear = nn.Linear(768, config['num_classes'])
        model.to(device)  
        
    print(model)
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    
    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay = 1e-05)
    optimizer = torch.optim.Adam(model.linear.parameters(), lr=1e-3, weight_decay = 1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 1) 
    print(train_loader.dataset[0][0].shape)

    # f = open(os.path.join(save_path, 'epoch_acc.txt'), 'w')
    avg_accuracy = 0.0
    for epoch in range(max_epoch):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                outputs = model(inputs)
            outputs = model.linear(outputs)
            loss = criterion(outputs, targets)
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
        if epoch >= max_epoch-10:
            avg_accuracy += valid_accuracy 
        scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())

    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy/10))
    
if __name__ =='__main__':
    train()