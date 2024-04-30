import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import timm
import numpy as np
import utils

import random
import rein
import clip
import copy

import dino_variant
from sklearn.metrics import f1_score

import adaptformer

def set_requires_grad(model: nn.Module, keywords):
    """
    notice:key in name!
    """
    requires_grad_names = []
    num_params = 0
    num_trainable = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
        if any(key in name for key in keywords):
            param.requires_grad = True
            requires_grad_names.append(name)
            num_trainable += param.numel()
        else:
            param.requires_grad = False

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--net', default='dinov2', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--noise_rate', '-n', type=float, default=0.4)
    parser.add_argument('--adapter', default='rein', type=str)


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

    tuning_config = argparse.Namespace()

    if args.adapter == 'adaptformer':
        # Adaptformer
        tuning_config.ffn_adapt = True
        tuning_config.ffn_num = 64
        tuning_config.ffn_option="parallel"
        tuning_config.ffn_adapter_layernorm_option="none"
        tuning_config.ffn_adapter_init_option="lora"
        tuning_config.ffn_adapter_scalar="0.1"
        tuning_config.d_model=768
        # VPT
        tuning_config.vpt_on = False
        tuning_config.vpt_num = 1

        tuning_config.fulltune = False
    elif args.adapter == 'vpt':
        # Adaptformer
        tuning_config.ffn_adapt = False
        tuning_config.ffn_num = 64
        tuning_config.ffn_option="parallel"
        tuning_config.ffn_adapter_layernorm_option="none"
        tuning_config.ffn_adapter_init_option="lora"
        tuning_config.ffn_adapter_scalar="0.1"
        tuning_config.d_model=768
        # VPT
        tuning_config.vpt_on = True
        tuning_config.vpt_num = 50

    if args.net == 'dinov2':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant

        model = torch.hub.load('facebookresearch/dinov2', model_load)
        dino_state_dict = model.state_dict()


        if args.adapter == 'rein':
            model = rein.ReinsDinoVisionTransformer(
                **variant
            )
            new_state_dict = dino_state_dict
        if args.adapter == 'adaptformer' or args.adapter == 'vpt':
            new_state_dict = dict()
            for k in dino_state_dict.keys():
                new_k = k.replace("mlp.", "")
                new_state_dict[new_k] = dino_state_dict[k]
            extra_tokens = dino_state_dict['pos_embed'][:, :1]
            src_weight = dino_state_dict['pos_embed'][:, 1:]
            src_weight = src_weight.reshape(1, 37, 37, 768).permute(0, 3, 1, 2)

            dst_weight = F.interpolate(
                src_weight.float(), size=16, align_corners=False, mode='bilinear')
            dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
            dst_weight = dst_weight.to(src_weight.dtype)
            dino_state_dict['pos_embed'] = torch.cat((extra_tokens, dst_weight), dim=1)
            model = adaptformer.VisionTransformer(patch_size=14, tuning_config =  tuning_config)
        model.load_state_dict(new_state_dict, strict=False)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.to(device)  

    elif args.net == 'dinov1':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        variant = dino_variant._dinov1_variant
        dino_state_dict = model.state_dict()

        new_state_dict = dict()
        for k in dino_state_dict.keys():
            new_k = k.replace("mlp.", "")
            new_state_dict[new_k] = dino_state_dict[k]

        print(dino_state_dict.keys())
        if args.adapter == 'rein':
            model = rein.ReinsDinoVisionTransformer(
                **variant
            )
        if args.adapter == 'adaptformer' or args.adapter == 'vpt':
            model = adaptformer.VisionTransformer(patch_size=16, tuning_config =  tuning_config)
        model.load_state_dict(new_state_dict, strict=False)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.to(device)  

    elif args.net == 'clip':
        print(clip.available_models())
        variant = dino_variant._clip_variant
        model, preprocess = clip.load("ViT-B/16", device=device)
        clip_state_dict = model.state_dict()

        state_dict = {}

        for k in clip_state_dict.keys():
            if k.startswith("visual."):
                new_k = k.replace("visual.", "")
                state_dict[new_k] = clip_state_dict[k]
        state_dict["positional_embedding"] = clip_state_dict["positional_embedding"]

        conv1 = clip_state_dict["conv1.weight"]
        state_dict["conv1.weight"] = conv1

        model = rein.ReinsDinoVisionTransformer(
            **variant
        )
        u, w = model.load_state_dict(clip_state_dict, False)
        print(u, w, "are misaligned params in vision transformer")
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.to(device)  

    elif args.net == 'mae':
        variant = dino_variant._dinov1_variant
        dino_state_dict = torch.load('mae_pretrain_vit_base.pth')['model']
        print(dino_state_dict.keys())
        new_state_dict = dict()
        for k in dino_state_dict.keys():
            new_k = k.replace("mlp.", "")
            new_state_dict[new_k] = dino_state_dict[k]

        if args.adapter == 'rein':
            model = rein.ReinsDinoVisionTransformer(
                **variant
            )
        if args.adapter == 'adaptformer' or args.adapter == 'vpt':
            model = adaptformer.VisionTransformer(patch_size=16, tuning_config =  tuning_config)
        model.load_state_dict(new_state_dict, strict=False)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.to(device)  

    print(model)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    model.eval()
    
    model2 = copy.deepcopy(model)
    model2.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])
    model2.to(device)

    if args.adapter == 'adaptformer' or args.adapter == 'vpt':
        set_requires_grad(model, ['adapt', 'linear', 'embeddings'])
        set_requires_grad(model2, ['adapt', 'linear', 'embeddings'])
    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay = 1e-05)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-5)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3, weight_decay = 1e-5)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, lr_decay)
    saver = timm.utils.CheckpointSaver(model2, optimizer, checkpoint_dir= save_path, max_history = 1) 
    print(train_loader.dataset[0][0].shape)

    avg_accuracy = 0.0
    avg_kappa=0
    for epoch in range(max_epoch):
        model.train()
        model2.train()
        total_loss = 0
        total = 0
        correct = 0
        correct2 = 0
        correct_linear = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            features_rein = model.forward_features(inputs)
            if args.adapter == 'rein':
                features_rein = features_rein[:, 0, :]
            outputs = model.linear_rein(features_rein)

            features_rein2 = model2.forward_features(inputs)
            if args.adapter == 'rein':
                features_rein2 = features_rein2[:, 0, :]
            outputs2 = model2.linear_rein(features_rein2)

            with torch.no_grad():
                # features_ = model.forward_features_no_rein(inputs)
                features_ = model.forward_features_no_rein(inputs)
                if args.adapter == 'rein':
                    features_ = features_[:, 0, :]
            outputs_ = model.linear(features_)
            # print(outputs.shape, outputs_.shape)

            with torch.no_grad():
                pred = (outputs_).max(1).indices
                linear_accurate = (pred==targets)

                pred2 = outputs.max(1).indices
                linear_accurate2 = (pred2==targets)

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

            _, predicted = outputs2[:len(targets)].max(1)            
            correct2 += predicted.eq(targets).sum().item()   

            _, predicted = outputs_[:len(targets)].max(1)            
            correct_linear += predicted.eq(targets).sum().item()   
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc2: %.3f%% | Acc1: %.3f%% | LinearAcc: %.3f%% | (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct2/total, 100.*correct/total, 100.*correct_linear/total, correct, total), end = '')                       
        train_accuracy = correct/total
        train_avg_loss = total_loss/len(train_loader)
        print()

        ## validation
        model.eval()
        model2.eval()

        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy = utils.validation_accuracy_ours(model2, valid_loader, device, args.adapter)
        valid_accuracy_ = utils.validation_accuracy_ours(model, valid_loader, device, args.adapter)
        valid_accuracy_linear = utils.validation_accuracy_linear(model, valid_loader, device, args.adapter)
        
        scheduler.step()
        scheduler2.step()
        if epoch >= max_epoch-10:
            avg_accuracy += valid_accuracy 
            kappa =  utils.validation_kohen_kappa_ours(model2, valid_loader, device)
            avg_kappa += kappa
        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID_2 [acc - {:.4f}], VALID_1 [acc - {:.4f}], VALID(linear) [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy, valid_accuracy_, valid_accuracy_linear))
        print(scheduler.get_last_lr())
    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy/10))
        f.write('|')
        f.write(str(avg_kappa/10))
if __name__ =='__main__':
    train()