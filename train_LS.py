import os
import torch
import torch.nn.functional as F
import argparse
import timm
import numpy as np
import utils

import random

def calculate_noise_rate(dataset, data, dataset_path, batch_size):
    clean_loader, _ = utils.get_cifar_noisy(data, dataset_path, batch_size, noisy_rate=0.0)

    cleanset = clean_loader.dataset
    clean_targets = torch.tensor(cleanset.targets)
    filtered_targets = dataset.targets
    print((clean_targets== filtered_targets).sum(), clean_targets.size(0))

    return 1-(clean_targets==filtered_targets).sum()/clean_targets.size(0)

def calculate_features(model, x, index=7):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.act1(x)
    x = model.maxpool(x)

    n = 0
    for i in range(len(model.layer1)):
        x = model.layer1[i](x)
        if n == index:
            return x
        n+=1
    for i in range(len(model.layer2)):
        x = model.layer2[i](x)
        if n == index:
            return x
        n+=1
    for i in range(len(model.layer3)):
        x = model.layer3[i](x)
        if n == index:
            return x
        n+=1
    for i in range(len(model.layer4)):
        x = model.layer4[i](x)
        if n == index:
            return x
        n+=1

def prototype_vector(model, num_layers, num_classes, clean_loader, noise_loader, device):
    model.eval()
    current_dist_ratio = 10 
    with torch.no_grad():
     for n in range(num_layers):
        features_list = []
        target_list = []
        f = calculate_features(model, torch.zeros(1, 3, 32, 32).to(device), n)

        prototypes = torch.zeros(num_classes, f.size(1)).to(device)
        for batch_idx, (data1, data2) in enumerate(zip(clean_loader, noise_loader)):
            input1, target1 = data1
            input2, target2 = data2

            clean_data = input1[target1==target2].to(device)
            clean_target = target1[target1==target2].to(device)
            features = calculate_features(model, clean_data, index=n)
            features = F.adaptive_avg_pool2d(features, [1,1]).view(-1, features.size(1))
            features_list.append(features)
            target_list.append(clean_target)
        features_list = torch.cat(features_list, dim=0)
        target_list = torch.cat(target_list, dim=0)

        for c in range(num_classes):
            features = features_list[target_list==c]
            prototypes[c] = features.mean(dim=0)
        print(prototypes.shape)

        dist_list=[]
        noise_dist_list=[]
        clean_dist_list=[]
        for batch_idx, (data1, data2) in enumerate(zip(clean_loader, noise_loader)):
            input1, target1 = data1
            input2, target2 = data2

            # print(noise_data.size(0), clean_data.size(0))
            features = calculate_features(model, input1.to(device), index=n) # featrues for clean and noisy data
            features = F.adaptive_avg_pool2d(features, [1,1]).view(-1, features.size(1))

            prototypes_ = torch.index_select(prototypes, 0, target2.to(device)) # select prototype vector using corresponding corrupted label (more corrupted)

            # dist = torch.norm(features-prototypes_, p=2, dim=1, keepdim=True)
            # print(prototypes_.shape)
            dist = torch.matmul(F.normalize(features.unsqueeze(1), dim=2), F.normalize(prototypes_.unsqueeze(2), dim=1))
            # dist = torch.matmul(features.unsqueeze(1), prototypes_.unsqueeze(2), dim=1))

            # print(dist.shape)
            # print(dist.shape)
            clean_dist = dist[target1==target2]
            noise_dist = dist[target1!=target2]

            dist_list.append(dist)
            clean_dist_list.append(clean_dist)
            noise_dist_list.append(noise_dist)
        dist_list=torch.cat(dist_list, dim=0)
        clean_dist_list=torch.cat(clean_dist_list, dim=0)
        noise_dist_list=torch.cat(noise_dist_list, dim=0)
        # print(clean_dist_list.shape, noise_dist_list.shape)
        clean_dist_list = (clean_dist_list-dist_list.min())/dist_list.max()
        noise_dist_list = (noise_dist_list-dist_list.min())/dist_list.max()
        dist_ratio = noise_dist_list.mean()/clean_dist_list.mean()
        print('{} layer noise robust ratio: {}, noise_dist: {}, clean_dist: {}'.format(n+1, dist_ratio, noise_dist_list.mean(), clean_dist_list.mean()))

        if dist_ratio < current_dist_ratio:
            current_dist_ratio = dist_ratio
            select = n
    return select

def filter_loader(model, train_loader, layer_index, num_classes, device):
    model.eval()

    with torch.no_grad():
        features_list = []
        target_list = []
        f = calculate_features(model, torch.zeros(1, 3, 32, 32).to(device), layer_index)
        prototypes = torch.zeros(num_classes, f.size(1)).to(device)

        for batch_idx, (data1) in enumerate(train_loader):
            input1, target1 = data1
            input1 = input1.to(device)

            features = calculate_features(model, input1, index=layer_index)
            features = F.adaptive_avg_pool2d(features, [1,1]).view(-1, features.size(1))
            
            features_list.append(features)
            target_list.append(target1.to(device))
        features_list = torch.cat(features_list, dim=0)
        target_list = torch.cat(target_list, dim=0)

        for c in range(num_classes):
            features = features_list[target_list==c]
            prototypes[c] = features.mean(dim=0)
        print(prototypes.shape)

        targets_list = []
        for batch_idx, (data1) in enumerate(train_loader):
            input1, target1 = data1
            input1 = input1.to(device)
            
            features = calculate_features(model, input1.to(device), index=layer_index) # featrues for clean and noisy data
            features = F.adaptive_avg_pool2d(features, [1,1]).view(-1, features.size(1))
            
            features = features.unsqueeze(1)
            # print((features-prototypes.unsqueeze(0)).shape)
            # dist = torch.norm(features-prototypes.unsqueeze(0), p=2, dim=2)
            # print(features.shape, prototypes.T.shape)
            dist = torch.matmul(F.normalize(features, dim=2), F.normalize(prototypes.T, dim=1)).view(-1, num_classes)
            # print(dist.shape)
            targets = (dist).max(dim=1).indices
            targets_list.append(targets)
        targets_list = torch.cat(targets_list, dim=0).cpu()
        train_loader.dataset.targets = targets_list
    return train_loader.dataset

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'resnet18', type=str)
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--nr', default = 0.2, type=float)

    parser.add_argument('--save_path', '-s', type=str)

    parser.add_argument('--save_path_noisy', '-sn1', type=str)
    parser.add_argument('--save_path_noisier', '-sn2', type=str)

    args = parser.parse_args()

    config = utils.read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu

    model_name = args.net
    dataset_path = config['id_dataset']
    save_path = config['save_path'] + args.save_path
    noise_path = config['save_path'] + args.save_path_noisy
    noisier_path = config['save_path'] + args.save_path_noisier

    num_classes = int(config['num_classes'])
    class_range = list(range(0, num_classes))

    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])
    lrde = [100, 150]

    print(model_name, dataset_path.split('/')[-2], batch_size, class_range)
    
    noise_loader, noisier_loader, valid_loader = utils.get_cifar_noisy_and_noisier(args.data, dataset_path, batch_size, args.nr, args.nr+0.2)
    if not os.path.exists(config['save_path']):
        os.mkdir(config['save_path'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        raise ValueError('save_path already exists')
    
    print(args.net)

    if 'resnet' in args.net:
        model = timm.create_model(args.net, pretrained=False, num_classes=num_classes)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)    

        model_noisy = timm.create_model(args.net, pretrained=False, num_classes=num_classes)
        model_noisy.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model_noisy.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)       

        model_noisier = timm.create_model(args.net, pretrained=False, num_classes=num_classes)
        model_noisier.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model_noisier.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)       
        if args.net == 'resnet18':
            num_layers = 8
        if args.net == 'resnet34':
            num_layers = 16
    model.to(device)
    model_noisy.to(device)
    model_noisier.to(device)

    model_noisy.load_state_dict((torch.load(noise_path+'/last.pth.tar', map_location = device)['state_dict']))
    model_noisier.load_state_dict((torch.load(noisier_path+'/last.pth.tar', map_location = device)['state_dict']))
    model_noisy.eval()
    model_noisier.eval()

    selected_layer_index = prototype_vector(model_noisier, num_layers, num_classes, noise_loader, noisier_loader, device)
    print(selected_layer_index)
    filtered_dataset = filter_loader(model_noisy, noise_loader, selected_layer_index, num_classes, device)
    train_loader = torch.utils.data.DataLoader(filtered_dataset, batch_size, shuffle=True, pin_memory=True, num_workers = 4)

    noise_rate = calculate_noise_rate(filtered_dataset, args.data, dataset_path, batch_size)
    print('#Noise Rate of Filtered Dataset: ', noise_rate, selected_layer_index)

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    print(utils.validation_accuracy(model_noisy, valid_loader, device))
    print(utils.validation_accuracy(model_noisier, valid_loader, device))

    
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay = 1e-04)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lrde)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 2)   
    print(train_loader.dataset[0][0].shape)
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
        scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())
if __name__ =='__main__':
    train()