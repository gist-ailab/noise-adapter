import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import timm
import numpy as np
import utils

from kmeans_pytorch import kmeans

def random_masking(self, x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore

def calculate_cluster_target(model, loader, num_classes, num_cluster, device):
    predictions = []
    cluster_id_list = []

    with torch.no_grad():
        for batch_idx, (inputs, targets, cluster_ids, cluster_centers) in enumerate(loader):
            inputs, targets, cluster_ids, cluster_centers = inputs.to(device), targets.to(device), cluster_ids.to(device), cluster_centers.to(device)

            outputs = model(inputs)
            _, preds = outputs.max(dim=1)

            predictions.append(preds)
            cluster_id_list.append(cluster_ids)
    predictions = torch.cat(predictions, dim=0)
    cluster_id_list = torch.cat(cluster_id_list, dim=0)

    cluster2target = {}
    for i in range(num_cluster):
        preds = predictions[cluster_id_list==i]
        classes, counts = torch.unique(preds, return_counts=True)
        # print(classes, counts, counts.max(0))
        selected_class = classes[counts.max(0).indices.item()]
        print(classes, counts, selected_class)
        cluster2target[i] = selected_class
    return cluster2target

def forward_embeddings(model, loader, device):
    embeddings = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            features_t = model.forward_features(inputs)[:, 0]
            # features_t = model.global_pool(features_t).view(-1, 512)
            # features_t = features_t / torch.norm(features_t, dim=1, keepdim=True)

            embeddings.append(features_t)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings

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

    num_clusters = 200
    print(model_name, dataset_path.split('/')[-2], batch_size, class_range)
    
    if not os.path.exists(config['save_path']):
        os.mkdir(config['save_path'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        raise ValueError('save_path already exists')
    
    if 'cifar' in args.data:
        train_loader, valid_loader = utils.get_cifar_noisy(args.data, dataset_path, batch_size, args.nr)
    elif 'food101n' in args.data:
        train_loader, valid_loader = utils.get_food101n(dataset_path, batch_size)

    print(args.net)

    model = timm.create_model(args.net, pretrained=True, num_classes=num_classes)  
    model.to(device)

    ema_model = timm.utils.ModelEmaV2(model, decay = 0.999, device = device)
    
    embeddings = forward_embeddings(model, train_loader, device)
    cluster_ids, cluster_centers = kmeans(X=embeddings, num_clusters = num_clusters, distance='euclidean', device= device)
    print(cluster_centers.shape)
    print(cluster_ids)


    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    print(utils.validation_accuracy(model, valid_loader, device))
    
    train_loader = utils.modify_train_loader_with_cluster(train_loader, cluster_ids, cluster_centers)
    cluster_target = calculate_cluster_target(ema_model.module, train_loader, num_classes, num_clusters, device)

    if 'vit' in args.net:
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.003, momentum=0.9, weight_decay = 1e-04)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9, weight_decay = 1e-04)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lrde)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 2)   
    print(train_loader.dataset[0][0].shape)

    ce_lambda = 1.0
    check = False
    for epoch in range(max_epoch):
        ## training
        model.train()
        ema_model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets, cluster_ids, cluster_centers) in enumerate(train_loader):
            inputs, targets, cluster_ids, cluster_centers = inputs.to(device), targets.to(device), cluster_ids.to(device), cluster_centers.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            outputs_ema = ema_model.module(inputs)
            
            if check:
                for i in range(len(cluster_ids)):
                    cluster_ids[i] = cluster_target[cluster_ids[i].item()]
                ce_loss = criterion(outputs, cluster_ids)
            
            else:
                ce_loss = criterion(outputs, targets)
            consistency_loss = F.mse_loss(outputs, outputs_ema)
            
            loss = ce_loss + consistency_loss
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
        ema_accuracy = utils.validation_accuracy_cluster(ema_model.module, train_loader, device)
        
        if ema_accuracy < (train_accuracy + 0.002) and ema_accuracy > (train_accuracy - 0.002) and not check:
            check = True
            train_loader, valid_loader = utils.get_cifar_noisy(args.data, dataset_path, batch_size, args.nr)
            embeddings = forward_embeddings(model, train_loader, device)
            cluster_ids, cluster_centers = kmeans(X=embeddings, num_clusters = num_clusters, distance='euclidean', device= device)
            train_loader = utils.modify_train_loader_with_cluster(train_loader, cluster_ids, cluster_centers)
            cluster_target = calculate_cluster_target(ema_model.module, train_loader, num_classes, num_clusters, device)

        print(ema_accuracy, train_accuracy, check)
        print(utils.validation_accuracy(ema_model.module, valid_loader, device))
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())
if __name__ =='__main__':
    train()