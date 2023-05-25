import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import timm
import numpy as np

import utils
import models
import losses

def softmax_entropy(x, target):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(target * x.log_softmax(1)).sum(1)

def mask_label(input_mask, pseudo_label, ratio, num_classes=10):
    label_weight = ratio
    targets = F.one_hot(pseudo_label)
    uniform = torch.ones_like(targets)/targets.size(1)

    return (1-label_weight) * targets + label_weight * uniform

def masking_image(x, ratio, size = 16):
    with torch.no_grad():
        x_ = x.clone()
        B, _, H, W = x.shape

        mshape = B, 1, round(H / size), round(W / size)
        input_mask = torch.rand(mshape, device=x_.device)
        input_mask = (input_mask > ratio).float()
        input_mask = F.interpolate(input_mask, scale_factor=size, mode='nearest')
        masked_x = x_ * input_mask
    return masked_x, input_mask

def return_am(gap_features, img_size = 32):
    B, C, W, H = gap_features.shape
    cam = gap_features.mean(1, keepdim=True)
    cam_min = cam.view(B, 1, W*H).min(dim=2, keepdim=True).values
    cam = cam - cam_min.view(B, 1, 1, 1)
    cam_max = cam.view(B, 1, W*H).max(dim=2, keepdim=True).values
    cam = cam/cam_max.view(B,1,1,1)
    size_factor = int(32/W)
    cam = F.interpolate(cam, scale_factor=size_factor, mode='bilinear')
    return cam

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'resnet18', type=str)
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
        lrde = [50, 75]
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

    optimizer = torch.optim.SGD(model.parameters(), lr = 0.02, momentum=0.9, weight_decay = 1e-03)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150])
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 2)   
    print(train_loader.dataset[0][0].shape)

    f = open(save_path + '/record.txt', 'w')
    ce_lambda = 1.0
    check = False
    for epoch in range(300):
        ## training
        model.train()
        ema_model.eval()
        total_loss = 0
        total = 0
        correct = 0
        correct_ema = 0
        supcon_loss_total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                features_ema = ema_model.module.forward_features(inputs)
                out = F.avg_pool2d(features_ema, 4)
                out = out.view(out.size(0), -1)
                outputs_ema = ema_model.module.fc(out)

            pseudo_label_weight, pseudo_label_ema = outputs_ema.max(dim=1)
            outputs = model(inputs)

            if check:
                # After Mean teacher is learn easy data.
                # Maksing loss is used.
                # Easy data (Correctly labeled data) => Learng masked image modeling
                # easy_data = inputs[pseudo_label_ema == targets]
                easy_label = pseudo_label_ema[pseudo_label_ema == targets]
                
                # CE loss for Easy data
                outputs_easy = outputs[pseudo_label_ema == targets]
                ce_loss = criterion_noreduction(outputs_easy, easy_label)

                # Mask Loss
                masked_image, input_mask = masking_image(inputs, 0.25)
                outputs_masking = model(masked_image)
                # print(outputs_masking.shape, pseudo_label_ema.shape)
                # ema_am = return_am(features_ema)
                label_masking = mask_label(input_mask, pseudo_label_ema, 0.25)
                mic_loss = softmax_entropy(outputs_masking, label_masking)
                    
                with torch.no_grad():
                    masked_image, input_mask = masking_image(inputs, 0.25)
                    outputs_ema_masking = ema_model.module(masked_image)
                softmax_masking = outputs_ema_masking.softmax(1)
                softmax_masking = softmax_masking.max(1).values
                mp_loss = criterion_noreduction(outputs, pseudo_label_ema)[softmax_masking>0.75]
                # mp_loss = 0
                

                loss = ce_loss.mean() + mic_loss.mean() #+ mp_loss.mean()
            else:
                # Training is started.
                # softmax entropy between mean teacher and student is used.
                ce_loss = criterion(outputs, targets)
                #consistency_loss = softmax_entropy(outputs, outputs_ema) #criterion_noreduction(outputs, pseudo_label_ema) * pseudo_label_weight
                loss = ce_loss #+ consistency_loss.mean()
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
        if check:
            print(softmax_masking)

        valid_accuracy = utils.validation_accuracy(model, valid_loader, device)
        scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        
        if ema_accuracy > train_accuracy and not check:
            check = True

        print(ema_accuracy, train_accuracy, check, supcon_loss_total/len(train_loader))
        # print()
        valid_accuracy_ema = utils.validation_accuracy(ema_model.module, valid_loader, device)
        print(valid_accuracy_ema)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())
        f.write('{}\t{}\t{}\t{}\t{}\n'.format(epoch, ema_accuracy, train_accuracy, valid_accuracy_ema, valid_accuracy))
    f.close()
if __name__ =='__main__':
    train()