import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import argparse

from utils import *
from evaluation import *
import resnet_sigmoid
import resnet_scale


import torch
import torch.nn as nn
import numpy as np

def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'resnet18', type=str)
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--method' ,'-m', default = 'msp', type=str)

    args = parser.parse_args()

    config = read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    dataset_path = config['id_dataset']
    batch_size = config['batch_size']
    save_path = config['save_path'] + args.save_path
    
    num_classes = int(config['num_classes'])

    if 'cifar' in args.data:
        _, valid_loader = get_cifar(args.data, dataset_path, batch_size)
    else:
        valid_loader = get_svhn(dataset_path, batch_size)

    if args.net =='resnet18_sigmoid':
        model = resnet_sigmoid.resnet18(num_classes = num_classes)
    if args.net =='resnet18_scale':
        model = resnet_scale.resnet18(num_classes = num_classes)
    if args.net =='resnet18':
        model = timm.create_model(args.net, pretrained=False, num_classes=num_classes)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        
                
    state_dict = (torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict'])    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if args.method == 'msp':
        calculate_score = calculate_msp
    if args.method == 'odin':
        calculate_score = calculate_odin
    if args.method == 'norm':
        calculate_score = calculate_norm
    if args.method == 'allnm':
        calculate_score = calculate_allnm
    if args.method == 'cosine':
        calculate_score = calculate_cosine
    if args.method == 'energy':
        calculate_score = calculate_energy
    if args.method == 'gradnm':
        calculate_score = calculate_gradnorm
    if args.method == 'react':
        calculate_score = calculate_react
    if args.method == 'md':
        calculate_score = calculate_mdscore
    if args.method == 'godn':
        calculate_score = calculate_godin
    if args.method == 'mls':
        calculate_score = calculate_mls
    if args.method == 'dice':
        calculate_score = calculate_msp
    f = open(save_path+'/{}_result.txt'.format(args.method), 'w')
    valid_accuracy = validation_accuracy(model, valid_loader, device)
    print('In-distribution accuracy: ', valid_accuracy)
        
    f.write('Accuracy for ValidationSet: {}\n'.format(str(valid_accuracy)))
    #MSP
    #image_norm(valid_loader)  
    preds_in = calculate_score(model, valid_loader, device).cpu()
    if 'cifar' in args.data:
        OOD_results(preds_in, model, get_svhn('/SSDe/yyg/data/svhn', batch_size), device, args.method+'-SVHN', f)
    else:
        _, cifar_loader = get_cifar('cifar10', '/SSDe/yyg/data/cifar10', batch_size)
        OOD_results(preds_in, model, cifar_loader, device, args.method+'-CIFAR10', f)     

  
    OOD_results(preds_in, model, get_textures('/SSDe/yyg/data/ood-set/textures/images'), device, args.method+'-TEXTURES', f)
    OOD_results(preds_in, model, get_lsun('/SSDe/yyg/data/ood-set/LSUN'), device, args.method+'-LSUN', f)
    OOD_results(preds_in, model, get_lsun('/SSDe/yyg/data/ood-set/LSUN_resize'), device, args.method+'-LSUN-resize', f)
    OOD_results(preds_in, model, get_lsun('/SSDe/yyg/data/ood-set/iSUN'), device, args.method+'-iSUN', f)
    OOD_results(preds_in, model, get_places('/SSDd/yyg/data/places256'), device, args.method+'-Places365', f)
    OOD_results(preds_in, model, get_mnist('/SSDe/yyg/data/ood-set/mnist', transform_imagenet = False), device, args.method+'-mnist', f)
    OOD_results(preds_in, model, get_fnist('/SSDe/yyg/data/ood-set/fnist'), device, args.method+'-fashion-mnist', f)
    f.close()


if __name__ =='__main__':
    eval()