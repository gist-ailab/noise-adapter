import torch
import torch.nn.functional as F
import timm
import argparse

from torch.autograd import Variable

from utils import *

react_threshold = None
#MSP
def calculate_msp(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = torch.softmax(outputs, dim=1)
            outputs = outputs.max(dim=1).values
            predictions.append(outputs)
    predictions = torch.cat(predictions).to(device)
    return predictions

#MLS
def calculate_mls(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.max(dim=1).values
            predictions.append(outputs)
    predictions = torch.cat(predictions).to(device)
    return predictions

#ODIN
"""
ODIN method
original code is on https://github.com/facebookresearch/odin
"""
def calculate_odin(model, loader, device):
    model.eval()
    predictions = []
    for batch_idx, (inputs,_) in enumerate(loader):
        inputs = inputs.to(device)
        inputs = Variable(inputs, requires_grad = True)
        outputs = model(inputs)
        
        #label
        labels = outputs.data.max(1)[1]
        labels = Variable(labels)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float()-0.5)*2
        gradient[:, 0] = gradient[:, 0]
        gradient[:, 1] = gradient[:, 1]
        gradient[:, 2] = gradient[:, 2]
        temp_inputs = torch.add(inputs.data, -0.0004* gradient)
        temp_inputs = Variable(temp_inputs)

        with torch.no_grad():
            outputs = model(temp_inputs)
            outputs = torch.softmax(outputs/1000.0, dim=1)
            outputs = outputs.max(1)[0]
        # outputs = torch.norm(x, dim=1, keepdim=True)
        predictions.append(outputs)
    predictions = torch.cat(predictions).to(device)
    return predictions


#Norm
def calculate_norm(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            x = inputs.to(device)
            x = model.conv1(x)
            x = model.bn1(x)
            x = F.relu(x)
            
            x = model.layer1(x)                 
            x = model.layer2(x)            
            x = model.layer3(x)
            x = model.layer4[0](x)    
            # x = model.layer4[1](x)  
            # x = model.global_pool(x).view(-1, 512)
            norm = torch.norm(x, p=2, dim=[2,3], keepdim=True).mean(1)
            # print(norm.shape, x.shape)
            # x = model.bn1(x)
            # x = F.relu(x)
            # x = model.conv2(x)
            
            # norm = torch.norm(F.relu(x), p=2, dim=[2,3])

            predictions.append(norm)
    predictions = torch.cat(predictions).to(device)
    return predictions

# def calculate_norm(model, loader, device):
#     thr = torch.tensor(6.8846).to('cuda:2')
#     model.eval()
#     predictions = []
#     with torch.no_grad():
#         for batch_idx, (inputs, t) in enumerate(loader):
#             inputs = inputs.to(device)
#             x = model.to_patch_embedding(inputs)
#             b, n, _ = x.shape

#             cls_tokens = repeat(model.cls_token, '() n d -> b n d', b = b)
#             x = torch.cat((cls_tokens, x), dim=1)
#             x += model.pos_embedding[:, :(n + 1)]
#             x = model.dropout(x)

#             for i, (attn, ff) in enumerate(model.transformer.layers):
#                 x = attn(x) + x
#                 x = ff(x) + x
#                 if i == 4:
#                     norm = torch.norm(x.mean(1), dim=1, p=3, keepdim=True)
#                     for i in range(len(norm)):
#                         if norm[i]>thr:
#                             x = x/norm.view(-1, 1, 1) * thr
#             x = x.mean(dim = 1) if model.pool == 'mean' else x[:, 0]
#             x = model.to_latent(x)
#             norm = torch.norm(x, p=1, dim=1,keepdim=True)

#             predictions.append(norm)
#     predictions = torch.cat(predictions).to(device)
#     return predictions    

def attention(x):
    attention = x.mean(3).mean(2)
    attention = attention.view(attention.size(0), -1)
    attention = torch.softmax(attention, dim=1)
    x = x * attention.view(x.size(0), x.size(1), 1, 1)
    
    attention = x.mean(1)
    attention = attention.view(attention.size(0), x.size(2)**2)
    attention = torch.softmax(attention, dim=1)
    x = x* attention.view(x.size(0), 1, x.size(2), x.size(3))
    return x
#Norm
def calculate_allnm(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            x = inputs.to(device)
            norm = torch.zeros([inputs.size(0), 1]).to(device)
            
            # ResNet
            # x = model.conv1(x)
            # x = model.bn1(x)
            # x = model.act1(x)
            # x = model.maxpool(x)

            # x = model.layer1(x)                 
            # x = model.layer2(x)            
            # x = model.layer3(x)
            # xgap = F.adaptive_avg_pool2d(x, [1,1])
            # norm = torch.norm(xgap, dim=1, keepdim=True)      
            # x = x/norm
            # # x = attention(x)
            
            # x = model.layer4(x) # Batch, 512, 4, 4
            # xgap = F.adaptive_avg_pool2d(x, [1,1]).view(-1, x.size(1))
            # norm = torch.norm(xgap,dim=1,keepdim=True)
            
            # DenseNet
            # out = model.conv1(x)
            # out = model.trans1(model.block1(out))
            # out = model.trans2(model.block2(out))
            # # print(out.shape)
            # out_gap = F.adaptive_avg_pool2d(out, [1,1])
            # norm = torch.norm(out_gap, dim=1 ,keepdim=True)
            # out = out/norm
            
            # out = model.block3(out)
            # out = model.relu(model.bn1(out))
            # out = F.avg_pool2d(out, 8)
            # norm = torch.norm(out, dim=1, keepdim=True)
            
            # WRN
            out = model.conv1(x)
            out = model.block1(out)
            out = model.block2(out)
            # out_gap = F.adaptive_avg_pool2d(out, [1,1])
            # norm = torch.norm(out_gap, dim=1 ,keepdim=True)
            # out = out/norm
            
            out = model.block3.layer[0].bn1(out)
            out = model.block3.layer[0].relu1(out)
            outgap = F.adaptive_avg_pool2d(out, [1,1])
            norm = torch.norm(outgap, dim=1, keepdim=True)      
            out = out/norm

            out2 = out                 
            out = model.block3.layer[0].conv1(out)
            out = model.block3.layer[0].bn2(out)
            out = model.block3.layer[0].relu2(out)
            out = model.block3.layer[0].conv2(out)
            out = torch.add(model.block3.layer[0].convShortcut(out2), out)

            out = model.block3.layer[1](out)    
            out = model.block3.layer[2](out)
            out = model.block3.layer[3](out)
            out = model.block3.layer[4](out)
            out = model.block3.layer[5](out)
            out = model.relu(model.bn1(out))
            out = F.avg_pool2d(out, 8)
            out = out.view(-1, model.nChannels)
            norm = torch.norm(out, dim=1, keepdim=True)             
            
            predictions.append(norm)
    predictions = torch.cat(predictions).to(device)
    return predictions



#Cosine
def calculate_cosine(model, loader, device):
    model.eval()
    predictions = []

    w_norm = torch.norm(model.fc.state_dict()['weight'], dim=1, keepdim=True)    
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            x = inputs.to(device)
            x = model.conv1(x)

            # x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.global_pool(x)
            x = model.bn1(x)
            x = F.relu(x)
            features = torch.flatten(x, 1)            
            f_norm = torch.norm(features, dim=1, keepdim=True)
            outputs = model.fc(features/torch.clip(f_norm, min=1e-06))
            outputs = outputs/w_norm.view(1,-1)
            outputs = outputs/torch.clip(f_norm, min=1e-09)
            #print(outputs.shape, f_norm.shape)
            outputs = outputs.max(dim=1).values.view(-1,1)# * f_norm
            # print(outputs.max(dim=1).values.shape, outputs.shape)
            predictions.append(outputs.view(-1, 1))
    predictions = torch.cat(predictions).to(device)
    return predictions
# def calculate_cosine(model, loader, device):
#     model.eval()
#     predictions = []

#     w_norm = torch.norm(model.fc.state_dict()['weight'], dim=1, keepdim=True)    
#     with torch.no_grad():
#         for batch_idx, (inputs, t) in enumerate(loader):
#             inputs = inputs.to(device)
#             features = model.forward_features(inputs)
#             if type(model).__name__ == 'ResNet':
#                 features = model.global_pool(features)
#             f_norm = torch.norm(features, dim=1, keepdim=True)
#             outputs = model.fc(features)
#             outputs = outputs/w_norm.view(1,-1)
#             outputs = outputs/torch.clip(f_norm, min=1e-09)
#             outputs = outputs.max(dim=1).values
#             predictions.append(outputs.view(-1, 1))
#     predictions = torch.cat(predictions).to(device)
#     return predictions

def calculate_thnm(model, loader, threshold, device):
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            x = inputs.to(device)
            norm = torch.zeros([inputs.size(0), 1]).to(device)
            
            # ResNet
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.act1(x)
            x = model.maxpool(x)

            x = model.layer1(x)                 
            x = model.layer2(x)            
            x = model.layer3(x)
            xgap = F.adaptive_avg_pool2d(x, [1,1])
            norm = torch.norm(xgap, dim=1, keepdim=True)     
            if norm > threshold: 
                x = x/norm * threshold
                            
            x = model.layer4(x) # Batch, 512, 4, 4
            xgap = F.adaptive_avg_pool2d(x, [1,1]).view(-1, x.size(1))
            norm = torch.norm(xgap,dim=1,keepdim=True)
            predictions.append(norm)
    predictions = torch.cat(predictions).to(device)
    return predictions
"""
Energy score
source code from 'https://github.com/deeplearning-wisc/gradnorm_ood/blob/master/test_ood.py'
"""
def calculate_energy(model, loader, device):
    model.eval()
    predictions = []
  
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)

            energy = torch.logsumexp(outputs.data, dim=1)
            predictions.append(energy)
    predictions = torch.cat(predictions).to(device)
    return predictions    

"""
GradNorm score
source code from 'https://github.com/deeplearning-wisc/gradnorm_ood/blob/master/test_ood.py'
"""
def calculate_gradnorm(model, loader, device):
    model.eval()
    predictions = []

    logsoftmax = torch.nn.LogSoftmax(dim=-1).to(device)
    for batch_idx, (inputs, t) in enumerate(loader):
        # for image in inputs:
        #     image = image.unsqueeze(0).to(device)
            image = inputs.to(device)

            model.zero_grad()
            outputs = model(image)
            targets = torch.ones((image.shape[0], outputs.size(1))).to(device)
            
            loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))
            loss.backward()

            layer_grad = model.fc.weight.grad.data
            layer_grad_norm = torch.sum(torch.abs(layer_grad), dim=0).view(-1,1)
            # print(layer_grad.shape, layer_grad_norm.shape)

            predictions.extend(layer_grad_norm)

    predictions = torch.cat(predictions).to(device)
    return predictions    


"""
ReAct + Energy
source code from "https://github.com/deeplearning-wisc/react"
"""
def calculate_react(model, loader, thr, device):
    threshold = thr#1.0#1.5 #2.4
    model.eval()
    predictions = []        
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            inputs = inputs.to(device)
            
            x = model.conv1(inputs)
            x = model.bn1(x)
            x = F.relu(x)
            x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)

            x = model.avgpool(x)
            features = torch.flatten(x, 1)
            # features = model.forward_features(inputs)
            # features = model.global_pool(features) if type(model).__name__ == 'ResNet' else features         
            features = torch.clip(features, max=threshold)
            outputs = model.fc(features)
            energy = torch.logsumexp(outputs, dim=1)
            predictions.append(energy)
    predictions = torch.cat(predictions).to(device)
    return predictions  


"""
Isomax: minimum distance score
source code from "https://github.com/dlmacedo/entropic-out-of-distribution-detection"
"""
def calculate_mdscore(model, loader, device):
    model.eval()
    predictions = []        
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            score, _ = outputs.max(dim=1)
            predictions.append(score)
    predictions = torch.cat(predictions).to(device)
    return predictions  


def calculate_godin(model, loader, device):
    # h of the model in generalized odin
    model.eval()
    predictions = [] 
    for batch_idx, (inputs, t) in enumerate(loader):
        inputs = inputs.to(device)
        inputs = Variable(inputs, requires_grad = True)
        outputs = model(inputs)
        max_scores, _ = torch.max(outputs, dim = 1)
        max_scores.backward(torch.ones(len(max_scores)).to(device))
        
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[::, 0] = (gradient[::, 0] )
        gradient[::, 1] = (gradient[::, 1] )
        gradient[::, 2] = (gradient[::, 2] )
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data, gradient, alpha= 0.005)          
        with torch.no_grad():
            outputs = model(tempInputs)
            scores,_ = outputs.max(dim=1)
            predictions.append(scores)
    predictions = torch.cat(predictions).to(device)
    return predictions  
### 
def image_norm(loader):
    norms = []
    for x,y in loader:
        # norm = torch.norm(x, dim=1)
        # norm = torch.norm(norm, dim=1)
        # norm = torch.norm(norm, dim=1)
        norm = x.mean(1).mean(1).mean(1)

        norms.append(norm)
    norms = torch.cat(norms)
    print(norms.mean(), norms.var())


def OOD_auroc(preds_id, model, loader, device, method):
    if method == 'msp':
        preds_ood = calculate_msp(model, loader, device).cpu()
    if method == 'norm':
        preds_ood = calculate_norm(model, loader, device).cpu()
    if method == 'cosine':
        preds_ood = calculate_cosine(model, loader, device).cpu()
    
    auroc, _, _= get_measures(preds_id, preds_ood)
    return auroc, preds_ood.mean()


def OOD_results(preds_id, model, loader, device, method, file):  
    #image_norm(loader)  
    if 'msp' in method:
        preds_ood = calculate_msp(model, loader, device).cpu()
    if 'odin' in method:
        preds_ood = calculate_odin(model, loader, device).cpu()
    if 'norm' in method:
        preds_ood = calculate_norm(model, loader, device).cpu()
    if 'allnm' in method:
        preds_ood = calculate_allnm(model, loader, device).cpu()
    if 'cosine' in method:
        preds_ood = calculate_cosine(model, loader, device).cpu()
    if 'energy' in method:
        preds_ood = calculate_energy(model, loader, device).cpu()
    if 'gradnm' in method:
        preds_ood = calculate_gradnorm(model, loader, device).cpu()
    if 'react' in method:
        preds_ood = calculate_react(model, loader, device).cpu()
    if 'md' in method:
        preds_ood = calculate_mdscore(model, loader, device).cpu()
    if 'godn' in method:
        preds_ood = calculate_godin(model, loader, device).cpu()
    if 'mls' in method:
        preds_ood = calculate_mls(model, loader, device).cpu()
    if 'dice' in method:
        preds_ood = calculate_msp(model, loader, device).cpu()
    print(torch.mean(preds_ood), torch.mean(preds_id))
    show_performance(preds_id, preds_ood, method, file=file)

