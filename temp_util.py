import torch
import torch.utils.data as data
import os
import random

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from scipy import io
from torchvision import transforms
from torchvision import datasets as dset
import torchvision

random.seed(0)

normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


train_transform =  transforms.Compose([transforms.Resize([112,112]), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
test_transform = transforms.Compose([transforms.Resize([112,112]), transforms.ToTensor(), normalize])

def get_loader(folder, batch_size=32, noise_rate = 0.0, shuffle = True, filter_path=None):
    # print(noise_rate)

    train_data = dset.ImageFolder(os.path.join(folder, 'train'), transform = train_transform)
    if noise_rate > 0.0:
        new_samples = []
        for data in train_data.samples:
            if random.random()<0.2:
                label = 0 if data[1] == 1 else 1
            else:
                label = data[1]
            new_samples.append([data[0], label])
            # print(data[1], label)
        train_data.samples = new_samples
    if not filter_path is None:
        new_samples = []
        with open(filter_path, 'r') as f:
            lines = f.readlines()
        for i, data in enumerate(train_data.samples):
            label = lines[i].split('\n')
            new_samples.append([data[0], int(label[0])])
        train_data.samples = new_samples
        

    test_data = dset.ImageFolder(os.path.join(folder, 'test'), transform = test_transform)


    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=shuffle, pin_memory=True, num_workers = 8)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

def validation_accuracy(model, loader, device):
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            #print(outputs.shape)
            total += targets.size(0)
            _, predicted = outputs.max(1)  
            correct += predicted.eq(targets).sum().item()
    valid_accuracy = correct/total
    return valid_accuracy

