import json
import torch.utils.data as data
import numpy as np
import torch
import torch.nn.functional as F
import os
import random

from PIL import Image
from scipy import io
from torchvision import transforms
from torchvision import datasets as dset
import torchvision

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

imagenet_normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

train_transform =  transforms.Compose([transforms.Resize([224,224]), transforms.RandomHorizontalFlip(), transforms.ToTensor(), imagenet_normalize])
test_transform = transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor(), imagenet_normalize])


class masking_dataset(torch.utils.data.Dataset): 
    def __init__(self, dataset, transform, ratio):
        self.dataset = dataset
        self.transform = transform
        self.ratio = ratio

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]

        x_ = x.clone()
        _, H, W = x.shape

        mshape = 1, round(H / 2), round(W / 2)
        input_mask = torch.rand(mshape, device=x_.device)
        input_mask = (input_mask > self.ratio).float()
        input_mask = F.interpolate(input_mask.unsqueeze(0), scale_factor=2, mode='nearest')
        masked_x = x_ * input_mask.squeeze(0)
        return x, masked_x, y

def read_conf(json_path):
    """
    read json and return the configure as dictionary.
    """
    with open(json_path) as json_file:
        config = json.load(json_file)
    return config
      
def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class

class cifar10Nosiy(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, nosiy_rate=0.0, asym=False):
        np.random.seed(0)
        # print(np.random.randn(5))
        super(cifar10Nosiy, self).__init__(root, transform=transform, target_transform=target_transform, download=True)
        if asym:
            # automobile < - truck, bird -> airplane, cat <-> dog, deer -> horse
            source_class = [9, 2, 3, 5, 4]
            target_class = [1, 0, 5, 3, 7]
            for s, t in zip(source_class, target_class):
                cls_idx = np.where(np.array(self.targets) == s)[0]
                n_noisy = int(nosiy_rate * cls_idx.shape[0])
                noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
                for idx in noisy_sample_index:
                    self.targets[idx] = t
            return
        elif nosiy_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(nosiy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
            class_noisy = int(n_noisy / 10)
            noisy_idx = []
            for d in range(10):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                # print(noisy_class_index[:10])
                noisy_idx.extend(noisy_class_index)
                # print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(n_classes=10, current_class=self.targets[i])
            print(len(noisy_idx))
            # print("Print noisy label generation statistics:")
            for i in range(10):
                n_noisy = np.sum(np.array(self.targets) == i)
                # print("Noisy class %s, has %s samples." % (i, n_noisy))
            return

class cifar100Nosiy(torchvision.datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, nosiy_rate=0.0, asym=False, seed=0):
        super(cifar100Nosiy, self).__init__(root, download=download, transform=transform, target_transform=target_transform)
        self.download = download
        if asym:
            """mistakes are inside the same superclass of 10 classes, e.g. 'fish'
            """
            nb_classes = 100
            P = np.eye(nb_classes)
            n = nosiy_rate
            nb_superclasses = 20
            nb_subclasses = 5

            if n > 0.0:
                for i in np.arange(nb_superclasses):
                    init, end = i * nb_subclasses, (i+1) * nb_subclasses
                    P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

                    y_train_noisy = multiclass_noisify(np.array(self.targets), P=P, random_state=seed)
                    actual_noise = (y_train_noisy != np.array(self.targets)).mean()
                assert actual_noise > 0.0
                print('Actual noise %.2f' % actual_noise)
                self.targets = y_train_noisy.tolist()
            return
        elif nosiy_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(nosiy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(100)]
            class_noisy = int(n_noisy / 100)
            noisy_idx = []
            for d in range(100):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                # print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(n_classes=100, current_class=self.targets[i])
            # print(len(noisy_idx))
            # print("Print noisy label generation statistics:")
            for i in range(100):
                n_noisy = np.sum(np.array(self.targets) == i)
                # print("Noisy class %s, has %s samples." % (i, n_noisy))
            return

def get_ham10000(path, batch_size, noise_rate = 0.0):
    train_data = torchvision.datasets.ImageFolder(path + 'train', train_transform)
    valid_data = torchvision.datasets.ImageFolder(path + 'test', train_transform)

    if noise_rate > 0.0:
        np.random.seed(0)
        new_samples = []
        for sample in train_data.samples:
            if np.random.rand() < noise_rate:
                noisy_label = np.random.choice(np.delete(np.arange(0, 7), sample[1]))
                new_samples.append([sample[0], noisy_label])
            else:
                new_samples.append(sample)
    train_data.samples = new_samples

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, valid_loader 

if __name__ == '__main__':
    # get_food101n('/SSDe/yyg/data/Food-101N_release', 2)
    # get_clothing1m('/SSDe/yyg/data/Clothing1M', 2)
    get_animal10n('/SSDb/yyg/data/animal10n', 2)

