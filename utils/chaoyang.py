import torch.utils.data as data
import torch
import os
import json

from PIL import Image

class CHAOYANG():
    def __init__(self, root_dir, train = True, transforms=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transforms
        
        self.label_txt = os.path.join(root_dir, 'train.json' if train else 'test.json')

        self.samples = []
        with open(self.label_txt, 'r') as f:
            load_list = json.load(f)

            for i in range(len(load_list)):
                img_path = os.path.join(root_dir, load_list[i]["name"])
                label = (load_list[i]["label"])
                
                self.samples.append([img_path, label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, label = self.samples[idx]
        sample = Image.open(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample, label
    
if __name__ == '__main__':
    aptos = CHAOYANG('./data/chaoyang-data', True)
    print(len(aptos))