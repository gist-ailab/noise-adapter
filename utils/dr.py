"""DR kaggle Dataset"""
import torch.utils.data as data
import torch
import os

from PIL import Image

class DR():
    def __init__(self, root_dir, train = True, transforms=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transforms
        self.samples = []        
        self.label_txt = os.path.join(root_dir, 'kaggle_original.csv' if train else 'golden.txt')
        self.img_folder = 'train' if train else 'test'
        
        self.img_path = os.listdir(os.path.join(self.root_dir, self.img_folder))

        self.label2int = {
            'normal': 0,
            'NPDRI': 1,
            'NPDRII': 2,
            'NPDRIII': 3,
            'PDR': 4

        }

        self.temp = {0: 0, 1:0, 2:0, 3:0, 4:0}

        with open(self.label_txt, 'r') as f:
            lines = f.readlines()

            lines = lines[1:] if train else lines

            for line in lines:
                img_name, label = line.split(' ')
                label = self.label2int[label[:-1]] # [:-1] to remove \n
                # print(img, label[:-1])
                # print(img, self.label2int[label[:-1]])
                self.temp[label]+=1
        print(self.temp)
                
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, label = self.samples[idx]
        sample = Image.open(sample)
        sample = sample.convert("RGB")

        if self.transform:
            sample = self.transform(sample)

        return sample, label


if __name__ == '__main__':
    # train_set = DR('data/diabetic-retinopathy-detection')
    test_set = DR('data/diabetic-retinopathy-detection', train=False)

