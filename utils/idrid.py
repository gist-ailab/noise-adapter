import torch.utils.data as data
import torch
import os

from PIL import Image

class IDRID():
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
        
        self.label_txt = os.path.join(root_dir, '2. Groundtruths', 'a. IDRiD_Disease Grading_Training Labels.csv' if train else 'b. IDRiD_Disease Grading_Testing Labels.csv')
            
        self.samples = []
        with open(self.label_txt, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.split(',')
                img_name, label = line[0], line[1]
                
                img_name = os.path.join(root_dir, '1. Original Images', 'a. Training Set' if train else 'b. Testing Set', img_name+'.jpg')
                # label = label.replace('\n', '')
                label = int(label)

                self.samples.append([img_name, label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, label = self.samples[idx]
        sample = Image.open(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample, label
    
if __name__ == '__main__':
    aptos = IDRID('./data/IDRID', False)
    print(len(aptos))