# Curriculum Fine-tuning of Vision Foundation Model for Medical Image Classification in the Presence of Noisy Labels.
In this study, we introduce Cufit, a curriculum fine-tuning paradigm of vision foundation model for medical image classification under label noise, based on linear probing and adapter modules.

---
# Getting Started
## Environment Setup
   This code is tested under LINUX 20.04 and Python 3.8 environment, and the code requires following packages to be installed:
    
   - [Pytorch](https://pytorch.org/): Tested under 2.0.1 version of Pytorch-GPU.
   - [torchvision](https://pytorch.org/vision/stable/index.html): which will be installed along Pytorch. Tested under 0.15.2 version.
   - [scipy](https://www.scipy.org/): Tested under 1.4.1 version.
   - [scikit-learn](https://scikit-learn.org/stable/): Tested under 0.22.1 version.

## Dataset Preparation
   Some public datasets are required to be downloaded for running evaluation. Required dataset can be downloaded in following links:    
   - [HAM10000](https://challenge.isic-archive.com/data/#2018)
   - [APTOS-2019](https://www.kaggle.com/c/aptos2019-blindness-detection/data)
   - CIFAR10 and 100 will be automatically downloaded.

### Config file need to be changed for your path to download. For example,
~~~
# conf/HAM10000.json
{
    "epoch" : "100",
    "id_dataset" : "./cifar10",   # Your path to HAM10000 dataset
    "batch_size" : 32,
    "save_path" : "./cifar10/",   # Your path to checkpoint
    "num_classes" : 7
}
~~~

---
## How to Run
### To train a model by our setting (i.e., ours) with DINOv2-small for HAM10000 using 40% noise rate.
~~~
python train_rein_ours_three_head.py --data 'ham10000' --noise_rate 0.4 --netsize s --gpu 'gpu-num' --save_path 'save_name'
~~~
for example,
~~~
python train_rein_ours_three_head.py --data ham10000 --noise_rate 0.4 --netsize s --gpu 0 --save_path dinov2s_rein_3module
~~~
Other baselines full-training, linear probing, and adapter only are also available. Please use train_fully.py, train_linear.py, and train_rein.py.


