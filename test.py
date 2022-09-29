from __future__ import print_function  
import argparse
import torch
import torch.nn as nn  
import torch.optim as optim
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from model.network import Net
from model.VAE_50dim import VAE
from model.MSAR_DefogNet import DefogNet
from model.U_Net import UNet
from model.MCRN import MCRN
from model.DehazingAndFusion import DehazingAndFusion
from model.RSC_Net import RSC_Net
from model.SPANet import Generator
from datasets.datasets import CloudRemovalDataset, RICEDataset, WHUDataset
from os.path import exists, join, basename
import time
from torchvision import transforms
from utils import to_psnr, validation
import os
import time

# ---  hyper-parameters for testing the neural network --- #
test_data_dir = './data/test/real/'
test_batch_size = 1
data_threads = 15


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# --- Validation data loader --- #
test_dataset = CloudRemovalDataset(root_dir = test_data_dir, transform = transforms.Compose([transforms.ToTensor()]), train=False)

test_dataloader = DataLoader(test_dataset, batch_size = test_batch_size, num_workers=data_threads, shuffle=False)

# --- Define the network --- #
model = VAE()

# --- Multi-GPU --- #
model = model.to(device)
model = nn.DataParallel(model, device_ids=[1])


# --- Load the network weight --- #
model.load_state_dict(torch.load('./checkpoints/VAE/50dim/cloud_removal.pth'))

# if isinstance(model,torch.nn.DataParallel):
#     model = model.module
   
# --- Use the evaluation model in testing --- #
model.eval()
print('--- Testing starts! ---')
print(len(test_dataloader))
test_psnr, test_ssim = validation(model, test_dataloader, device, save_tag=False)
print('test_psnr: {0:.6f}, test_ssim: {1:.6f}'.format(test_psnr, test_ssim))

