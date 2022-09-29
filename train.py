"""
paper: Pyramid Channel-based Feature Attention Network for image dehazing 
file: network.py
about: model for PCFAN
author: Tao Wang
date: 01/13/21
"""
# --- Imports --- #
from __future__ import print_function  
import argparse
import torch
import torch.nn as nn  
import torch.optim as optim
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from model.network import Net
from model.VAE_50dim import VAE
from loss.edg_loss import edge_loss
from datasets.datasets import CloudRemovalDataset, RICEDataset
from os.path import exists, join, basename
import time
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from utils import to_psnr, validation, print_log
import os
from PIL import Image
import cv2


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Training hyper-parameters for neural network')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=15, help='number of threads for data loader to use')
parser.add_argument('--net', default='', help="path to net_Dehazing (to continue training)")
parser.add_argument('--continueEpochs', type=int, default=0, help='continue epochs')
parser.add_argument("--n_GPUs", help='list of GPUs for training neural network', default=[0], type=list)
opt = parser.parse_args()
print(opt)



# ---  hyper-parameters for training and testing the neural network --- #
train_data_dir = './data/RICE1/'
val_data_dir = './data/RICE1/'
train_batch_size = opt.batchSize
val_batch_size = opt.testBatchSize
train_epoch = opt.nEpochs
data_threads = opt.threads
GPUs_list = opt.n_GPUs
continueEpochs = opt.continueEpochs
category = '50dim-l1loss'


device_ids = GPUs_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
print('===> Building model')
model = VAE()
# w_model = Net()


# --- Define the MSE loss --- #
MSELoss = nn.MSELoss()
MSELoss = MSELoss.to(device)
L1_Loss = nn.SmoothL1Loss()
L1_Loss = L1_Loss.to(device)


# --- Multi-GPU --- #
model = model.to(device)
# w_model = w_model.to(device)
model = nn.DataParallel(model, device_ids=device_ids)
# w_model = nn.DataParallel(w_model, device_ids=device_ids)

# --- Load the network weight --- #
# w_model.load_state_dict(torch.load('./checkpoints/PCFAN/cloud_removal.pth'))


# --- Build optimizer and scheduler --- #
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999)) 
scheduler = StepLR(optimizer,step_size= train_epoch // 2,gamma=0.1)


# --- Load training data and validation/test data --- #
train_dataset = RICEDataset(root_dir=train_data_dir, transform=transforms.Compose([transforms.ToTensor()]))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, num_workers=data_threads, shuffle=True)

val_dataset = RICEDataset(root_dir=val_data_dir, transform=transforms.Compose([transforms.ToTensor()]), train=False)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=data_threads, shuffle=False)


old_val_psnr, old_val_ssim = validation(model, val_dataloader, device, save_tag=False)
print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))

for epoch in range(1 + opt.continueEpochs, opt.nEpochs + 1 + opt.continueEpochs):
    print("Training...")
    epoch_loss = 0
    psnr_list = []
    for iteration, inputs in enumerate(train_dataloader,1):

        cloud, ref = Variable(inputs['cloud_image']), Variable(inputs['ref_image'])
        cloud = cloud.to(device)
        ref = ref.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        model.train()
        cloud_removal, mean, log_var = model(cloud)
        MSE_loss = MSELoss(cloud_removal, ref)
        l1_loss = L1_Loss(cloud_removal, ref)
        kl_div = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        EDGE_loss = edge_loss(cloud_removal, ref, device)
        
        # w_model.eval()
        # weighted_l1loss = Weighted_SmoothL1Loss(w_model, cloud_removal, cloud, ref, device, kernel_size=16, stride=16)
        Loss = l1_loss + 0.01*kl_div + 0.18*EDGE_loss
        epoch_loss += Loss
        Loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print("===>Epoch[{}]({}/{}): Loss: {:.5f} MSELoss: {:.5f} KL_div: {:.6f} L1_loss:{:.4f} EDGE_loss:{:.4f}".format(epoch, iteration, len(train_dataloader), Loss.item(), MSE_loss.item(), kl_div.item(), l1_loss.item(),                   EDGE_loss.item()))
            
        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(cloud_removal, ref))

    scheduler.step()
    
    train_psnr = sum(psnr_list) / len(psnr_list)
    save_checkpoints = './checkpoints/VAE/50dim-l1loss/RICE/'
    if os.path.isdir(save_checkpoints)== False:
        os.mkdir(save_checkpoints)

    # --- Save the network  --- #
    torch.save(model.state_dict(), './checkpoints/VAE/50dim-l1loss/RICE/cloud_removal.pth')

    # --- Use the evaluation model in testing --- #
    model.eval()

    val_psnr, val_ssim = validation(model, val_dataloader, device, save_tag=False)
    
    # --- update the network weight --- #
    if val_psnr >= old_val_psnr:
        torch.save(model.state_dict(), './checkpoints/VAE/50dim-l1loss/RICE/cloud_removal_best.pth')
        old_val_psnr = val_psnr
        
    print_log(epoch, train_epoch, train_psnr, train_psnr, val_psnr, val_ssim, category)
