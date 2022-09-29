import argparse
import torch
import torch.nn as nn  
import torch.optim as optim
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from model.network import VAE
from loss.edg_loss import edge_loss
from datasets.datasets import CloudRemovalDataset
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from utils import to_psnr, validation, print_log


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Training hyper-parameters for neural network')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--threads', type=int, default=15, help='number of threads for data loader to use')
parser.add_argument("--n_GPUs", help='list of GPUs for training neural network', default=[0], type=list)
opt = parser.parse_args()
print(opt)


# ---  hyper-parameters for training and testing the neural network --- #
train_data_dir = './data/train/'
train_batch_size = opt.batchSize
train_epoch = opt.nEpochs
data_threads = opt.threads
GPUs_list = opt.n_GPUs
category = '50dim'


device_ids = GPUs_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
print('===> Building model')
model = VAE()


# --- Define the MSE loss --- #
L1_Loss = nn.SmoothL1Loss()
L1_Loss = L1_Loss.to(device)


# --- Multi-GPU --- #
model = model.to(device)
model = nn.DataParallel(model, device_ids=device_ids)


# --- Build optimizer and scheduler --- #
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999)) 
scheduler = StepLR(optimizer,step_size= train_epoch // 2,gamma=0.1)


# --- Load training data and validation/test data --- #
train_dataset = CloudRemovalDataset(root_dir=train_data_dir, transform=transforms.Compose([transforms.ToTensor()]))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, num_workers=data_threads, shuffle=True)


# --- Training --- #
for epoch in range(1, opt.nEpochs + 1):
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
        l1_loss = L1_Loss(cloud_removal, ref)
        kl_div = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        EDGE_loss = edge_loss(cloud_removal, ref, device)
        Loss = l1_loss + 0.01*kl_div + 0.18*EDGE_loss
        epoch_loss += Loss
        Loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print("===>Epoch[{}]({}/{}): Loss: {:.5f} KL_div: {:.6f} L1_loss:{:.4f} EDGE_loss:{:.4f}".format(epoch, iteration, len(train_dataloader), Loss.item(), kl_div.item(), l1_loss.item(), EDGE_loss.item()))
            
        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(cloud_removal, ref))

    scheduler.step()
    
    train_psnr = sum(psnr_list) / len(psnr_list)

    # --- Save the network  --- #
    torch.save(model.state_dict(), './checkpoints/cloud_removal_{}.pth'.format(epoch))
    
    # --- Print log --- #
    print_log(epoch, train_epoch, train_psnr, category)
