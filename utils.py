import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage import metrics
from torch.autograd import Variable
import os, time

def to_psnr(cloud, gt):
    mse = F.mse_loss(cloud, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(cloud, gt):
    cloud_list = torch.split(cloud, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    cloud_list_np = [cloud_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(cloud_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(cloud_list))]
    ssim_list = [metrics.structural_similarity(cloud_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(cloud_list))]

    return ssim_list


def validation(net, val_data_loader, device, save_tag=False):
    """
    :param net: Network
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param save_tag: tag of saving image or not
    :return: average PSNR and SSIM value
    """
    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):
        with torch.no_grad():
            cloud, gt, image_name = Variable(val_data['cloud_image']), Variable(val_data['ref_image']),val_data['image_name']
            cloud = cloud.to(device)
            gt = gt.to(device)
            cloud_removal ,_ ,_ = net(cloud)


            # --- Calculate the average PSNR --- #
            psnr = to_psnr(cloud_removal, gt)

            # --- Calculate the average SSIM --- #
            ssim = to_ssim_skimage(cloud_removal, gt)
                   
        psnr_list.extend(psnr)
        ssim_list.extend(ssim)

        # --- Save image --- #
        if save_tag:
            path = './results/'
            if not os.path.exists(path):
                os.makedirs(path)
            save_image(cloud_removal, image_name, path)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)

    return avr_psnr, avr_ssim


def save_image(cloud_removal, image_name, path):
    cloud_removal = torch.split(cloud_removal, 1, dim=0)
    batch_num = len(cloud_removal)
    for ind in range(batch_num):
        utils.save_image(cloud_removal[ind], path+'{}'.format(image_name[ind][:-3] + 'png'))


def print_log(epoch, num_epochs, train_psnr, train_ssim):
    print('Epoch [{0}/{1}], Train_PSNR:{2:.2f}, Train_SSIM:{3:.4f}'
          .format(epoch, num_epochs, train_psnr, train_ssim))

    # --- Write the training log --- #
    with open('./logs/train_log.txt', 'a') as f:
        print('Date: {0}s, Epoch: [{1}/{2}], Train_PSNR: {3:.2f}, Train_SSIM: {4:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      epoch, num_epochs, train_psnr, train_ssim), file=f)
