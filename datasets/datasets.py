import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from skimage import io, transform
import os
import numpy as np
import random
import torchvision


class CloudRemovalDataset(Dataset):
    def __init__(self, root_dir, 
                       crop=False, 
                       crop_size=256, 
                       rotation=False, 
                       color_augment=False, 
                       transform=None, 
                       train = True):
        """
        Args:
             split_file: Path to the split file
             root_dir: Directory with all the images
             transform: Optional transform to be appeared on a sample
        """
        if train:
            data_list = root_dir + 'train.txt'
        else:
            data_list = root_dir + 'test.txt'

        with open(data_list) as f:
            contents = f.readlines()
            image_files = [i.strip() for i in contents]
        
        self.image_files = image_files
       
        self.root_dir = root_dir
        self.transform = transform        
        self.crop = crop
        self.crop_size = crop_size
        self.rotation = rotation
        self.color_augment = color_augment
        self.rotate90 = transforms.RandomRotation(90)  
        self.rotate45 = transforms.RandomRotation(45)    
        self.train = train

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        image_name = self.image_files[idx]
          
        cloud_image = Image.open(self.root_dir + 'cloud/' + image_name).convert('RGB')
        ref_image = Image.open(self.root_dir + 'reference/' + image_name).convert('RGB')

        if self.rotation:
            degree = random.choice([90, 180, 270])
            cloud_image = transforms.functional.rotate(cloud_image, degree) 
            ref_image = transforms.functional.rotate(ref_image, degree)

        if self.color_augment:
            cloud_image = transforms.functional.adjust_gamma(cloud_image, 1)
            ref_image = transforms.functional.adjust_gamma(ref_image, 1)                           
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            cloud_image = transforms.functional.adjust_saturation(cloud_image, sat_factor)
            ref_image = transforms.functional.adjust_saturation(ref_image, sat_factor)
            
        if self.transform:
            cloud_image = self.transform(cloud_image)
            ref_image = self.transform(ref_image)

        if self.crop:
            W = cloud_image.size()[1]
            H = cloud_image.size()[2] 

            Ws = np.random.randint(0, W-self.crop_size-1, 1)[0]
            Hs = np.random.randint(0, H-self.crop_size-1, 1)[0]
            
            cloud_image = cloud_image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
            ref_image = ref_image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
                       

        if self.train:
            return {'cloud_image': cloud_image, 'ref_image': ref_image}
        else:
            return {'cloud_image': cloud_image, 'ref_image': ref_image, 'image_name': image_name}
            