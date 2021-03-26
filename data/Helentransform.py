import torch
from torchvision import datasets,transforms
from skimage.util import random_noise
from torchvision.transforms import functional as TF
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class DoNothing:
    def __call__(self, sample):
        return sample;
    
class ToPILImage:
    def __call__(self, sample):     
        sample['image']=TF.to_pil_image(sample['image'])
        sample['label'] = [TF.to_pil_image(sample['label'][i])
              for i in range(sample['label'].shape[0])]
        return sample
class GaussianNoise:
    def __call__(self, sample):  
        sample['image'] = np.array(sample['image'], np.uint8)
        sample['image']=random_noise(sample['image'])
        sample['image'] = TF.to_pil_image(np.uint8(255 * sample['image']))  
        return sample
    
class RandomAffine(transforms.RandomAffine):
    def __call__(self, sample):                   
        degree=random.uniform(self.degrees[0],self.degrees[1])          
        maxx=self.translate[0]*sample['image'].size[0];
        maxy=self.translate[1]*sample['image'].size[1];
        translate_range = (random.uniform(-maxx,maxx), random.uniform(-maxy,maxy));        
        scale_range = random.uniform(self.scale[0],self.scale[1]);          
        sample['image']=TF.affine(img=sample['image'],angle=degree,translate=translate_range,scale=scale_range,shear=0,fillcolor=0,resample=Image.NEAREST)
        for i in range(len(sample['label'])):
            sample['label'][i]=TF.affine(img=sample['label'][i],angle=degree,translate=translate_range,scale=scale_range,shear=0,fillcolor=0,resample=Image.NEAREST)                        
        return sample;
    
class Resize(transforms.Resize):
    def __call__(self, sample):
        sample['image']=TF.resize(img=sample['image'],size=self.size,interpolation=Image.NEAREST)
        for i in range(len(sample['label'])):
            sample['label'][i]=TF.resize(img=sample['label'][i],size=self.size,interpolation=Image.NEAREST)
        return sample;
    
class Resize_image(transforms.Resize):
    def __call__(self, sample):             
        sample['image']=TF.resize(img=sample['image'],size=self.size,interpolation=Image.NEAREST)        
        return sample;    
    
class Resize_label(transforms.Resize):
    def __call__(self, sample):                     
        for i in range(len(sample['label'])):
            sample['label'][i]=TF.resize(img=sample['label'][i],size=self.size,interpolation=Image.NEAREST)
        return sample;        
    
class ToTensor(transforms.ToTensor):
    def __call__(self, sample):   
        if ('image_org' in sample):
            sample['image_org']=TF.to_tensor(sample['image_org'])   
        if ('label_org' in sample):
            for i in range(len(sample['label_org'])):
                sample['label_org'][i]=TF.to_tensor(sample['label_org'][i])                    
        sample['image']=TF.to_tensor(sample['image'])           
        for i in range(len(sample['label'])):
            sample['label'][i]=TF.to_tensor(sample['label'][i])         
        sample['label'][0]+=0.0001        
        sample['label']=torch.cat(tuple(sample['label']),0);#[L,64,64]        
        return sample;        
    
class HorizontalFlip():
    def __call__(self, sample):        
        sample['image']=TF.hflip(sample['image'])
        for i in range(len(sample['label'])):
            sample['label'][i]=TF.hflip(sample['label'][i])
        return sample;
    
class Padding():
    def __init__(self,size,pdlabel):
        self.size=size;
        self.pdlabel=pdlabel;
    def __call__(self, sample ):                
        '''
        tmp=Image.new('RGB', self.size, (0,0,0));        
        tmp.paste(sample['image'],(0,0));        
        sample['image']=tmp;        
        '''
        if ('image_org' in sample):
            tmp=Image.new('RGB', self.size, (0,0,0));        
            tmp.paste(sample['image_org'],(0,0));     
            sample['image_org']=tmp;
        if ('label_org' in sample):
            for i in range(len(sample['label_org'])):
                tmp=Image.new('L', self.size, 0);                
                tmp.paste(sample['label_org'][i],(0,0));     
                sample['label_org'][i]=tmp;
        #sample['image']=Image.new('RGB', self.size, (0,0,0)).paste(sample['image'],(0,0));        
        if self.pdlabel:
            for i in range(len(sample['label'])):            
                #sample['label'][i]=Image.new('L', self.size, 0).paste(sample['label'][i],(0,0));
                tmp=Image.new('L', self.size, 0);        
                tmp.paste(sample['label'][i],(0,0));        
                sample['label'][i]=tmp;
        return sample;