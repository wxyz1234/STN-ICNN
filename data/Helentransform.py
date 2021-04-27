import torch
from torchvision import datasets,transforms
from skimage.util import random_noise
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from config import OnServer;
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
if not OnServer:
    import matplotlib;matplotlib.use('TkAgg');
    
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
        '''
        image=sample['image_org']                
        plt.imshow(image);        
        plt.show(block=True);         
        image=sample['label_org'][5]
        plt.imshow(image);        
        plt.show(block=True);                 
        '''
        sample['image']=TF.affine(img=sample['image'],angle=degree,translate=translate_range,scale=scale_range,shear=0,fillcolor=0,resample =Image.NEAREST)
        for i in range(len(sample['label'])):
            sample['label'][i]=TF.affine(img=sample['label'][i],angle=degree,translate=translate_range,scale=scale_range,shear=0,fillcolor=0,resample =Image.NEAREST)                                                
        '''
        sample['image_org']=TF.affine(img=sample['image_org'],angle=degree,translate=translate_range,scale=scale_range,shear=0,fillcolor=0,resample =Image.NEAREST)
        for i in range(len(sample['label'])):            
            sample['label_org'][i]=TF.affine(img=sample['label_org'][i],angle=degree,translate=translate_range,scale=scale_range,shear=0,fillcolor=0,resample =Image.NEAREST)                
        '''
        if ('image_org' in sample):
            sample['image_org']=sample['image']
        if ('label_org' in sample):
            sample['label_org']=sample['label']
        '''
        image=sample['image_org']                
        plt.imshow(image);        
        plt.show(block=True);         
        image=sample['label_org'][5]        
        plt.imshow(image);        
        plt.show(block=True); 
        '''
        return sample;
    
class Resize(transforms.Resize):
    def __call__(self, sample):                
        '''
        print(F.interpolate(TF.to_tensor(sample['image']).unsqueeze(0),self.size, mode ="bilinear", align_corners=True).squeeze(0).shape);
        print(F.interpolate(TF.to_tensor(sample['label'][0]).unsqueeze(0),self.size, mode ="nearest").squeeze(0).shape);
        input("check")
        '''
        sample['image'] = TF.to_tensor(sample['image'])        
        sample['image'] = TF.to_pil_image(F.interpolate(sample['image'].unsqueeze(0),self.size, mode ="bilinear", align_corners=True).squeeze(0));        
        resized_labels=[]        
        for i in range(len(sample['label'])):
            resized_labels.append(TF.to_tensor(sample['label'][i]));
            resized_labels[i]=TF.to_pil_image(F.interpolate(resized_labels[i].unsqueeze(0),self.size, mode ="nearest").squeeze(0));
        sample['label']=resized_labels;                
        '''        
        sample['image']=TF.resize(img=TF.to_pil_image(sample['image']),size=self.size,interpolation=Image.NEAREST)           
        label_new=[];
        for i in range(len(sample['label'])):
            label_new.append(TF.to_tensor(sample['label'][i]));
            label_new[i]=TF.resize(img=TF.to_pil_image(label_new[i]),size=self.size,interpolation=Image.NEAREST)            
        sample['label']=label_new;
        '''
        return sample;
    
class ToTensor(transforms.ToTensor):
    def __call__(self, sample):   
        if ('image_org' in sample):
            sample['image_org']=TF.to_tensor(sample['image_org'])   
        if ('label_org' in sample):
            for i in range(len(sample['label_org'])):
                sample['label_org'][i]=TF.to_tensor(sample['label_org'][i])            
            sample['label_org'][0]+=0.0001        
            sample['label_org']=torch.cat(tuple(sample['label_org']),0);#[L,64,64] 
        sample['image']=TF.to_tensor(sample['image'])           
        for i in range(len(sample['label'])):
            sample['label'][i]=TF.to_tensor(sample['label'][i])         
        sample['label'][0]+=0.0001        
        sample['label']=torch.cat(tuple(sample['label']),0);#[L,64,64]        
        return sample;  

  
class ToTensor_label(transforms.ToTensor):
    def __call__(self, sample):   
        '''
        print(sample['image_org'].size);
        print(sample['image'].size);
        print(sample['label_org'][0].size);
        print(sample['label'][0].size);
        '''
        if ('image_org' in sample):
            sample['image_org']=TF.to_tensor(sample['image_org'])   
        if ('label_org' in sample):
            for i in range(len(sample['label_org'])):
                sample['label_org'][i]=TF.to_tensor(sample['label_org'][i])            
            sample['label_org'][0]+=0.0001        
            sample['label_org']=torch.cat(tuple(sample['label_org']),0);#[L,64,64] 
        return sample;      
    
class HorizontalFlip():
    def __call__(self, sample):        
        sample['image']=TF.hflip(sample['image'])
        for i in range(len(sample['label'])):
            sample['label'][i]=TF.hflip(sample['label'][i])
        return sample;
class Padding_mid():
    def __init__(self,size):
        self.size=size;        
    def __call__(self, sample):        
        orig = TF.to_pil_image(sample['image_org'])
        orig_label = [TF.to_pil_image(sample['label_org'][r])for r in range(len(sample['label_org']))]        
        desired_size = 1024
        delta_width = desired_size - orig.size[0]
        delta_height = desired_size - orig.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        orig_size = np.array([orig.size[0], orig.size[1]])
        padding = np.array([pad_width, pad_height, delta_width - pad_width, delta_height - pad_height])        
        pad_orig = TF.to_tensor(TF.pad(orig, tuple(padding)))
        orig_label = [TF.to_tensor(TF.pad(orig_label[r], tuple(padding)))for r in range(len(orig_label))]        
        orig_label = torch.cat(orig_label, dim=0).float()        
        orig_label[0] = torch.tensor(1.) - torch.sum(orig_label[1:], dim=0, keepdim=True)                                
        assert pad_orig.shape == (3, 1024, 1024)
        assert orig_label.shape == (9, 1024, 1024)
        sample['image_org']=pad_orig;
        sample['label_org']=orig_label;        
        return sample
    
class Padding_leftup():
    def __init__(self,size):
        self.size=size;        
    def __call__(self, sample):        
        desired_size = 1024
        delta_width = desired_size - sample['size'][0]
        delta_height = desired_size - sample['size'][1]        
        padding = np.array([0, 0, delta_width, delta_height])       
        if ('image_org' in sample):
           orig = TF.to_pil_image(sample['image_org'])  
           pad_orig = TF.to_tensor(TF.pad(orig, tuple(padding)))
           assert pad_orig.shape == (3, 1024, 1024)
           sample['image_org']=pad_orig;
        if ('label_org' in sample):
            orig_label = [TF.to_pil_image(sample['label_org'][r])for r in range(len(sample['label_org']))] 
            orig_label = [TF.to_tensor(TF.pad(orig_label[r], tuple(padding)))for r in range(len(orig_label))]        
            orig_label = torch.cat(orig_label, dim=0).float()        
            orig_label[0] = torch.tensor(1.) - torch.sum(orig_label[1:], dim=0, keepdim=True)         
            sample['label_org']=orig_label;        
            assert orig_label.shape == (9, 1024, 1024)        
        return sample    
    
class Padding_oldchange():
    def __init__(self,size):
        self.size=size;        
    def __call__(self, sample):
        '''
        tmp=Image.new('RGB', self.size, (0,0,0));        
        tmp.paste(sample['image'],(0,0));        
        sample['image']=tmp;        
        '''
        orig = TF.to_pil_image(sample['image_org'])    
        sample['image_org']=orig;
        orig_label = [TF.to_pil_image(sample['label_org'][r])for r in range(len(sample['label_org']))]       
        sample['label_org']=orig_label;        
        tmp=Image.new('RGB', self.size, (0,0,0));        
        tmp.paste(sample['image_org'],(0,0));     
        pad_orig = TF.to_tensor(tmp);            
        for i in range(len(sample['label_org'])):
            tmp=Image.new('L', self.size, 0);                
            tmp.paste(sample['label_org'][i],(0,0));     
            sample['label_org'][i]=tmp;
        orig_label = [TF.to_tensor(sample['label_org'][r])for r in range(len(orig_label))]        
        orig_label = torch.cat(orig_label, dim=0).float()        
        orig_label[0] = torch.tensor(1.) - torch.sum(orig_label[1:], dim=0, keepdim=True)         
        sample['image_org']=pad_orig;   
        sample['label_org']=orig_label;   
        #sample['image']=Image.new('RGB', self.size, (0,0,0)).paste(sample['image'],(0,0));                
        return sample;

class Padding_old():
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
            for i in range(len(sample['label_org'])):            
                #sample['label'][i]=Image.new('L', self.size, 0).paste(sample['label'][i],(0,0));
                tmp=Image.new('L', self.size, 0);        
                tmp.paste(sample['label_org'][i],(0,0));        
                sample['label_org'][i]=tmp;
        return sample;