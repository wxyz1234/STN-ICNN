import torch
from torchvision import datasets,transforms
from torch.utils import data
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image
import numpy as np
from torchvision.transforms import functional as TF
from config import train_txt,val_txt,test_txt,image_path,label_path,batch_size,OnServer
from data.Helentransform import GaussianNoise,RandomAffine,Resize,ToTensor,DoNothing,Resize_image,Resize_label,ToPILImage,HorizontalFlip,Padding
import time,datetime
'''
part1_time=0;    
part2_time=0;    
part3_time=0;         
prev_time=time.time();   
'''
class Augmentation:
    def __init__(self,mode):
        degree = 15
        translate_range = (0.1,0.1);
        scale_range = (0.9, 1.2);           
        self.name="Augmentation"             
        self.augmentation=[];
        self.augmentation.append([DoNothing()]);
        if (mode=="train"):
            self.augmentation.append([GaussianNoise(),
                                        RandomAffine(degrees=degree, translate=translate_range,scale=scale_range),
                                        transforms.Compose([GaussianNoise(),RandomAffine(degrees=degree, translate=translate_range,scale=scale_range)])]);
                    
            self.augmentation.append([RandomAffine(degrees=degree, translate=(0,0), scale=(1,1)),
                                        RandomAffine(degrees=0, translate=translate_range, scale=(1,1)),
                                        RandomAffine(degrees=0, translate=(0,0), scale=scale_range)]);
      
            self.augmentation.append([RandomAffine(degrees=0, translate=translate_range, scale=scale_range),
                                        RandomAffine(degrees=degree, translate=(0,0), scale=scale_range),
                                        RandomAffine(degrees=degree, translate=translate_range, scale=(1,1))]);
                    
            self.augmentation.append([RandomAffine(degrees=degree, translate=translate_range,scale=scale_range)]);  
        self.num=len(self.augmentation);
        self.transforms=[];        
        for i in range(self.num):
            self.transforms.append(transforms.Compose([                                
                transforms.RandomChoice(self.augmentation[i]),  
                Padding(size=(1024, 1024)),                                                           
            ]));

        
class data_loader_Aug:
    def __init__(self,mode,batch_size,stage):
        self._path_list=[]
        self._num=0;
        self.choice=Augmentation(mode);
        if (mode=='train'):file_path=train_txt
        if (mode=='test'):file_path=test_txt
        if (mode=='val'):file_path=val_txt        
        with open(file_path) as f:                        
            lines=f.readlines()
            for line in lines:
                if (line.strip()==""):continue                                                                
                path=line.split(',')[1].strip()
                self._path_list.append(path);
            self._num=len(self._path_list);
        self._data_loader=get_loader_Aug(mode,batch_size,self.choice,stage);        
    def get_loader(self):
        return self._data_loader;
    def get_namelist(self):
        return self._path_list;
    def get_len(self):
        return self._num;     

class Helen_Aug(data.Dataset):
    def __init__(self,image_trans,mode='train',stage="stage1"):
        self.image_trans=image_trans          
        self.path_list=[]        
        self.num=0;
        self.mode=mode
        self.stage=stage;
        self.preprocess_data(mode);           
        self.trans2=transforms.Compose([Resize(size=(128, 128),interpolation=Image.NEAREST),ToTensor()]);              
    def __len__(self):
        return self.num;
    def __getitem__(self,index):   
        '''
        global part1_time,part2_time,part3_time,prev_time;
        now_time=time.time();
        part3_time+=now_time-prev_time;        
        prev_time=now_time;     
        '''
        path=self.path_list[index]
        path2=image_path+'/'+path+".jpg"        
        image=Image.open(path2)
        label=[];   
        label.append(Image.new('L', (image.size[0],image.size[1]),color=1))
        path2=label_path+'/'+path+'/'
        for i in range(2,10):
            image2=Image.open(path2+path+"_lbl"+str(i//10)+str(i%10)+".png")   
            image2=image2.convert('L');            
            label.append(image2) 
        '''
        now_time=time.time();
        part1_time+=now_time-prev_time;        
        prev_time=now_time;     
        '''
        '''   
        for i in range(len(label)):
            label[i]=np.array(label[i]);
        bg=255 - np.sum(label, axis=0, keepdims=True)
        labels = np.concatenate([bg, label], axis=0)
        labels = np.uint8(labels)          
        labels = [TF.to_pil_image(labels[i])
                  for i in range(labels.shape[0])]    
        '''
        sample={"image":image,"label":label,"index":index,"size":(image.size)}                
        sample=self.image_trans(sample);           
        sample['image_org']=sample['image'].copy();
        if (self.stage!='stage1'):
            sample['label_org']=sample['label'].copy();            
        sample=self.trans2(sample);
        sample['label'][0] = torch.sum(sample['label'][1:9], dim=0, keepdim=True)
        sample['label'][0]  = 1 - sample['label'][0]   
        #sample['label_org'][0] = torch.sum(sample['label_org'][1:9], dim=0, keepdim=True)
        #sample['label_org'][0]  = 1 - sample['label_org'][0]                                                 
        '''
        now_time=time.time();
        part2_time+=now_time-prev_time;        
        prev_time=now_time;     
        '''
        #sample['label']=sample['label'].unsqueeze(dim=0);
        #sample['label']=torch.zeros(9,1024,1024).scatter_(0, sample['label'], 255);#[L,s0,s1]                 
        '''
        print("loaddata part1_time=",part1_time);     
        print("loaddata part2_time=",part2_time);     
        print("loaddata part3_time=",part3_time);  
        '''
        return sample
    def preprocess_data(self,mode):
        if (mode=='train'):file_path=train_txt
        if (mode=='test'):file_path=test_txt
        if (mode=='val'):file_path=val_txt        
        with open(file_path) as f:
            lines=f.readlines()
            for line in lines:
                if (line.strip()==""):continue
                path=line.split(',')[1].strip()
                self.path_list.append(path);
            self.num=len(self.path_list);            
        print("Preprocess the {} data, it has {} images".format(mode, self.num))        

def get_loader_Aug(mode,batch_size,choice,stage):
    dataset = ConcatDataset([Helen_Aug(image_trans=choice.transforms[i], mode=mode,stage=stage)
                              for i in range(choice.num)]);
    if OnServer:
        data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'),num_workers=4)
    else:
        data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'),num_workers=0)
    #data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=0)
    return data_loader

