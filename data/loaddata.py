import torch
from torchvision import datasets,transforms
from torch.utils import data
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image
import numpy as np
from torchvision.transforms import functional as TF
from config import train_txt,val_txt,test_txt,image_path,label_path,OnServer
from data.Helentransform import GaussianNoise,RandomAffine,Resize,ToTensor,DoNothing,ToPILImage,HorizontalFlip,Padding_leftup,Padding_mid,Padding_oldchange,Padding_old,ToTensor_label
import time,datetime
from model.model import label_channel,label_list
from prefetch_generator import BackgroundGenerator
import matplotlib.pyplot as plt
if not OnServer:
    import matplotlib;matplotlib.use('TkAgg');

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
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
                                        RandomAffine(degrees=0, translate=(0.3, 0.3), scale=(1,1)),
                                        RandomAffine(degrees=0, translate=(0,0), scale=(0.8, 1.5))]);
      
            self.augmentation.append([RandomAffine(degrees=0, translate=translate_range, scale=scale_range),
                                        RandomAffine(degrees=degree, translate=(0,0), scale=scale_range),
                                        RandomAffine(degrees=degree, translate=translate_range, scale=(1,1))]);
                    
            self.augmentation.append([RandomAffine(degrees=degree, translate=translate_range,scale=scale_range)]);  
        self.num=len(self.augmentation);
        '''
        self.transforms=[];        
        for i in range(self.num):
            self.transforms.append(transforms.Compose([                                
                transforms.RandomChoice(self.augmentation[i]),  
                                                                     
            ]));
        '''
class data_loader:
    def __init__(self,mode,batch_size):
        self._path_list=[]
        self._num=0;        
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
        self._data_loader=get_loader(mode,batch_size);        
    def get_loader(self):
        return self._data_loader;
    def get_namelist(self):
        return self._path_list;
    def get_len(self):
        return self._num;    
        
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
    
class data_loader_parts_Aug:
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
        self._data_loader=get_loader_parts_Aug(mode,batch_size,self.choice,stage);        
    def get_loader(self):
        return self._data_loader;
    def get_namelist(self):
        return self._path_list;
    def get_len(self):
        return self._num;  
    
class Helen(data.Dataset):
    def __init__(self,mode='train'):        
        self.path_list=[]        
        self.num=0;
        self.mode=mode        
        self.preprocess_data(mode);           
        self.sum_trans=transforms.Compose([ToTensor_label(),Padding_mid(size=(1024, 1024))]);
        self.sum_trans_leftup=transforms.Compose([ToTensor_label(),Padding_leftup(size=(1024, 1024))]);
    def __len__(self):
        return self.num;
    def __getitem__(self,index):   
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
        sample={"index":index,"size":image.size,'image_org':image,'label_org':label}  
        sample=self.sum_trans(sample);        
        assert sample['image_org'].shape == (3, 1024, 1024)
        assert sample['label_org'].shape == (9, 1024, 1024)        
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

class Helen_Aug(data.Dataset):
    def __init__(self,augmentation,mode='train',stage="stage1"):
        self.augmentation=augmentation          
        self.path_list=[]        
        self.num=0;
        self.mode=mode
        self.stage=stage;
        self.preprocess_data(mode);           
        '''
        if stage=="stage1":
            self.trans2=transforms.Compose([Resize(size=(128, 128),interpolation=Image.NEAREST)]);                          
        else:
            self.trans2=transforms.Compose([Resize_image(size=(128, 128),interpolation=Image.NEAREST)]); 
        '''        
        self.trans2=transforms.Compose([Resize(size=(128, 128),interpolation=Image.NEAREST)]);
        #self.paddtrans=transforms.Compose([Padding(size=(1024, 1024)),ToTensor()]);   
        self.paddtrans=transforms.Compose([Padding_mid(size=(1024, 1024)),ToTensor()]);   
        
        self.image_trans=transforms.Compose([transforms.RandomChoice(self.augmentation),]);
        self.image_trans2=transforms.Compose([Resize(size=(128, 128),interpolation=Image.NEAREST),
                                           ToTensor(),Padding_mid(size=(1024, 1024))]);        
        self.sum_trans=transforms.Compose([transforms.RandomChoice(self.augmentation),
                                           Resize(size=(128, 128),interpolation=Image.NEAREST),
                                           ToTensor(),Padding_mid(size=(1024, 1024))]);
        self.sum_trans_leftup=transforms.Compose([transforms.RandomChoice(self.augmentation),
                                   Resize(size=(128, 128),interpolation=Image.NEAREST),
                                   ToTensor(),Padding_leftup(size=(1024, 1024))]);
        self.sum_trans_oldchange=transforms.Compose([transforms.RandomChoice(self.augmentation),
                           Resize(size=(128, 128),interpolation=Image.NEAREST),
                           ToTensor(),Padding_oldchange(size=(1024, 1024))]);
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
        label = np.concatenate([bg, label], axis=0)
        label = np.uint8(label)
        label = [TF.to_pil_image(label[i])
                  for i in range(label.shape[0])]                    
        '''
        
        sample={"image":image,"label":label,"index":index,"size":image.size,'image_org':image,'label_org':label}  
        #sample=self.sum_trans_leftup(sample);
        sample=self.sum_trans(sample);
        assert sample['image'].shape == (3, 128, 128)
        assert sample['label'].shape == (9, 128, 128)
        assert sample['image_org'].shape == (3, 1024, 1024)
        assert sample['label_org'].shape == (9, 1024, 1024)      
        '''
        sample={"image":image,"label":label,"index":index,"size":image.size}                    
        sample=self.image_trans(sample);           
        sample['image_org']=sample['image'].copy();
        if (self.mode=="test"):
            sample['label_org']=sample['label'].copy();
        sample=self.trans2(sample);                        
        paddtrans=transforms.Compose([Padding_old(size=(1024, 1024),pdlabel=False),ToTensor()]);
        sample=paddtrans(sample);
        '''
        '''
        #check!!!        
        sample={"image":image,"label":label,"index":index,"size":image.size}                    
        sample=self.image_trans(sample);           
        sample['image_org']=sample['image'].copy();
        sample=self.trans2(sample);                
        
        paddtrans=transforms.Compose([Padding(size=(1024, 1024),pdlabel=(self.stage!="stage1")),ToTensor()]);                
        sample=paddtrans(sample);        
        sample['label']=torch.softmax(sample['label'],dim=0);
        sample['label'][0] = torch.sum(sample['label'][1:9], dim=0, keepdim=True);
        sample['label'][0]  = 1 - sample['label'][0];             
        #check!!!
        '''
        
        '''
        sample=self.image_trans(sample);        
        #sample['image_org']=sample['image'];        
        #sample['label_org']=sample['label'];        
        sample=self.trans2(sample);                                         
        sample=self.paddtrans(sample);  
        ''' 
        '''
        sample['label']=torch.softmax(sample['label'],dim=0);
        sample['label'][0] = torch.sum(sample['label'][1:9], dim=0, keepdim=True);
        sample['label'][0]  = 1 - sample['label'][0]; 
        sample['label_org']=torch.softmax(sample['label_org'],dim=0);
        sample['label_org'][0] = torch.sum(sample['label_org'][1:9], dim=0, keepdim=True);
        sample['label_org'][0]  = 1 - sample['label_org'][0]; 
        '''    
        '''
        lnum=len(sample['label']);        
        sample['label']=torch.argmax(sample['label'],dim=0,keepdim=False);#[batch_size,1024,1024]                
        sample['label']=sample['label'].unsqueeze(dim=0);#[batch_size,1,1024,1024]        
        sample['label']=torch.zeros(lnum,128,128).scatter_(0, sample['label'], 1);#[batch_size,L,1024,1024]                                 
        sample['label_org']=torch.argmax(sample['label_org'],dim=0,keepdim=False);#[batch_size,1024,1024]                
        sample['label_org']=sample['label_org'].unsqueeze(dim=0);#[batch_size,1,1024,1024]        
        sample['label_org']=torch.zeros(lnum,1024,1024).scatter_(0, sample['label_org'], 1);#[batch_size,L,1024,1024]         
        '''
        '''                         
        sample['label'][0] = torch.sum(sample['label'][1:9], dim=0, keepdim=True);
        sample['label'][0]  = 1 - sample['label'][0];         
        sample['label_org'][0] = torch.sum(sample['label_org'][1:9], dim=0, keepdim=True);
        sample['label_org'][0]  = 1 - sample['label_org'][0];         
        '''
        '''
        now_time=time.time();
        part2_time+=now_time-prev_time;        
        prev_time=now_time;             
        
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
        
class Helen_parts_Aug(data.Dataset):
    def __init__(self,augmentation,mode='train',stage="stage1"):
        self.augmentation=augmentation          
        self.path_list=[]        
        self.num=0;
        self.mode=mode
        self.stage=stage;
        self.preprocess_data(mode);           

        self.sum_trans=[transforms.Compose([transforms.RandomChoice(self.augmentation),
                                           ToTensor(),])for r in range(6)];
    def __len__(self):
        return self.num;
    def __getitem__(self,index):   
        path=self.path_list[index];        
        image_f=[];
        label_f=[];
        for k in range(6):
            path2='./data/facial_parts/'+path+'/'+"lbl0"+str(k)+"_img.jpg"
            image=Image.open(path2)
            label=[];   
            path2=label_path+'/'+path+'/'
            for i in range(label_channel[k]):
                path3='./data/facial_parts/'+path+'/'+"lbl0"+str(k)+"_label0"+str(i)+".jpg"#label_list[i]
                image2=Image.open(path3);
                image2=image2.convert('L');            
                label.append(image2);

            sample_tmp={"image":image,"label":label}   
            sample_tmp=self.sum_trans[k](sample_tmp); 
            image_f.append(sample_tmp['image']);
            label_f.append(sample_tmp['label']);
        sample={"image":image_f,"label":label_f}
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
    #dataset = ConcatDataset([Helen_Aug(augmentation=choice.augmentation[i], mode=mode,stage=stage)for i in range(2,choice.num)]);        
    dataset = ConcatDataset([Helen_Aug(augmentation=choice.augmentation[i], mode=mode,stage=stage)for i in range(choice.num)]);
    if OnServer:
        data_loader = DataLoaderX(dataset, batch_size=batch_size, shuffle=(mode=='train'),num_workers=4)
    else:
        data_loader = DataLoaderX(dataset, batch_size=batch_size, shuffle=(mode=='train'),num_workers=0)
    data_loader = DataLoaderX(dataset, batch_size=batch_size, shuffle=False,num_workers=0)
    return data_loader

def get_loader_parts_Aug(mode,batch_size,choice,stage):
    dataset = ConcatDataset([Helen_parts_Aug(augmentation=choice.augmentation[i], mode=mode,stage=stage)
                              for i in range(choice.num)]);    
    if OnServer:
        data_loader = DataLoaderX(dataset, batch_size=batch_size, shuffle=(mode=='train'),num_workers=4)
    else:
        data_loader = DataLoaderX(dataset, batch_size=batch_size, shuffle=(mode=='train'),num_workers=0)
    #data_loader = DataLoaderX(dataset, batch_size=batch_size, shuffle=False,num_workers=0)
    return data_loader

def get_loader(mode,batch_size):
    dataset = Helen(mode=mode)
    if OnServer:
        data_loader = DataLoaderX(dataset, batch_size=batch_size, shuffle=False,num_workers=4)
    else:
        data_loader = DataLoaderX(dataset, batch_size=batch_size, shuffle=False,num_workers=0)    
    return data_loader

