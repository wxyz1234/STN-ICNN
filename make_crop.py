import torch
import torchvision
from torchvision import datasets,transforms
from torchvision.transforms import functional as TF
from config import OnServer
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim
from PIL import Image
import numpy as np
from torch.autograd import Variable

import sys
import os
from skimage import io
import matplotlib.pyplot as plt
import pylab
import time,datetime

from config import batch_size,output_path,epoch_num,loss_image_path,OnServer,UseF1
from model.model import ETE_stage1,ETE_select,ETE_stage2,label_channel,label_list,make_inverse,calc_centroid
from data.loaddata import data_loader,data_loader_Aug

from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

train_data=data_loader("train",batch_size);
test_data=data_loader("test",batch_size);
val_data=data_loader("val",batch_size);

use_gpu = torch.cuda.is_available()
if OnServer:
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
else:
    import matplotlib;matplotlib.use('TkAgg');
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      

def printoutput(in_data):
    unloader = transforms.ToPILImage()
    k=0;
    hists=[]                        
    global label_list;       
    for sample in in_data.get_loader():        
        if (use_gpu):
            sample['image_org']=sample['image_org'].to(device)  
            sample['label_org']=sample['label_org'].to(device)  
        N=sample['image_org'].shape[0];                     
        theta_label = torch.zeros((N,6,2,3),device=device,requires_grad=False); #[batch_size,6,2,3]    
        W=1024.0;
        H=1024.0;
        cens = torch.floor(calc_centroid(sample['label_org'])) #[batch_size,9,2]          
        points = torch.floor(torch.cat([cens[:, 1:6],cens[:, 6:9].mean(dim=1, keepdim=True)],dim=1)) #[batch_size,6,2]
        for i in range(6):
            theta_label[:,i,0,0]=(81.0-1.0)/(W-1.0);
            theta_label[:,i,1,1]=(81.0-1.0)/(H-1.0);
            theta_label[:,i,0,2]=-1+2*points[:,i,0]/(W-1.0);
            theta_label[:,i,1,2]=-1+2*points[:,i,1]/(H-1.0);         
        parts=[];
        parts_label=[];        
        for i in range(6):            
            affine_stage2=F.affine_grid(theta_label[:,i],(N,3,81,81),align_corners=True);
            parts.append(F.grid_sample(sample['image_org'],affine_stage2,align_corners=True));
            affine_stage2=F.affine_grid(theta_label[:,i],(N,label_channel[i],81,81),align_corners=True);
            parts_label.append(F.grid_sample(sample['label_org'][:,label_list[i]],affine_stage2,align_corners=True));            
            parts_label[i][:,0]+=0.00001;           
            for j in range(sample['image_org'].shape[0]):                
                parts_label_tmp=parts_label[i][j].argmax(dim=0, keepdim=True);                
                parts_label[i][j]=torch.zeros(label_channel[i],81,81).to(device).scatter_(0, parts_label_tmp, 255);             
            for j in range(N):
                path="./data/facial_parts/"+in_data.get_namelist()[k+j];
                if (not os.path.exists(path)):
                    os.mkdir(path);                
                image3=transforms.ToPILImage()(parts[i][j].cpu().clone()).convert('RGB')                                              
                image3.save(path+'/'+'lbl0'+str(i)+'_img'+'.jpg',quality=100);                 
                for l in range(label_channel[i]):
                    image3=unloader(np.uint8(parts_label[i][j][l].cpu().detach().numpy()))                                       
                    image3.save(path+'/'+'lbl0'+str(i)+'_label0'+str(l)+'.jpg',quality=100);             
        k+=N;
        if (k%200==0):print(k);
    print("Printoutput Finish!");

    
print("use_gpu=",use_gpu)
printoutput(train_data);
printoutput(val_data);
printoutput(test_data);