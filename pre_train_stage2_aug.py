import torch
import torchvision
from torchvision import datasets,transforms
from torchvision.transforms import functional as TF
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

from config import batch_size,pre_output_path,epoch_num,loss_image_path,OnServer,UseF1,pre_stage2_output_path
from model.model import EndtoEndModel,ETE_stage1,ETE_select,ETE_stage2,label_channel,label_list,make_inverse,calc_centroid
from data.loaddata import data_loader_Aug,data_loader_parts_Aug
import warnings
warnings.filterwarnings("ignore")

train_data=data_loader_parts_Aug("train",batch_size,"stage2");
test_data=data_loader_parts_Aug("test",batch_size,"stage2");
val_data=data_loader_parts_Aug("val",batch_size,"stage2");

use_gpu = torch.cuda.is_available()
if OnServer:
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
else:
    import matplotlib;matplotlib.use('TkAgg');
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      

#model_stage2=Network2();    
model_stage2=ETE_stage2(device);

bestloss=1000000
bestf1=0
def train(epoch):   
    model_stage2.train();     
    '''
    part1_time=0;    
    part2_time=0;    
    part3_time=0;         
    prev_time=time.time();             
    '''    
    unloader = transforms.ToPILImage()
    losstmp=0;
    k=0;
    
    for batch_idx,sample in enumerate(train_data.get_loader()):   
        '''
        now_time=time.time();
        part3_time+=now_time-prev_time;        
        prev_time=now_time;
        '''        
        if (use_gpu):
            for i in range(6):
                sample['image'][i]=sample['image'][i].to(device)                            
                sample['label'][i]=sample['label'][i].to(device)
        optimizer_stage2.zero_grad();     

        stage2_label=model_stage2(sample['image'],None);
        
        parts2=[];
        parts_label2=[];
        loss=[];
        
        for i in range(6):                 

            '''
            for j in range(sample['image'].size()[0]):
                if (not os.path.exists("./data/trainimg_output/"+train_data.get_namelist()[(k+j)%2000])):
                    os.mkdir("./data/trainimg_output/"+train_data.get_namelist()[(k+j)%2000]);                
                image3=transforms.ToPILImage()(sample['image_org'][j].cpu().clone()).convert('RGB')                   
                image3.save("./data/trainimg_output/"+train_data.get_namelist()[(k+j)%2000]+'/'+str((k+j)//2000)+'_orgimage'+'.jpg',quality=100);    
                     
                image3=transforms.ToPILImage()(parts2[i][j].cpu().clone()).convert('RGB')         
                image3.save("./data/trainimg_output/"+train_data.get_namelist()[(k+j)%2000]+'/'+str((k+j)//2000)+'lbl0'+str(i)+'_thetalabel'+'.jpg',quality=100);
                image3=unloader(np.uint8(parts_label2[i][j][1].cpu().detach().numpy()))                                       
                image3.save("./data/trainimg_output/"+train_data.get_namelist()[(k+j)%2000]+'/'+str((k+j)//2000)+'lbl0'+str(i)+'_label'+'_thetalabel'+'.jpg',quality=100); 
            '''                
            
            loss_tmp=fun.cross_entropy(stage2_label[i],sample['label'][i].argmax(dim=1, keepdim=False))                        
            loss.append(loss_tmp);
        k+=sample['image'][0].shape[0];
        if (batch_idx%100==0):            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(sample['image']), len(train_data.get_loader().dataset),
                100. * batch_idx / len(train_data.get_loader()),torch.sum(torch.stack(loss))))                            
        '''    
        now_time=time.time();
        part1_time+=now_time-prev_time;        
        prev_time=now_time;        
        '''        
        
        loss=torch.stack(loss)
        loss.backward(torch.ones(6, device=device, requires_grad=False))
        losstmp+=torch.sum(loss).item();
        
        optimizer_stage2.step();        
        
        '''
        now_time=time.time();
        part2_time+=now_time-prev_time;        
        prev_time=now_time;
        
        print("batch_idx=",batch_idx);                       
        print("part1_time=",part1_time);     
        print("part2_time=",part2_time);     
        print("part3_time=",part3_time);     
        '''                   
def test(epoch):
    model_stage2.eval();   
    global bestloss,bestf1;    
    test_loss=0    
    hists=[]
    for sample in val_data.get_loader():
        if (use_gpu):
            for i in range(6):
                sample['image'][i]=sample['image'][i].to(device)            
                sample['label'][i]=sample['label'][i].to(device)                   
            
        stage2_label=model_stage2(sample['image'],None);        
                            
        for i in range(6):                   
            test_loss+=fun.cross_entropy(stage2_label[i],sample['label'][i].argmax(dim=1, keepdim=False)).data;    
            
            output_2 = torch.softmax(stage2_label[i], dim=1).argmax(dim=1, keepdim=False)
            output_2=output_2.cpu().clone()
            target_2 = sample['label'][i].argmax(dim=1, keepdim=False)
            target_2=target_2.cpu().clone();            
            hist = np.bincount(9 * target_2.reshape([-1]) + output_2.reshape([-1]),minlength=81).reshape(9, 9)
            hists.append(hist);                              
    hists_sum=np.sum(np.stack(hists, axis=0), axis=0)
    for i in range(9):
        for j in range(9):
            print(hists_sum[i][j],end=' ')
        print();
    print();
    tp=0;
    tpfn=0;
    tpfp=0;
    f1score=0.0;
    for i in range(1,9):
        tp+=hists_sum[i][i].sum()
        tpfn+=hists_sum[i,:].sum()
        tpfp+=hists_sum[:,i].sum()   
        
    f1score=2*tp/(tpfn+tpfp)
    test_loss/=len(test_data.get_loader().dataset)
    print('\nTest set: {} Cases，Average loss: {:.4f}\n'.format(
        len(test_data.get_loader().dataset),test_loss))
    print("STN-iCNN tp=",tp)
    print("STN-iCNN tpfp=",tpfp)
    print("STN-iCNN tpfn=",tpfn)    
    print('\nTest set: {} Cases，F1 Score: {:.4f}\n'.format(
        len(test_data.get_loader().dataset),f1score))
            
    loss_list.append(test_loss.data.cpu().numpy());
    f1_list.append(f1score);
    if (UseF1):
        if (f1score>bestf1):
            bestf1=f1score
            print("Best data Updata\n");            
            torch.save(model_stage2,"./preBestNet_stage2")
    else:
        if (test_loss<bestloss):
            bestloss=test_loss
            print("Best data Updata\n");            
            torch.save(model_stage2,"./preBestNet_stage2")
            
def printoutput():
    model_stage2=torch.load("./preBestNet_stage2",map_location="cpu")
    if (use_gpu):
        model_stage2=model_stage2.to(device)    
    model_stage2.eval();   
    global bestloss,bestf1;    
    test_loss=0    
    hists=[]
    k=0;
    for sample in test_data.get_loader():
        if (use_gpu):
            for i in range(6):
                sample['image'][i]=sample['image'][i].to(device)            
                sample['label'][i]=sample['label'][i].to(device)
            
        stage2_label=model_stage2(sample['image'],None);                            
        
        for i in range(6):              
            test_loss+=fun.cross_entropy(stage2_label[i],sample['label'][i].argmax(dim=1, keepdim=False)).data;    
            for j in range(sample['image'].shape[0]):
                path=pre_stage2_output_path+'/aug'+'/'+test_data.get_namelist()[k+j]; 
                if not os.path.exists(path):
                    os.makedirs(path); 
                image=TF.to_pil_image(sample['image'][i][j].unsqueeze(0).cpu());
                image.save(path+'/'+test_data.get_namelist()[k+j]+'lbl0'+str(i)+'_img.jpg',quality=100);       
                image=TF.to_pil_image(sample['label'][i][j][1].unsqueeze(0).cpu());
                image.save(path+'/'+test_data.get_namelist()[k+j]+'lbl0'+str(i)+'_label.jpg',quality=100);                    
                image=torch.softmax(stage2_label[i][j].cpu(), dim=0).argmax(dim=0, keepdim=True);
                image=torch.zeros(label_channel[i],81,81).scatter_(0, image, 1);                                  
                image=TF.to_pil_image(image[1].unsqueeze(0),mode="L");
                image.save(path+'/'+test_data.get_namelist()[k+j]+'lbl0'+str(i)+'_train.jpg',quality=100);                
            
            output_2 = torch.softmax(stage2_label[i], dim=1).argmax(dim=1, keepdim=False)
            output_2=output_2.cpu().clone()
            target_2 = sample['label'][i].argmax(dim=1, keepdim=False)
            target_2=target_2.cpu().clone();            
            hist = np.bincount(9 * target_2.reshape([-1]) + output_2.reshape([-1]),minlength=81).reshape(9, 9)
            hists.append(hist);                                         
        k+=sample['image'][0].shape[0];
            
    hists_sum=np.sum(np.stack(hists, axis=0), axis=0)
    tp=0;
    tpfn=0;
    tpfp=0;
    f1score=0.0;
    for i in range(9):
        for j in range(9):
            print(hists_sum[i][j],end=' ')
        print()
    for i in range(1,9):
        tp+=hists_sum[i][i].sum()
        tpfn+=hists_sum[i,:].sum()
        tpfp+=hists_sum[:,i].sum()    
    f1score=2*tp/(tpfn+tpfp)    
    test_loss/=len(test_data.get_loader().dataset)
    print('\nPrintoutput Average loss: {:.4f}\n'.format(test_loss))
    print("STN-iCNN stage2 tp=",tp)
    print("STN-iCNN stage2 tpfp=",tpfp)
    print("STN-iCNN tstage2 pfn=",tpfn)    
    print('\nPrintoutputF1 Score: {:.4f}\n'.format(f1score))        
    print("printoutput Finish");    

def makeplt(title):
    loss_list=np.load(loss_image_path+'\\loss_list_'+plttitle+'.npy')
    loss_list=loss_list.tolist();
    f1_list=np.load(loss_image_path+'\\f1_list_'+plttitle+'.npy')
    f1_list=f1_list.tolist();
    x_list=np.load(loss_image_path+'\\x_list_'+plttitle+'.npy')
    x_list=x_list.tolist();
    
    fig = plt.figure()
    plt.title(title);
    ax1 = fig.add_subplot(111)
    ax1.plot(x_list, loss_list,'r',label="loss")    
    ax2 = ax1.twinx()
    ax2.plot(x_list, f1_list,'b',label="f1_score")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("f1_score")    
    ax1.legend(loc=2);
    ax2.legend(loc=4);    
    
    plt.savefig(loss_image_path+'\\loss_'+plttitle+'.jpg');    
    
loss_list=[];
f1_list=[];
x_list=[];
print("use_gpu=",use_gpu)
if (use_gpu):
    model_stage2=model_stage2.to(device)    
  
optimizer_stage2=optim.Adam(model_stage2.parameters(),lr=0.001) 
scheduler_stage2=optim.lr_scheduler.StepLR(optimizer_stage2, step_size=5, gamma=0.5)       

Training=OnServer;
#Training=True;
plttitle="PreTrain_stage2"
if Training:
    for epoch in range(epoch_num):
        x_list.append(epoch);
        train(epoch)
        scheduler_stage2.step()        
        test(epoch)
    torch.save(model_stage2,"./preNetdata_stage2")         
    '''
    x_list_stage2=np.array(x_list)
    np.save(loss_image_path+'\\x_list_'+plttitle+'.npy',x_list_stage2) 
    f1_list_stage2=np.array(f1_list)
    np.save(loss_image_path+'\\f1_list_'+plttitle+'.npy',f1_list_stage2) 
    loss_list_stage2=np.array(loss_list)
    np.save(loss_image_path+'\\loss_list_'+plttitle+'.npy',loss_list_stage2) 
    makeplt(plttitle);
    '''
if (not OnServer):
    printoutput()  
print("FINISH!");