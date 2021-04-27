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

from config import batch_size,pre_output_path,epoch_num,loss_image_path,OnServer,UseF1
from model.model import EndtoEndModel,ETE_stage1,ETE_select,ETE_stage2,label_channel,label_list,make_inverse
from data.loaddata import data_loader_Aug
import warnings
warnings.filterwarnings("ignore")

train_data=data_loader_Aug("train",batch_size,"stage1");
test_data=data_loader_Aug("test",batch_size,"stage1");
val_data=data_loader_Aug("val",batch_size,"stage1");

use_gpu = torch.cuda.is_available()
if OnServer:
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
else:
    import matplotlib;matplotlib.use('TkAgg');
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      

#model_stage1=Network2();    
model_stage1=ETE_stage1(device);

bestloss=1000000
bestf1=0
def train(epoch):   
    model_stage1.train();         
    '''
    part1_time=0;    
    part2_time=0;    
    part3_time=0;         
    prev_time=time.time();  
    '''
    for batch_idx,sample in enumerate(train_data.get_loader()):           
        '''
        now_time=time.time();
        part3_time+=now_time-prev_time;        
        prev_time=now_time;
        '''
        if (use_gpu):
            sample['image']=sample['image'].to(device)            
            sample['label']=sample['label'].to(device)                   
        target=sample['label'];
        optimizer_stage1.zero_grad();                                   
        stage1_label=model_stage1(sample['image'])        
        target=torch.softmax(target, dim=1).argmax(dim=1, keepdim=False);    
        loss=fun.cross_entropy(stage1_label,target)  
                                        
        '''
        now_time=time.time();
        part1_time+=now_time-prev_time;        
        prev_time=now_time;  
        '''
        loss.backward()                
        optimizer_stage1.step();          
        '''
        now_time=time.time();
        part2_time+=now_time-prev_time;        
        prev_time=now_time;
        '''
        if (batch_idx%250==0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(sample['image']), len(train_data.get_loader().dataset),
                100. * batch_idx / len(train_data.get_loader()),loss))
            '''
            print("batch_idx=",batch_idx);                       
            print("part1_time=",part1_time);     
            print("part2_time=",part2_time);     
            print("part3_time=",part3_time);    
            '''
def test():
    model_stage1.eval();         
    global bestloss,bestf1
    test_loss=0    
    hists=[]
    for sample in val_data.get_loader():
        if (use_gpu):
            sample['image']=sample['image'].to(device)                                            
            sample['label']=sample['label'].to(device)   
        stage1_label=model_stage1(sample['image'])                
        target=torch.softmax(sample['label'], dim=1).argmax(dim=1, keepdim=False);               
        if (use_gpu):
            test_loss+=fun.cross_entropy(stage1_label,target,size_average=False).to(device).data
        else:
            test_loss+=fun.cross_entropy(stage1_label,target,size_average=False).data           
        output=stage1_label.argmax(dim=1, keepdim=False);
        target=target.cpu().clone();            
        output=output.cpu().clone();
        hist = np.bincount(9 * output.reshape([-1]) + target.reshape([-1]),minlength=81).reshape(9, 9)
        hists.append(hist);                              
    hists_sum=np.sum(np.stack(hists, axis=0), axis=0)
    tp=0;
    tpfn=0;
    tpfp=0;
    f1score=0.0;
    for i in range(1,9):
        tp+=hists_sum[i][i].sum()
        tpfn+=hists_sum[i,:].sum()
        tpfp+=hists_sum[:,i].sum()   
        
    f1score=2*tp/(tpfn+tpfp)
    test_loss/=len(val_data.get_loader().dataset)
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
            print("Best data Stage1 Updata\n");
            torch.save(model_stage1,"./preBestNet_stage1")                       
    else:
        if (test_loss<bestloss):
            bestloss=test_loss
            print("Best data Stage1 Updata\n");
            torch.save(model_stage1,"./preBestNet_stage1")                       
def printoutput():
    model_stage1=torch.load("./preBestNet_stage1only",map_location="cpu")
    if (use_gpu):
        model_stage1=model_stage1.to(device)    
    unloader = transforms.ToPILImage()
    k=0;
    hists=[]                                 
    for sample in test_data.get_loader():
        if (use_gpu):
            sample['image']=sample['image'].to(device)            
            sample['label']=sample['label'].to(device)

        stage1_label=model_stage1(sample['image'])             
        
        for i in range(batch_size):             
            path=pre_output_path+'/'+test_data.get_namelist()[k];   
            if not os.path.exists(path):
                os.makedirs(path);                
            image=sample['image'][i].cpu().clone();                
            image =unloader(image)
            image.save(path+'/'+test_data.get_namelist()[k]+'.jpg',quality=100);                            
            
            output=stage1_label[i].cpu().clone();
            output=torch.softmax(output, dim=0).argmax(dim=0, keepdim=False) 
            output=output.unsqueeze(0);            
            output=torch.zeros(9,128,128).scatter_(0, output, 255);  
            for j in range(9):                                
                image3=unloader(np.uint8(output[j].numpy()))                
                image3.save(path+'/'+test_data.get_namelist()[k]+'lbl0'+str(j)+'.jpg',quality=100);            
            k+=1
            if (k>=test_data.get_len()):break        
                    
        target2=torch.softmax(sample['label'], dim=1).argmax(dim=1, keepdim=False);
        target2=target2.cpu().clone(); 
        output2=stage1_label.cpu().detach().clone();                 
        output2=output2.argmax(dim=1, keepdim=False);        
        hist = np.bincount(9 * target2.reshape([-1]) + output2.reshape([-1]),minlength=81).reshape(9, 9)
        hists.append(hist);
        if (k>=test_data.get_len()):break        
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
    print('Printoutput F1 Score: {:.4f}\n'.format(f1score))
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
    model_stage1=model_stage1.to(device)    
  
optimizer_stage1=optim.Adam(model_stage1.parameters(),lr=0.001) 
scheduler_stage1=optim.lr_scheduler.StepLR(optimizer_stage1, step_size=5, gamma=0.5)       

Training=OnServer;
#Training=True;
plttitle="PreTrain_stage1"
if Training:
    for epoch in range(epoch_num):
        x_list.append(epoch);
        train(epoch)
        scheduler_stage1.step()        
        test()
    torch.save(model_stage1,"./preNetdata_stage1_only")         
    x_list_stage1=np.array(x_list)
    np.save(loss_image_path+'\\x_list_'+plttitle+'.npy',x_list_stage1) 
    f1_list_stage1=np.array(f1_list)
    np.save(loss_image_path+'\\f1_list_'+plttitle+'.npy',f1_list_stage1) 
    loss_list_stage1=np.array(loss_list)
    np.save(loss_image_path+'\\loss_list_'+plttitle+'.npy',loss_list_stage1) 
    makeplt(plttitle);
if (not OnServer):
    printoutput()  
print("FINISH!");