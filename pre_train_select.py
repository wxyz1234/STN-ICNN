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
from model.model import EndtoEndModel,ETE_stage1,ETE_select,ETE_stage2,label_channel,label_list,make_inverse,calc_centroid
from data.loaddata import data_loader_Aug
import warnings
warnings.filterwarnings("ignore")

train_data=data_loader_Aug("train",batch_size,"stage1");
test_data=data_loader_Aug("test",batch_size,"stage1");
val_data=data_loader_Aug("val",batch_size,"stage1");

use_gpu = torch.cuda.is_available()
if OnServer:
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
else:
    import matplotlib;matplotlib.use('TkAgg');
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      
    
model_stage1=ETE_stage1(device);
model_select=ETE_select(device);

bestloss=1000000

def calc_centroid_old(tensor):#input [batch_size,9,128,128]->output [batch_size,9,2]
    input=tensor.float()+1e-10;
    n, l, h, w = input.shape    
    index_y=torch.from_numpy(np.arange(h)).float().to(tensor.device);
    index_x=torch.from_numpy(np.arange(w)).float().to(tensor.device);    
    center_x=input.sum(2)*index_x.view(1,1,-1);
    center_x=center_x.sum(2,keepdim=True)/input.sum([2, 3]).view(n, l, 1);
    center_y=input.sum(3)*index_y.view(1,1,-1);
    center_y=center_y.sum(2,keepdim=True)/input.sum([2, 3]).view(n, l, 1);
    output = torch.cat([center_x, center_y], 2)
    return output;

def train(epoch):   
    model_stage1.train();     
    model_select.train();   
    '''
    part1_time=0;    
    part2_time=0;    
    part3_time=0;         
    prev_time=time.time();  
    '''
    k=0;
    for batch_idx,sample in enumerate(train_data.get_loader()):           
        '''
        now_time=time.time();
        part3_time+=now_time-prev_time;        
        prev_time=now_time;
        '''
        if (use_gpu):
            sample['image']=sample['image'].to(device)            
            sample['label']=sample['label'].to(device)               
            sample['size'][0]=sample['size'][0].to(device);
            sample['size'][1]=sample['size'][1].to(device);
            
        optimizer_stage1.zero_grad();                           
        optimizer_select.zero_grad();                
        
        stage1_label=model_stage1(sample['image'])             
        theta=model_select(stage1_label,sample['size'])

        theta_label = torch.zeros((sample['image'].size()[0],6,2,3),device=device,requires_grad=False); #[batch_size,6,2,3]    
        W=1024.0;
        H=1024.0;
        '''
        cens = torch.floor(calc_centroid_old(sample['label'])) #[batch_size,9,2]           
        for i in range(sample['image'].size()[0]):    
            for j in range(9):
                cens[i,j,0]=cens[i,j,0]*(sample['size'][0][i]-1.0)/(128.0-1.0)
                cens[i,j,1]=cens[i,j,1]*(sample['size'][1][i]-1.0)/(128.0-1.0)        
        points = torch.floor(torch.cat([cens[:, 1:6],cens[:, 6:9].mean(dim=1, keepdim=True)],dim=1)) #[batch_size,6,2]
        '''
        '''
        points2 = torch.floor(calc_centroid(sample['label_org'])) #[batch_size,9,2]
        print("cens resize:");
        print(points);
        print("cens org:");   
        print(points2);
        print("delta");
        print(points.cpu()-points2);
        input("wait");
        '''
        points=torch.floor(calc_centroid(sample['label_org']))  
        for i in range(6):
            theta_label[:,i,0,0]=(81.0-1.0)/(W-1.0);
            theta_label[:,i,1,1]=(81.0-1.0)/(H-1.0);
            theta_label[:,i,0,2]=-1+2*points[:,i,0]/(W-1.0);
            theta_label[:,i,1,2]=-1+2*points[:,i,1]/(H-1.0); 
        if (torch.min(theta_label)<-1 or torch.max(theta_label)>1):
            print("FUCK");
            print(k);

        '''
        for i in range(sample['image'].shape[0]):            
            if (not os.path.exists("./data/select_pre/"+train_data.get_namelist()[(k+i)%2000])):
                os.mkdir("./data/select_pre/"+train_data.get_namelist()[(k+i)%2000]);
            image=sample['image_org'][i].cpu().clone();                                 
            image=transforms.ToPILImage()(image).convert('RGB')
            plt.imshow(image);        
            plt.show(block=True);                
            image.save('./data/select_pre/'+train_data.get_namelist()[(k+i)%2000]+'/'+str((k+i)//2000)+'_img'+'.jpg',quality=100);
            for j in range(6):                            
                affine_stage2=F.affine_grid(theta_label[i][j].unsqueeze(0),(1,1,81,81),align_corners=True);    
                image=F.grid_sample(sample['label_org'][i][label_list[j][1]].unsqueeze(0).unsqueeze(0).to(device),affine_stage2,align_corners=True);
                image=image.squeeze(0).cpu();                                                
                image=transforms.ToPILImage()(image);
                image.save('./data/select_pre/'+train_data.get_namelist()[(k+i)%2000]+'/'+str((k+i)//2000)+'_'+str(j)+'_thetalabel'+'.jpg',quality=100);
                image=sample['label_org'][i][label_list[j][1]]
                image=transforms.ToPILImage()(image);
                image.save('./data/select_pre/'+train_data.get_namelist()[(k+i)%2000]+'/'+str((k+i)//2000)+'_'+str(j)+'_orglabel'+'.jpg',quality=100);
                #plt.imshow(image);        
                #plt.show(block=True);      
        '''
        
        k+=sample['image'].shape[0];
        loss=fun.smooth_l1_loss(theta, theta_label); 
        '''
        now_time=time.time();
        part1_time+=now_time-prev_time;        
        prev_time=now_time;  
        '''
        loss.backward()
                
        optimizer_select.step();
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
    model_select.eval();         
    global bestloss
    test_loss=0    
    for sample in val_data.get_loader():
        if (use_gpu):
            sample['image']=sample['image'].to(device)                                            
            sample['label']=sample['label'].to(device)   
            sample['size'][0]=sample['size'][0].to(device);
            sample['size'][1]=sample['size'][1].to(device);
        stage1_label=model_stage1(sample['image'])              
        theta=model_select(stage1_label,sample['size'])      
        
        theta_label = torch.zeros((sample['image'].size()[0],6,2,3),device=device,requires_grad=False); #[batch_size,6,2,3]    
        W=1024.0;
        H=1024.0;
        '''        
        cens = torch.floor(calc_centroid_old(sample['label'])) #[batch_size,9,2]     
        for i in range(sample['image'].size()[0]):    
            for j in range(9):
                cens[i,j,0]=cens[i,j,0]*(sample['size'][0][i]-1.0)/(128.0-1.0)
                cens[i,j,1]=cens[i,j,1]*(sample['size'][1][i]-1.0)/(128.0-1.0)        
        points = torch.floor(torch.cat([cens[:, 1:6],cens[:, 6:9].mean(dim=1, keepdim=True)],dim=1)) #[batch_size,6,2]
        '''
        points=torch.floor(calc_centroid(sample['label_org']))  
        for i in range(6):
            theta_label[:,i,0,0]=(81.0-1.0)/(W-1.0);
            theta_label[:,i,1,1]=(81.0-1.0)/(H-1.0);
            theta_label[:,i,0,2]=-1+2*points[:,i,0]/(W-1.0);
            theta_label[:,i,1,2]=-1+2*points[:,i,1]/(H-1.0); 
            
        loss=fun.smooth_l1_loss(theta, theta_label); 
        test_loss+=fun.smooth_l1_loss(theta, theta_label).data;        
    test_loss/=len(val_data.get_loader().dataset)    
    print('\nTest set: {} Cases，Average loss: {:.8f}\n'.format(
        len(test_data.get_loader().dataset),test_loss))        
    loss_list.append(test_loss.data.cpu().numpy());
    if (test_loss<bestloss):
        bestloss=test_loss
        print("Best data Stage1 Updata\n");
        torch.save(model_stage1,"./preBestNet_stage1")           
        torch.save(model_select,"./preBestNet_select")                          
def printoutput():
    CheckBest=True;
    if (CheckBest):
        model_stage1=torch.load("./preBestnet_stage1",map_location="cpu")
        model_select=torch.load("./preBestnet_select",map_location="cpu")
    else:
        model_stage1=torch.load("./preNetdata_stage1",map_location="cpu")
        model_select=torch.load("./preNetdata_select",map_location="cpu")
    model_select.select.change_device(device);
    if (use_gpu):
        model_stage1=model_stage1.to(device)
        model_select=model_select.to(device)
    unloader = transforms.ToPILImage()
    k=0;   
    loss=0;
    for sample in test_data.get_loader():
        if (use_gpu):
            sample['image']=sample['image'].to(device)            
            sample['label']=sample['label'].to(device)            
            sample['size'][0]=sample['size'][0].to(device);
            sample['size'][1]=sample['size'][1].to(device);
            sample['label_org']=sample['label_org'].to(device)    

        stage1_label=model_stage1(sample['image'])
        theta=model_select(stage1_label,sample['size'])                
        theta_label = torch.zeros((sample['image'].size()[0],6,2,3),device=device,requires_grad=False); #[batch_size,6,2,3]            
        W=1024.0;
        H=1024.0;
        '''
        cens = torch.floor(calc_centroid_old(sample['label'])) #[batch_size,9,2]           
        for i in range(sample['image'].size()[0]):    
            for j in range(9):
                cens[i,j,0]=cens[i,j,0]*(sample['size'][0][i]-1.0)/(128.0-1.0)
                cens[i,j,1]=cens[i,j,1]*(sample['size'][1][i]-1.0)/(128.0-1.0)        
        points = torch.floor(torch.cat([cens[:, 1:6],cens[:, 6:9].mean(dim=1, keepdim=True)],dim=1)) #[batch_size,6,2]                    
        points2 = torch.floor(calc_centroid(sample['label_org'])) #[batch_size,9,2]
        theta_label2 = torch.zeros((sample['image'].size()[0],6,2,3),device=device,requires_grad=False); #[batch_size,6,2,3]    
        for i in range(6):
            theta_label2[:,i,0,0]=(81.0-1.0)/(W-1.0);
            theta_label2[:,i,1,1]=(81.0-1.0)/(H-1.0);
            theta_label2[:,i,0,2]=-1+2*points2[:,i,0]/(W-1.0);
            theta_label2[:,i,1,2]=-1+2*points2[:,i,1]/(H-1.0);           
        if (abs(torch.max(points.cpu()-points2.cpu()))>20 or abs(torch.min(points.cpu()-points2.cpu()))>20):
            print("points resize:");
            print(points);
            print("points org:");   
            print(points2);
            print("delta");
            print(points.cpu()-points2.cpu());
            for i in range(sample['image'].size()[0]):
                print(test_data.get_namelist()[k+i]);
            input("wait");
        '''        
        points=torch.floor(calc_centroid(sample['label_org']))    
        for i in range(6):
            theta_label[:,i,0,0]=(81.0-1.0)/(W-1.0);
            theta_label[:,i,1,1]=(81.0-1.0)/(H-1.0);
            theta_label[:,i,0,2]=-1+2*points[:,i,0]/(W-1.0);
            theta_label[:,i,1,2]=-1+2*points[:,i,1]/(H-1.0);               
        
        loss+=fun.smooth_l1_loss(theta, theta_label).detach().data; 
        
        '''        
        f=open("1.txt",mode='w');
        print(sample['label'].size())
        for k in range(9):
            for i in range(sample['label'][0][k].size()[0]):
                for j in range(sample['label'][0][k].size()[1]):
                    print(float(sample['label'][0][k][i][j].data),end=' ',file=f);
                print(file=f);
        f.close();
        for i in range(batch_size):
            for j in range(9):
                image=sample['label'][i][j].cpu().clone();                                 
                image=transforms.ToPILImage()(image).convert('L')
                plt.imshow(image);
                plt.show(block=True);   
        input("check")
        '''
        
        '''
        print(theta);
        print(theta_label);
        input("check")
        '''
        output=[];
        for i in range(sample['image'].size()[0]):      
            '''
            if (test_data.get_namelist()[k]=="13601661_1"):
                print(cens[i]);
                print(points[i]);
                print(theta_label[i]);
                input("wait")
            '''
            path=pre_output_path+'/'+test_data.get_namelist()[k];   
            if not os.path.exists(path):
                os.makedirs(path);                
            image=sample['image'][i].cpu().clone();                
            image =unloader(image)
            image.save(path+'/'+test_data.get_namelist()[k]+'_img.jpg',quality=100);      
                      
            image=sample['image_org'][i].cpu().clone();               
            image2 =unloader(image)
            image2.save(path+'/'+test_data.get_namelist()[k]+'_img_org.jpg',quality=100);                            
        
            output2=stage1_label[i].cpu().clone();
            output2=torch.softmax(output2, dim=0).argmax(dim=0, keepdim=False) 
            output2=output2.unsqueeze(0);            
            output2=torch.zeros(9,128,128).scatter_(0, output2, 255); 
                
            output.append(output2);             
            for j in range(9):                                
                image3=unloader(np.uint8(output[i][j].numpy()))
                #image3=transforms.ToPILImage()(output[i][j].cpu().detach()).convert('L')
                image3.save(path+'/'+'stage1_'+test_data.get_namelist()[k]+'lbl0'+str(j)+'.jpg',quality=100);         
                image3=transforms.ToPILImage()(sample['label_org'][i][j].cpu().detach()).convert('L')
                image3.save(path+'/'+test_data.get_namelist()[k]+'lbl0'+str(j)+'_label'+'.jpg',quality=100);  
            image=image.to(device).unsqueeze(0);            
            for j in range(6):   
                affine_stage2=F.affine_grid(theta[i,j].unsqueeze(0),(1,3,81,81),align_corners=True);                                
                image3=F.grid_sample(sample['image_org'][i].to(device).unsqueeze(0),affine_stage2,align_corners=True);   
                image3=unloader(image3[0].cpu());
                image3.save(path+'/'+'select_'+test_data.get_namelist()[k]+'lbl0'+str(j)+'.jpg',quality=100);                   
                affine_stage2=F.affine_grid(theta_label[i,j].unsqueeze(0),(1,3,81,81),align_corners=True);                                
                image3=F.grid_sample(sample['image_org'][i].to(device).unsqueeze(0),affine_stage2,align_corners=True);   
                image3=unloader(image3[0].cpu());
                image3.save(path+'/'+'select_'+test_data.get_namelist()[k]+'lbl0'+str(j)+'_thetalabel'+'.jpg',quality=100);   
                '''
                affine_stage2=F.affine_grid(theta_label2[i,j].unsqueeze(0),(1,3,81,81),align_corners=True);                                
                image3=F.grid_sample(sample['image_org'][i].to(device).unsqueeze(0),affine_stage2,align_corners=True);   
                image3=unloader(image3[0].cpu());
                image3.save(path+'/'+'select_'+test_data.get_namelist()[k]+'lbl0'+str(j)+'_thetalabel2'+'.jpg',quality=100);   
                '''
                affine_stage2=F.affine_grid(theta[i,j].unsqueeze(0),(1,1,81,81),align_corners=True);    
                image3=F.grid_sample(sample['label_org'][i][label_list[j][1]].unsqueeze(0).unsqueeze(0),affine_stage2,align_corners=True);                   
                image3=transforms.ToPILImage()(image3[0].squeeze(0).cpu().detach()).convert('L');
                image3.save(path+'/'+'select_'+test_data.get_namelist()[k]+'lbl0'+str(j)+'_label'+'.jpg',quality=100);   
                affine_stage2=F.affine_grid(theta_label[i,j].unsqueeze(0),(1,1,81,81),align_corners=True);           
                image3=F.grid_sample(sample['label_org'][i][label_list[j][1]].unsqueeze(0).unsqueeze(0),affine_stage2,align_corners=True);   
                image3=transforms.ToPILImage()(image3[0].squeeze(0).cpu().detach()).convert('L');
                image3.save(path+'/'+'select_'+test_data.get_namelist()[k]+'lbl0'+str(j)+'_label'+'_thetalabel'+'.jpg',quality=100);   
            k+=1
            if (k>=test_data.get_len()):break        
            
        if (k>=test_data.get_len()):break   
    loss/=len(val_data.get_loader().dataset);
    print('\nTest set: {} Cases，Average loss: {:.8f}\n'.format(len(test_data.get_loader().dataset),loss))
    print("printoutput Finish");    
    
def makeplt(title):
    loss_list=np.load(loss_image_path+'\\loss_list_'+plttitle+'.npy')
    loss_list=loss_list.tolist();    
    x_list=np.load(loss_image_path+'\\x_list_'+plttitle+'.npy')
    x_list=x_list.tolist();
    fig = plt.figure()
    plt.title(title);
    ax1 = fig.add_subplot(111)
    ax1.plot(x_list, loss_list,'r',label="loss")        
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc=2);    
    
    plt.savefig(loss_image_path+'\\loss_'+plttitle+'.jpg');    
    plt.close(1)
loss_list=[];
x_list=[];
print("use_gpu=",use_gpu)
if (use_gpu):
    model_stage1=model_stage1.to(device)
    model_select=model_select.to(device)
  
optimizer_stage1=optim.Adam(model_stage1.parameters(),lr=0.01) 
optimizer_select=optim.Adam(model_select.parameters(),lr=0.01) 
scheduler_stage1=optim.lr_scheduler.StepLR(optimizer_stage1, step_size=5, gamma=0.5)       
scheduler_select=optim.lr_scheduler.StepLR(optimizer_select, step_size=5, gamma=0.5)       

plttitle="PreTrain_stage1select"
Training=OnServer;
#Training=True;
if Training:
    for epoch in range(epoch_num):
        x_list.append(epoch);
        train(epoch)
        scheduler_stage1.step()
        scheduler_select.step()        
        test()
    torch.save(model_stage1,"./preNetdata_stage1")     
    torch.save(model_select,"./preNetdata_select")   
    
    x_list_stage1=np.array(x_list)
    np.save(loss_image_path+'\\x_list_'+plttitle+'.npy',x_list_stage1)     
    loss_list_stage1=np.array(loss_list)
    np.save(loss_image_path+'\\loss_list_'+plttitle+'.npy',loss_list_stage1) 
    makeplt(plttitle);
if (not OnServer):
    printoutput()
print("FINISH!");