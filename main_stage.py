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

from config import batch_size,output_path,epoch_num,loss_image_path,OnServer,UseF1
from model.model import ETE_stage1,ETE_select,ETE_stage2,label_channel,label_list,make_inverse
from data.loaddata import data_loader_Aug

train_data=data_loader_Aug("train",batch_size,"main_stage");
test_data=data_loader_Aug("test",batch_size,"main_stage");
val_data=data_loader_Aug("val",batch_size,"main_stage");

use_gpu = torch.cuda.is_available()
if OnServer:
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
else:
    import matplotlib;matplotlib.use('TkAgg');
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      

model_stage1=ETE_stage1(device);
model_select=ETE_select(device);
model_stage2=ETE_stage2(device);

bestloss=1000000
bestf1=0
    
def train(epoch):   
    model_stage1.train();     
    model_select.train();     
    model_stage2.train();     
    '''
    part1_time=0;    
    part2_time=0;    
    part3_time=0;         
    prev_time=time.time();             
    '''    
    unloader = transforms.ToPILImage()
    
    for batch_idx,sample in enumerate(train_data.get_loader()):   
        '''
        now_time=time.time();
        part3_time+=now_time-prev_time;        
        prev_time=now_time;
        '''
        if (use_gpu):
            sample['image']=sample['image'].to(device)            
            sample['label']=sample['label'].to(device)      
            sample['image_org']=sample['image_org'].to(device)   
            sample['size'][0]=sample['size'][0].to(device);
            sample['size'][1]=sample['size'][1].to(device);
                     
        optimizer_stage1.zero_grad();                           
        optimizer_select.zero_grad();
        optimizer_stage2.zero_grad();
        
        stage1_label=model_stage1(sample['image'])
        theta=model_select(stage1_label,sample['size'])
        stage2_label=model_stage2(sample['image_org'],theta);

        #parts=[];
        parts_label=[];        
        loss=[]
        for i in range(6):            
            #affine_stage2=F.affine_grid(theta[:,i],(sample['image'].size()[0],3,80,80));
            #parts.append(F.grid_sample(sample['image'],affine_stage2));
            affine_stage2=F.affine_grid(theta[:,i],(sample['image'].size()[0],label_channel[i],80,80));            
            parts_label.append(F.grid_sample(sample['label'][:,label_list[i]],affine_stage2));             
                        
            parts_label[i][:,0]+=0.00001;  
            parts_label[i]=parts_label[i].detach();
            '''
            #print(theta[0][i]);
            image3=unloader(np.uint8(parts_label[i][0][1].cpu().detach().numpy()))           
            plt.imshow(image3)
            plt.show(block=True)                        
            '''
            '''
            print(parts_label[i].size());
            for l1 in range(80):
                for l2 in range(80):
                    print("%.2f"%float(parts_label[i][0][0][l1][l2].data),end=' ')
                print();
            print("FUCK");
            for l1 in range(80):
                for l2 in range(80):
                    print("%.2f"%float(parts_label[i][0][1][l1][l2].data),end=' ')
                print();
            print("FUCK");                                 
            tmp=parts_label[i].argmax(dim=1, keepdim=False);
            print(tmp.size())
            for l1 in range(80):
                for l2 in range(80):
                    print(int(tmp[0][l1][l2].data),end=' ')
                print();
            input("pause")
            flag=False;
            '''                 
            
            loss_tmp=fun.cross_entropy(stage2_label[i],parts_label[i].argmax(dim=1, keepdim=False))                        
            loss.append(loss_tmp);
        if (batch_idx%100==0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(sample['image']), len(train_data.get_loader().dataset),
                100. * batch_idx / len(train_data.get_loader()),np.sum(np.array(loss))))                            
        '''    
        now_time=time.time();
        part1_time+=now_time-prev_time;        
        prev_time=now_time;        
        '''        
        loss=torch.stack(loss)
        loss.backward(torch.ones(6, device=device, requires_grad=False))
        
        optimizer_stage2.step();
        optimizer_select.step();
        optimizer_stage1.step();
        '''
        now_time=time.time();
        part2_time+=now_time-prev_time;        
        prev_time=now_time;
        
        print("batch_idx=",batch_idx);                       
        print("part1_time=",part1_time);     
        print("part2_time=",part2_time);     
        print("part3_time=",part3_time);     
        '''                
def test():
    model_stage1.eval();     
    model_select.eval();     
    model_stage2.eval();   
    global bestloss,bestf1
    test_loss=0    
    hists=[]
    for sample in val_data.get_loader():
        if (use_gpu):
            sample['image']=sample['image'].to(device)            
            sample['label']=sample['label'].to(device)  
            sample['image_org']=sample['image_org'].to(device)      
            sample['size'][0]=sample['size'][0].to(device);
            sample['size'][1]=sample['size'][1].to(device);                  
            
        stage1_label=model_stage1(sample['image'])
        theta=model_select(stage1_label,sample['size'])
        stage2_label=model_stage2(sample['image_org'],theta);
        
        #parts=[];
        parts_label=[];        
        for i in range(6):            
            #affine_stage2=F.affine_grid(theta[:,i],(sample['image'].size()[0],3,80,80));
            #parts.append(F.grid_sample(sample['image'],affine_stage2));
            affine_stage2=F.affine_grid(theta[:,i],(sample['image'].size()[0],label_channel[i],80,80));
            parts_label.append(F.grid_sample(sample['label'][:,label_list[i]],affine_stage2));    
            
            parts_label[i][:,0]+=0.00001;           
            test_loss+=fun.cross_entropy(stage2_label[i],parts_label[i].argmax(dim=1, keepdim=False)).data;    

            '''
            print(parts_label[i].size());
            for l1 in range(80):
                for l2 in range(80):
                    print("%.2f"%float(parts_label[i][0][0][l1][l2].data),end=' ')
                print();
            print("FUCK");
            for l1 in range(80):
                for l2 in range(80):
                    print("%.2f"%float(parts_label[i][0][1][l1][l2].data),end=' ')
                print();
            print("FUCK");            
            tmp=parts_label[i].argmax(dim=1, keepdim=False);
            print(tmp.size())
            for l1 in range(80):
                for l2 in range(80):
                    print(int(tmp[0][l1][l2].data),end=' ')
                print();
            input("pause")
            '''
            
            output_2 = torch.softmax(stage2_label[i], dim=1).argmax(dim=1, keepdim=False)
            output_2=output_2.cpu().clone()
            target_2 = parts_label[i].argmax(dim=1, keepdim=False)
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
            torch.save(model_stage1,"./BestNet_stage1")
            torch.save(model_select,"./BestNet_select")
            torch.save(model_stage2,"./BestNet_stage2")
    else:
        if (test_loss<bestloss):
            bestloss=test_loss
            print("Best data Updata\n");
            torch.save(model_stage1,"./BestNet_stage1")           
            torch.save(model_select,"./BestNet_select")
            torch.save(model_stage2,"./BestNet_stage2")
def printoutput():
    model_stage1=torch.load("./BestNet_stage1",map_location="cpu")
    model_select=torch.load("./BestNet_select",map_location="cpu")
    model_stage2=torch.load("./BestNet_stage2",map_location="cpu")
    model_select.select.change_device(device);
    if (use_gpu):
        model_stage1=model_stage1.to(device)
        model_select=model_select.to(device)
        model_stage2=model_stage2.to(device)    
    unloader = transforms.ToPILImage()
    k=0;
    hists=[]                                 
    for sample in test_data.get_loader():
        if (use_gpu):
            sample['image']=sample['image'].to(device)            
            sample['label']=sample['label'].to(device)
            sample['image_org']=sample['image_org'].to(device)  
            sample['size'][0]=sample['size'][0].to(device);
            sample['size'][1]=sample['size'][1].to(device);
                      
        stage1_label=model_stage1(sample['image'])
        theta=model_select(stage1_label,sample['size'])
        stage2_label=model_stage2(sample['image_org'],theta);
        
        theta_inv=make_inverse(device,theta);
        output=[];#[batch_size,9,s0,s1]
        print("check");         
        for i in range(batch_size):            
            path=output_path+'/'+test_data.get_namelist()[k];   
            if not os.path.exists(path):
                os.makedirs(path);                
            image=sample['image_org'][i].cpu().clone();                
            image =unloader(image)
            image.save(path+'/'+test_data.get_namelist()[k]+'.jpg',quality=100);    
                        
            for j in range(6):
                '''
                print(test_data.get_namelist()[k]);
                print(stage2_label[j][i].argmax(dim=0).size());
                checktmp=stage2_label[j][i].argmax(dim=0);
                for l1 in range(80):
                    for l2 in range(80):
                        print(int(checktmp[l1][l2].data),end=' ')
                    print()   
                print();
                '''
                tmp=torch.zeros(label_channel[j],80,80).scatter_(0,stage2_label[j][i].argmax(dim=0).unsqueeze(0).cpu(),255);
                '''
                checktmp=stage2_label[j][i].argmax(dim=0);
                print(tmp.size());
                for l1 in range(80):
                    for l2 in range(80):
                        print(int(tmp[1][l1][l2].data),end=' ')
                    print()                 
                input('pause')                                
                '''
                for l in range(1,label_channel[j]):                    
                    image3=unloader(np.uint8(tmp[l].cpu()));
                    image3.save(path+'/'+'stage2_'+test_data.get_namelist()[k]+'lbl0'+str(j)+str(l)+'.jpg',quality=100);  
            
            final_label=[];#[8*[1,1,s0,s1]]            
            for j in range(6):
                affine_stage2_inv=F.affine_grid(theta_inv[i,j].unsqueeze(0),(1,1,1024,1024));
                stage2_label[j][i]=torch.zeros(label_channel[j],80,80).scatter_(0,stage2_label[j][i].argmax(dim=0).unsqueeze(0).cpu(),255);
                for l in range(1,label_channel[j]):     
                    labeltmp=stage2_label[j][i][l];                                        
                    labeltmp=labeltmp.unsqueeze(0).unsqueeze(0);                                        
                    final_label.append(F.grid_sample(labeltmp,affine_stage2_inv));        
            final_label=torch.cat(final_label, dim=1)#[8*[1,1,s0,s1]]->[1,8,s0,s1]   
            final_label=torch.squeeze(final_label,dim=0);#[1,8,s0,s1]->[8,s0,s1]
            
            bg=1-torch.sum(final_label, dim=0, keepdim=True);            
            final_label=torch.cat([bg, final_label],dim=0);#[9,s0,s1]
            final_label=torch.softmax(final_label,dim=0);  
            output.append(final_label);
                        
            image=output[i].cpu().clone();                         
            image = torch.softmax(image, dim=0).argmax(dim=0, keepdim=False)                                       
            image=image.unsqueeze(dim=0);                                    
            image=torch.zeros(9,1024,1024).scatter_(0, image, 255)                
                        
            for j in range(9):                                
                image3=unloader(np.uint8(image[j].numpy()))
                #image3=TF.resize(img=image3,size=sample[i]["size"],interpolation=Image.NEAREST)                      
                image3.save(path+'/'+test_data.get_namelist()[k]+'lbl0'+str(j)+'.jpg',quality=100);            
            k+=1
            if (k>=test_data.get_len()):break    
        output=torch.stack(output);
        target2=sample['label'].cpu().clone();             
        target2=torch.softmax(target2,dim=1).argmax(dim=1, keepdim=False);  
        output2=output.cpu().clone(); 
        output2=output2.argmax(dim=1, keepdim=False);          
        hist = np.bincount(9 * target2.reshape([-1]) + output2.reshape([-1]),minlength=81).reshape(9, 9)
        hists.append(hist);
        if (k>=test_data.get_len()):break        
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

UseHerit=False;    
if UseHerit:    
    model_stage1=torch.load("./Netdata_stage1")
    model_select=torch.load("./Netdata_select")
    model_stage2=torch.load("./Netdata_stage2")
else:
    model_stage1=torch.load("./preBestNet_stage1",map_location=device)
    model_select=torch.load("./preBestNet_select",map_location=device)

print("use_gpu=",use_gpu)
if (use_gpu):    
    model_stage1=model_stage1.to(device)
    model_select=model_select.to(device)
    model_stage2=model_stage2.to(device)
model_select.select.change_device(device);

optimizer_stage1=optim.Adam(model_stage1.parameters(),lr=0.000) 
optimizer_select=optim.Adam(model_select.parameters(),lr=0.000) 
optimizer_stage2=optim.Adam(model_stage2.parameters(),lr=0.001) 
scheduler_stage1=optim.lr_scheduler.StepLR(optimizer_stage1, step_size=5, gamma=0.5)       
scheduler_select=optim.lr_scheduler.StepLR(optimizer_select, step_size=5, gamma=0.5)       
scheduler_stage2=optim.lr_scheduler.StepLR(optimizer_stage2, step_size=5, gamma=0.5)       

plttitle="EndtoEndModel"
Training=False;
if Training:
    for epoch in range(epoch_num):
        x_list.append(epoch);
        train(epoch)
        scheduler_stage1.step()
        scheduler_select.step()
        scheduler_stage2.step()
        test()
    torch.save(model_stage1,"./Netdata_stage1") 
    torch.save(model_select,"./Netdata_select")
    torch.save(model_stage2,"./Netdata_stage2")
    x_list_mainstage=np.array(x_list)
    np.save(loss_image_path+'\\x_list_'+plttitle+'.npy',x_list_mainstage) 
    f1_list_mainstage=np.array(f1_list)
    np.save(loss_image_path+'\\f1_list_'+plttitle+'.npy',f1_list_mainstage) 
    loss_list_mainstage=np.array(loss_list)
    np.save(loss_image_path+'\\loss_list_'+plttitle+'.npy',loss_list_mainstage) 
    makeplt("plttitle");
if (not OnServer):
    printoutput()  