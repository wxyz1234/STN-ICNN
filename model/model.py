import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets,transforms
from torchvision.transforms import functional as TF
from PIL import Image
from config import batch_size
from data.Helentransform import GaussianNoise,RandomAffine,Resize,ToTensor,ToPILImage,HorizontalFlip,DoNothing
import time,datetime
import numpy as np

label_channel=[2,2,2,2,2,4];
label_list=[[0,1],[0,2],[0,3],[0,4],[0,5],[0,6,7,8]];
'''
def calc_centroid(tensor):#input [batch_size,9,128,128]->output [batch_size,9,2]
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
'''
def calc_centroid(tensor):#input [batch_size,9,128,128]->output [batch_size,9,2]
    input=tensor.float()+1e-10;
    n, l, h, w = input.shape    
    index_y=torch.from_numpy(np.arange(h)).float().to(tensor.device);
    index_x=torch.from_numpy(np.arange(w)).float().to(tensor.device);    
    center_x=input.sum(2)*index_x.view(1,1,-1);
    center_x=center_x.sum(2,keepdim=True);
    center_xchu=input.sum([2, 3]).view(n, l, 1);
    center_y=input.sum(3)*index_y.view(1,1,-1);
    center_y=center_y.sum(2,keepdim=True);
    center_ychu=input.sum([2, 3]).view(n, l, 1);                    
    
    center_x[:,6]=center_x[:,6]+center_x[:,7]+center_x[:,8];
    center_xchu[:,6]=center_xchu[:,6]+center_xchu[:,7]+center_xchu[:,8];
    center_y[:,6]=center_y[:,6]+center_y[:,7]+center_y[:,8];
    center_ychu[:,6]=center_ychu[:,6]+center_ychu[:,7]+center_ychu[:,8];    
    points=[];
    for i in range(1,7):
        points.append(torch.cat([center_x[:,i]*1.0/center_xchu[:,i], center_y[:,i]*1.0/center_ychu[:,i]],1));        
    points=torch.stack(points,dim=1);    
    return points;

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x

def make_inverse(device,theta):                
    ones = torch.tensor([[0.,0.,1.]]).repeat(batch_size,6,1,1).to(device)
    theta = torch.cat([theta, ones], dim=2).to(device)            
    theta = torch.inverse(theta)    
    theta = theta[:,:,0:2];    
    theta[:,:,0,2]+=2/81;
    theta[:,:,1,2]+=2/81;
    return theta;

class iCNN_Node(torch.nn.Module):    
    def __init__(self,in_channel,out_channel,in_size,dorelu=True):
        super(iCNN_Node, self).__init__();        
        #self.upsample=nn.UpsamplingNearest2d(scale_factor=2);
        #self.downsample=nn.MaxPool2d(2);  
        self.upsample=Interpolate(size=(in_size,in_size),mode='nearest');        
        self.downsample=nn.MaxPool2d(kernel_size=3, stride=2, padding=1);
        
        self.in_channel=in_channel;
        self.out_channel=out_channel;
        self.Conv=nn.Conv2d(in_channels=self.in_channel,out_channels=self.out_channel,kernel_size=5,padding=2);
        self.bn=nn.BatchNorm2d(out_channel);  
        self.dorelu=dorelu;        
    def set_conv(self,kernel_size,stride,padding):
        self.Conv=nn.Conv2d(in_channels=self.in_channel,out_channels=self.out_channel,kernel_size=kernel_size,stride=stride,padding=padding);
    def forward(self,x,x_pre=None,x_post=None):        
        x_list=[];        
        if (x_pre!=None):
            x_pre=self.downsample(x_pre);
            x_list.append(x_pre);
        x_list.append(x);
        if (x_post!=None):            
            x_post=self.upsample(x_post);
            x_list.append(x_post);        
        x_in=torch.cat(x_list,dim=1);             
        x_out=self.Conv(x_in);
        if self.dorelu:
            x_out=F.relu(self.bn(x_out));        
        return x_out;
class iCNN_Cell(torch.nn.Module):    
    def __init__(self,in_size_list):
        super(iCNN_Cell, self).__init__()
        self.in_size_list=in_size_list;
        self.Node_list=nn.ModuleList([]);
        self.Node_list.append(iCNN_Node(24,8,in_size_list[0]));
        self.Node_list.append(iCNN_Node(48,16,in_size_list[1]));
        self.Node_list.append(iCNN_Node(72,24,in_size_list[2]));
        self.Node_list.append(iCNN_Node(56,32,in_size_list[3]));
    def forward(self,in_channels):          
        out_channels=[];
        out_channels.append(self.Node_list[0].forward(in_channels[0],None,in_channels[1]));
        out_channels.append(self.Node_list[1].forward(in_channels[1],in_channels[0],in_channels[2]));
        out_channels.append(self.Node_list[2].forward(in_channels[2],in_channels[1],in_channels[3]));
        out_channels.append(self.Node_list[3].forward(in_channels[3],in_channels[2],None));
        return out_channels;
class iCNN_FaceModel(torch.nn.Module):
    def __init__(self,in_size,out_num):
        super(iCNN_FaceModel, self).__init__();
        self.in_size=in_size;
        self.in_size_list=[in_size];
        j=in_size;
        for i in range(3):
            j=j//2+(j&1);
            self.in_size_list.append(j);
        self.out_num=out_num;
        self.Cell_list=nn.ModuleList([iCNN_Cell(self.in_size_list) for _ in range(3)]);
        self.in_Node_list=nn.ModuleList([]);
        self.bninput=nn.BatchNorm2d(3);
        #self.downsample_image=nn.AvgPool2d(2);  
        self.downsample_image=nn.MaxPool2d(kernel_size=3, stride=2, padding=1);
        
        self.Node_list_st=nn.ModuleList([]);
        self.Node_list_st.append(iCNN_Node(3,8,self.in_size_list[0]));
        self.Node_list_st.append(iCNN_Node(3,16,self.in_size_list[1]));
        self.Node_list_st.append(iCNN_Node(3,24,self.in_size_list[2]));
        self.Node_list_st.append(iCNN_Node(3,32,self.in_size_list[3]));
        self.Node_list_ed=nn.ModuleList([]);    
        self.Node_list_ed.append(iCNN_Node(24,8+2*self.out_num,self.in_size_list[0]));
        self.Node_list_ed.append(iCNN_Node(40,16,self.in_size_list[1]));
        self.Node_list_ed.append(iCNN_Node(56,24,self.in_size_list[2]));        
        self.Node_ed=iCNN_Node(8+2*self.out_num,self.out_num,self.in_size_list[0],dorelu=False);
    def forward(self,in_image):           
        x_list=[];    
        in_image=self.bninput(in_image);
        for i in range(4):
            x_list.append(in_image);
            in_image=self.downsample_image(in_image);              
            assert x_list[i].size()[2]==self.in_size_list[i],print(x_list[i].size()[2],' ',self.in_size_list[i]);
        for i in range(4):
            x_list[i]=self.Node_list_st[i](x_list[i]);          
        for i in range(3):
            x_list=self.Cell_list[i](x_list);        
        for i in range(2,-1,-1):
            x_list[i]=self.Node_list_ed[i](x_list[i],None,x_list[i+1]);        
        x_out=self.Node_ed(x_list[0],None,None);                        
        return x_out;
'''
class SelectNetWork(torch.nn.Module):
    def __init__(self,in_size,device):
        super(SelectNetWork, self).__init__();        
        self.in_size=in_size;
        self.device=device;
        self.pooling=nn.AvgPool2d(kernel_size=3,stride=2,padding=1);
        self.Node_list=nn.ModuleList([]);
        self.Node_list.append(iCNN_Node(1,64));
        self.Node_list.append(iCNN_Node(64,64));
        self.Node_list.append(iCNN_Node(64,128));
        self.Node_list.append(iCNN_Node(128,128));
        self.Node_list.append(iCNN_Node(128,256));
        self.Node_list.append(iCNN_Node(256,256));
        self.Node_list.append(iCNN_Node(256,512));
        self.Node_list.append(iCNN_Node(512,512));
        for i in range(len(self.Node_list)):
            self.Node_list[i].set_conv(3,1,1);
        self.connected=nn.Linear(in_features = 512*8*8, out_features = 36);             
    def forward(self,x_in):        
        for i in range(8):
            x_in=self.Node_list[i](x_in);
            if i%2==1:
                x_in=self.pooling(x_in);            
        x_in=x_in.view(batch_size,-1);        
        x_out=self.connected(x_in);        
        x_out=F.tanh(x_out);        
        x_out=x_out.reshape(batch_size,6,2,3);        
        activate_tensor = torch.tensor([[[1., 0., 1.],[0., 1., 1.]]], device=self.device,requires_grad=False).repeat((batch_size,6,1,1))
        theta = x_out * activate_tensor;
        return theta;
'''
class SelectNetWork(torch.nn.Module):
    def __init__(self,in_size,device):
        super(SelectNetWork, self).__init__();        
        self.in_size=in_size;
        self.device=device;
        self.localize_net=nn.Sequential(
            nn.Conv2d(9, 6, kernel_size=3, stride=2, padding=1),  # 6 x 64 x 64
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6 x 64 x 64
            nn.BatchNorm2d(6),
            nn.ReLU(),

            nn.Conv2d(6, 6, kernel_size=3, stride=2, padding=1),  # 6 x 32 x 32
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6 x 32 x 32
            nn.BatchNorm2d(6),
            nn.ReLU(),

            nn.Conv2d(6, 6, kernel_size=3, stride=2, padding=1),  # 6 x 16 x 16
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6 x 16 x 16
            nn.BatchNorm2d(6),
            nn.ReLU(),

            nn.Conv2d(6, 6, kernel_size=3, stride=2, padding=1),  # 6 x 8 x 8
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6 x 8 x 8
            nn.BatchNorm2d(6),
            nn.ReLU(),

            nn.Conv2d(6, 6, kernel_size=3, stride=2, padding=1),  # 6 x 4 x 4
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6 x 4 x 4
            nn.BatchNorm2d(6),

            nn.Conv2d(6, 6, kernel_size=[3, 2], stride=2, padding=1),  # 6 x 2 x 3
            nn.BatchNorm2d(6),
            nn.ReLU(),

            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6 x 2 x 3
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6 x 2 x 3
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6 x 2 x 3
            nn.BatchNorm2d(6),
            nn.ReLU(),

            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),  # 6 x 2 x 3
            nn.Tanh()
            );
        self.model_res = torchvision.models.resnet18(pretrained=False)
        self.model_res.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.model_res.fc.in_features
        self.model_res.fc = nn.Linear(num_ftrs, 36)  # 6 x 2 x 3
    def change_device(self,device):
        self.device=device;
    def forward(self,x_in): 
        #x_out=self.localize_net(x_in);   
        x_out=self.model_res(x_in).view(-1, 6, 2, 3);        
        activate_tensor = torch.tensor([[[1., 0., 1.],[0., 1., 1.]]], device=self.device,requires_grad=False).repeat((x_in.size()[0],6,1,1)).detach()
        theta = x_out * activate_tensor;
        return theta;

class ETE_stage1(torch.nn.Module):
    def __init__(self,device):
        super(ETE_stage1,self).__init__();                
        self.device=device;
        self.stage1=iCNN_FaceModel(128,9);
    def forward(self,image_in):            
        stage1_label=self.stage1(image_in);#[batch_size,9,128,128]     
        return stage1_label;  
    
class ETE_select(torch.nn.Module):
    def __init__(self,device):
        super(ETE_select,self).__init__();                
        self.device=device;        
        self.select=SelectNetWork(128,device);        
    def forward(self,stage1_label,sample_size):       
        stage1_label=torch.softmax(stage1_label,dim=1);
        theta=self.select(stage1_label);#[batch_size,6,2,3]  
        '''         
        for i in range(6):
            tmp=(theta[:,i,0,2]+1)/2;
            tmp=tmp*(sample_size[0]-1.0);
            theta[:,i,0,2]=-1+2*tmp/(1024.0-1.0);
            tmp=(theta[:,i,1,2]+1)/2;
            tmp=tmp*(sample_size[1]-1.0);
            theta[:,i,1,2]=-1+2*tmp/(1024.0-1.0);                        
        '''
        return theta;
    
class ETE_stage2(torch.nn.Module):
    def __init__(self,device):
        super(ETE_stage2,self).__init__();                
        self.device=device;
        self.stage2=nn.ModuleList([]);
        self.stage2.append(iCNN_FaceModel(81,2));
        #self.stage2.append(iCNN_FaceModel(81,2));
        self.stage2.append(iCNN_FaceModel(81,2));
        #self.stage2.append(iCNN_FaceModel(81,2));
        self.stage2.append(iCNN_FaceModel(81,2));
        self.stage2.append(iCNN_FaceModel(81,4));                 
    def forward(self,image_in,theta):                   
        stage2_label=[];#[6*[batch_size,2/4,81,81]]
        if (theta!=None):
            image_stage2=[]#[6*[batch_size,3,81,81]]
            for i in range(6):
                affine_stage2=F.affine_grid(theta[:,i],(image_in.size()[0],3,81,81),align_corners=True);
                image_stage2.append(F.grid_sample(image_in,affine_stage2,align_corners=True));                      
        else:
            image_stage2=image_in;
        '''
        for i in range(6):
            stage2_label.append(self.stage2[i](image_stage2[i]));
        '''
        stage2_label.append(self.stage2[0](image_stage2[0]));
        stage2_label.append(torch.flip(self.stage2[0](torch.flip(image_stage2[1],[3])),[3]));
        stage2_label.append(self.stage2[1](image_stage2[2]));
        stage2_label.append(torch.flip(self.stage2[1](torch.flip(image_stage2[3],[3])),[3]));
        stage2_label.append(self.stage2[2](image_stage2[4]));
        stage2_label.append(self.stage2[3](image_stage2[5]));        
        
        return stage2_label;
class EndtoEndModel(torch.nn.Module):
    def __init__(self,device):
        super(EndtoEndModel,self).__init__();                
        self.device=device;
        self.stage1=iCNN_FaceModel(128,9);
        self.select=SelectNetWork(128,device);
        self.stage2=nn.ModuleList([]);
        self.stage2.append(iCNN_FaceModel(81,2));
        self.stage2.append(iCNN_FaceModel(81,2));
        self.stage2.append(iCNN_FaceModel(81,2));
        self.stage2.append(iCNN_FaceModel(81,2));
        self.stage2.append(iCNN_FaceModel(81,2));
        self.stage2.append(iCNN_FaceModel(81,4));                 
        
    def forward(self,image_in,image_org):           
        '''
        part1_time=0;    
        part2_time=0;    
        part3_time=0;         
        prev_time=time.time();                        
        '''      
        stage1_label=self.stage1(image_in);#[batch_size,9,128,128]    
        #stage1_label=stage1_label.argmax(dim=1, keepdim=True).float();            
        theta=self.select(stage1_label);#[batch_size,6,2,3]          
        '''
        now_time=time.time();
        part1_time+=now_time-prev_time;        
        prev_time=now_time;                       
        '''
        image_stage2=[]#[6*[batch_size,3,81,81]]
        for i in range(6):
            affine_stage2=F.affine_grid(theta[:,i],(batch_size,3,81,81));
            image_stage2.append(F.grid_sample(image_org,affine_stage2));      
            
        stage2_label=[];#[8*[batch_size,81,81]]
        for i in range(5):
            stage2_label.append(self.stage2[i](image_stage2[i])[:,1].unsqueeze(1));        
        mouth_label=self.stage2[5](image_stage2[5]);
        stage2_label.append(mouth_label[:,1].unsqueeze(1));
        stage2_label.append(mouth_label[:,2].unsqueeze(1));
        stage2_label.append(mouth_label[:,3].unsqueeze(1));                
                    
        theta_inv=make_inverse(self.device,theta)#[batch_size,6,2,3]        
        theta_list=[];
        for i in range(5):
            theta_list.append(theta_inv[:,i].unsqueeze(1));
        for i in range(3):
            theta_list.append(theta_inv[:,5].unsqueeze(1));
        theta_inv2=torch.cat(theta_list,dim=1);#[batch_size,6,2,3]->[batch_size,8,2,3]        
        '''
        now_time=time.time();
        part2_time+=now_time-prev_time;        
        prev_time=now_time;
        '''
        final_label=[];#[8*[batch_size,1,s0,s1]]
        for i in range(8):
            affine_stage2_inv=F.affine_grid(theta_inv2[:,i],(batch_size,1,1024,1024));
            final_label.append(F.grid_sample(stage2_label[i],affine_stage2_inv));        
        final_label=torch.cat(final_label, dim=1)#[8*[batch_size,1,s0,s1]]->[batch_size,8,s0,s1]        
        
        final_list=[];        
        for i in range(batch_size):
            bg=1-torch.sum(final_label[i], dim=0, keepdim=True);            
            final_list.append(torch.cat([bg, final_label[i]],dim=0).unsqueeze(0));            
        final_list=torch.cat(final_list,dim=0);#[batch_size,9,s0,s1]    
        final_list=torch.softmax(final_list,dim=1);        
        '''
        now_time=time.time();
        part3_time+=now_time-prev_time;        
        prev_time=now_time;
        
        print("model part1_time=",part1_time);     
        print("model part2_time=",part2_time);     
        print("model part3_time=",part3_time);     
        '''
        return final_list;