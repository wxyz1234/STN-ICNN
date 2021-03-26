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
import sys;

def make_inverse(theta):                
    ones = torch.tensor([[0.,0.,1.]]);
    theta = torch.cat([theta, ones], dim=0);
    theta = torch.inverse(theta)    
    theta = theta[0:2]; 
    theta[0,2]+=2/81;
    theta[1,2]+=2/81;       
    return theta;

def makeimage(W,H):
    #image=Image.new('RGB', (W,H), (0,0,0));    
    #print(image.getpixel((W-1,H-1)));sys.exit(0);
    image = np.zeros((2,W,H));
    for i in range(W):
        for j in range(H):
            image[:,i,j]=(i,j);
    #print(image.shape);    
    return torch.from_numpy(image);
def check(x1,x2,out):
    points[1]=x1;
    points[0]=x2;
    if out:
        print(points[1],points[0]);    
    global image,theta;    
    theta[0,0]=81.0/W;
    theta[1,1]=81.0/H;
    theta[0,2]=-1+2*(points[0]+1.0)/W;
    theta[1,2]=-1+2*(points[1]+1.0)/H; 
    #theta=theta.double();
    image=image.float();
    affine_stage2=F.affine_grid(theta.unsqueeze(0),(1,2,81,81),align_corners=True);  
    global image3;
    image3=F.grid_sample(image.unsqueeze(0),affine_stage2,align_corners=True);  
    image3=image3.squeeze(0);
    if out:
        print('('+str(int(image3[0,40,40].data))+','+str(int(image3[1,40,40].data))+')');
    if (int(image3[0,40,40].data)!=points[1] or int(image3[1,40,40].data)!=points[0]):
        print("check1 FUCK ",points[1],' ',points[0],' and ',int(image3[0,40,40].data),' ',int(image3[1,40,40].data));
        if (not out):
            input("wait");

def check2(x1,x2,out):
    global image_inv,theta;
    theta_inv=make_inverse(theta);
    points[1]=x1;
    points[0]=x2;
    affine_stage2=F.affine_grid(theta_inv.unsqueeze(0),(1,2,1024,1024),align_corners=True);  
    global image3;
    image_inv=image_inv.float();
    image3=F.grid_sample(image_inv.unsqueeze(0),affine_stage2,align_corners=True);  
    image3=image3.squeeze(0);
    if out:
        print('('+str(int(image3[0,x1,x2].data))+','+str(int(image3[1,x1,x2].data))+')');
    if (int(image3[0,x1,x2].data)!=40 or int(image3[1,x1,x2].data)!=40):
        print("check2 FUCK ",40,' ',40,' and ',int(image3[0,x1,x2].data),' ',int(image3[1,x1,x2].data));
        if (not out):
            input("wait");

W=1024;
H=1024;
image=makeimage(H,W);
image_inv=makeimage(81,81);
theta=torch.zeros(2,3);
points=[];
points.append(0);
points.append(0);
cx=100;
cy=1000;
'''
for i in range(W):
    for j in range(H):        
        check(i,j,False);
    if (i%50==0):print(i,' finish');
'''

#check(cx,cy,True);
'''
#print(image3.size());
f=open('./check.txt','w');
for i in range(81):
    for j in range(81):
        print('('+str(int(image3[0,i,j].data))+','+str(int(image3[1,i,j].data))+')',end=' ',file=f);
    print(file=f)
f.close();
'''

for i in range(423,W):
    for j in range(H):        
        check(i,j,False);
        check2(i,j,False);
    if (i%1==0):print(i,' finish');
    
#check2(cx,cy,True);
#print(image3.size());
f=open('./check.txt','w');
for i in range(max(0,cx-40),min(1024,cx+41)):
    for j in range(max(0,cy-40),min(1024,cy+41)):
        print('('+str(int(image3[0,i,j].data))+','+str(int(image3[1,i,j].data))+')',end=' ',file=f);
    print(file=f)
f.close();