import numpy as np
import torch

import os 
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.pylab import array

from PIL import Image
from argparse import ArgumentParser

import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.autograd import Variable
from torchvision.transforms import Compose, CenterCrop, Normalize,Scale
from torchvision.transforms import ToTensor, ToPILImage

from transform import ToLabel
from net import NetS

import time
import cv2
EXTENSIONS = ['.jpg', '.png']

NUM_CHANNELS = 1
NUM_CLASSES = 2


input_transform = Compose([
    Scale((224,224)),
    ToTensor(),
])
target_transform = Compose([
    Scale((224,224)),
    ToLabel(),
])

def infer(model):
   # print 'ok'
    label_np=array(Image.open('./test/Labels/333.png'))
    #label_np = cv2.cvtColor(label_np, cv2.COLOR_BGR2GRAY)
    img=Image.open('./test/Images/333.png')
    img_n=array(img)
    
    img_np=np.resize(img_n,(400,400,3))

    outputs = model(Variable(torch.from_numpy(img_np[np.newaxis, :].transpose(0,3,1,2)).float(),volatile = True).cuda())
    output = F.sigmoid(outputs)
    outputs = outputs.cpu()
    outputs=outputs.data.numpy()
    outputs = np.argmax(outputs,axis = 1)
    mask=np.array(outputs[0,:])


    fig,ax=plt.subplots(1,3)
    ax[0].imshow(img_np,cmap='gray')
    ax[1].imshow(mask)
    ax[2].imshow(label_np)
    plt.show()
    return 0

   
def midfunc():
    print 'load Model....\n'

    Net = NetS(ngpu=1)
    model='./save/model/main-Segan-epoch-26.pth'

    #print list(model.children())
    Net.load_state_dict(torch.load(model))
    
    Net = Net.cuda()
    print 'Done.\n'

    print 'compute output....\n'
    t1=time.time()
    infer(Net)
    print 'time:',time.time()-t1,'\n'
    return 0

if __name__ =='__main__':

    midfunc()
