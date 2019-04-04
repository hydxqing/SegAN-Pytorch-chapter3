'''
ReNet-Keras
Code written by: Xiaoqing Liu
If you use significant portions of this code or the ideas from our paper, please cite it :)
'''
from __future__ import print_function
import argparse
import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from net import NetS, NetC

from torchvision.transforms import Compose,Resize,ToTensor,ToPILImage
from transform import ToLabel
from torch.utils.data import DataLoader
from dataset import train,test

input_transform = Compose([
    Resize((400,400)),
    #CenterCrop(256),
    ToTensor(),
    #Normalize([112.65,112.65,112.65],[32.43,32.43,32.43])
    #Normalize([.485, .456, .406], [.229, .224, .225]),
])
target_transform = Compose([
    Resize((400,400)),
    #CenterCrop(324),
    ToLabel(),
    #Relabel(255, 1),
])

# Training settings
parser = argparse.ArgumentParser(description='Example')
#parser.add_argument('--batchSize', type=int, default=36, help='training batch size')
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning Rate. Default=0.02')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use, for now it only supports one GPU')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--decay', type=float, default=0.5, help='Learning rate decay. default=0.5')
parser.add_argument('--cuda', default='true', help='using GPU or not')
parser.add_argument('--seed', type=int, default=666, help='random seed to use. Default=1111')
#parser.add_argument('--outpath', default='./outputs', help='folder to output images and model checkpoints')
opt = parser.parse_args()

print(opt)


#try:
   # os.makedirs(opt.outpath)
#except OSError:
   # pass

# custom weights initialization called on NetS and NetC
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def dice_loss(input,target):
    num=input*target
    num=torch.sum(num,dim=2)
    num=torch.sum(num,dim=2)

    den1=input*input
    den1=torch.sum(den1,dim=2)
    den1=torch.sum(den1,dim=2)

    den2=target*target
    den2=torch.sum(den2,dim=2)
    den2=torch.sum(den2,dim=2)

    dice=2*(num/(den1+den2))

    dice_total=1-1*torch.sum(dice)/dice.size(0)#divide by batchsize

    return dice_total


cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

cudnn.benchmark = True
print('===> Building model')
NetS = NetS(ngpu = opt.ngpu)
# NetS.apply(weights_init)
#print(NetS)
NetC = NetC(ngpu = opt.ngpu)
# NetC.apply(weights_init)
#print(NetC)

if cuda:
    NetS = NetS.cuda()
    NetC = NetC.cuda()
    # criterion = criterion.cuda()

# setup optimizer
lr = opt.lr
decay = opt.decay
optimizerG = optim.Adam(NetS.parameters(), lr=lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(NetC.parameters(), lr=lr, betas=(opt.beta1, 0.999))
# load training data
#dataloader = loader(Dataset('./'),opt.batchSize)
dataloader = DataLoader(sonar(input_transform, target_transform),num_workers=1, batch_size=2, shuffle=True)
# load testing data
#dataloader_val = loader(Dataset_val('./'), 36)

log_path = './save/log.txt'
savedir = './save/model/'
max_iou = 0
NetS.train()
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 1):
        #train C
        NetC.zero_grad()
        input, label = Variable(data[0]), Variable(data[1])
        if cuda:
            input = input.cuda()
            target = label.cuda()
        target = target.type(torch.FloatTensor)
        target = target.cuda()
        output = NetS(input)
        #output = F.sigmoid(output*k)
        output = F.sigmoid(output)
        output = output.detach()
        output_masked = input.clone()
        input_mask = input.clone()
        #print(input_mask.size())
        #print(output.size())
        output_masked = (input_mask[:,0,:,:].unsqueeze(1)) * output
        #print(output_masked.size())
        #detach G from the network
       # output_masked=[]
        #for d in range(3):
            #print((input_mask[:,0,:,:]).size())
        output_mask_0 = (input_mask[:,0,:,:].squeeze() * output.squeeze())
        output_mask_1 = (input_mask[:,1,:,:].squeeze() * output.squeeze())
        output_mask_2 = (input_mask[:,2,:,:].squeeze() * output.squeeze())
           # output_mask = output_mask.unsqueeze(1)
        output_masked = torch.stack([output_mask_0,output_mask_1,output_mask_2],1)

        if cuda:
            output_masked = output_masked.cuda()
       # print(output_masked.size())
        result = NetC(output_masked)
        target_masked = input.clone()
        ##for d in range(3):
            #target_masked[:,d,:,:] = input_mask[:,d,:,:].unsqueeze(1) * target
        target_mask_0 = (input_mask[:,0,:,:].squeeze() * target.squeeze())
        target_mask_1 = (input_mask[:,1,:,:].squeeze() * target.squeeze())
        target_mask_2 = (input_mask[:,2,:,:].squeeze() * target.squeeze())
           # output_mask = output_mask.unsqueeze(1)
        target_masked = torch.stack([target_mask_0,target_mask_1,target_mask_2],1)

            #output_masked[:,d,:,:] = (input_mask[:,d,:,:].unsqueeze(1)) * output
        #print(target_masked.size())

        if cuda:
            target_masked = target_masked.cuda()
        target_D = NetC(target_masked)
        loss_D = - torch.mean(torch.abs(result - target_D))
        loss_D.backward()
        optimizerD.step()
        #clip parameters in D
        for p in NetC.parameters():
            p.data.clamp_(-0.05, 0.05)
        #train G
        NetS.zero_grad()
        output = NetS(input)
        output = F.sigmoid(output)

        #for d in range(3):
           # output_masked[:,d,:,:] = input_mask[:,d,:,:].unsqueeze(1) * output
        output_mask_0 = (input_mask[:,0,:,:].squeeze() * output.squeeze())
        output_mask_1 = (input_mask[:,1,:,:].squeeze() * output.squeeze())
        output_mask_2 = (input_mask[:,2,:,:].squeeze() * output.squeeze())
           # output_mask = output_mask.unsqueeze(1)
        output_masked = torch.stack([output_mask_0,output_mask_1,output_mask_2],1)

        if cuda:
            output_masked = output_masked.cuda()
        result = NetC(output_masked)
        #for d in range(3):
         #   target_masked[:,d,:,:] = input_mask[:,d,:,:].unsqueeze(1) * target
        target_mask_0 = (input_mask[:,0,:,:].squeeze() * target.squeeze())
        target_mask_1 = (input_mask[:,1,:,:].squeeze() * target.squeeze())
        target_mask_2 = (input_mask[:,2,:,:].squeeze() * target.squeeze())
           # output_mask = output_mask.unsqueeze(1)
        target_masked = torch.stack([target_mask_0,target_mask_1,target_mask_2],1)

        if cuda:
            target_masked = target_masked.cuda()
        target_G = NetC(target_masked)
        loss_dice = dice_loss(output,target)
        loss_G = torch.mean(torch.abs(result - target_G))
        loss_G_joint = torch.mean(torch.abs(result - target_G)) + loss_dice
        loss_G_joint.backward()
        optimizerG.step()
        #print('print!')
        if i % 50 == 0:
      
            print('epoch:%f'%epoch,'step:%f'%i,'G_loss:%f'%loss_G.data[0],'D_loss:%f'%loss_D.data[0])
            with open(log_path, "a") as myfile:
                myfile.write("\n%d\t%d\t%.4f\t%.4f" % (epoch,i, loss_G.data[0],loss_D.data[0] ))

    filename = 'main-'+'Segan'+'-epoch-'+str(epoch)+'.pth'
    torch.save(NetS.state_dict(), savedir+filename)


    if epoch % 10 == 0:
        lr = lr*decay
        if lr <= 0.00000001:
            lr = 0.00000001
        print('Learning Rate: {:.6f}'.format(lr))
        # print('K: {:.4f}'.format(k))
        #print('Max mIoU: {:.4f}'.format(max_iou))
        optimizerG = optim.Adam(NetS.parameters(), lr=lr, betas=(opt.beta1, 0.999))
        optimizerD = optim.Adam(NetC.parameters(), lr=lr, betas=(opt.beta1, 0.999))
