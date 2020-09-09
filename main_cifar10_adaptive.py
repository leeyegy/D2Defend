

from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import TRAINED_MODEL_PATH
from advertorch.attacks import CarliniWagnerL2Attack,GradientSignAttack,L2PGDAttack,SpatialTransformAttack,JacobianSaliencyMapAttack,MomentumIterativeAttack
from model_CNN import  ResNet18

import h5py

import numpy as np
import  time

from PIL import Image
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_generator import get_handled_cifar10_train_loader,get_handled_cifar10_test_loader,get_test_adv_loader,get_test_adv_loader_adaptive
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
import os, glob, datetime, time

import sys
from torch.autograd import Variable
sys.path.append("../../")
from networks import *
from defender import *
from config import  args
from time import *
import math
import cv2
from skimage.measure import compare_psnr, compare_ssim

def psnr1(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def psnr2(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim_and_save(com_data,cln_data):
    '''
    :param com_data: [N,C,H,W] | np.array [0,1]
    :param cln_data: [N,C,H,W] | np.array [0,1]
    :return:
    '''
    com_data = ((np.transpose(com_data,[0,2,3,1]))*255).astype(np.float32)
    cln_data = ((np.transpose(cln_data,[0,2,3,1]))*255).astype(np.float32) # only np.float32 is supported
    ssim_list=[]
    psnr_list = []
    for index in range(com_data.shape[0]):
        com_img = com_data[index]
        cln_img = cln_data[index]
        psnr = psnr2(com_img,cln_img)
        psnr_list.append(psnr)
        grayA = cv2.cvtColor(com_img, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(cln_img, cv2.COLOR_BGR2GRAY)
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        ssim_list.append(score)
        print("img index: {} , SSIM: {} PSNR:{}".format(index,score,psnr))
        com_filename = os.path.join("defend_image",str(index)+".png")
        cln_filename = os.path.join("clean_image",str(index)+".png")
        cv2.imwrite(com_filename,com_data[index])
        cv2.imwrite(cln_filename,cln_data[index])
    print("avg ssim:{} , avg psnr:{} ".format(np.asarray(ssim_list).mean(),np.asarray(psnr_list).mean()))


torch.multiprocessing.set_sharing_strategy('file_system')
def adaptive_ddid_batch(img_batch,sigma):
    '''
    :param img_batch: [batch, C,H,W ] | tensor.cuda()
    :param sigma:  [batch,  ]  | tensor.cuda()  NOTICE THAT   (sigma/255)**2 SHOULD BE CALCULATED IN THIS FUNCTION
    :return: [batch,C,H,W] | tensor.cuda()
    '''
    nb_batch = img_batch.size()[0]
    sigma = sigma.cpu().numpy()

    res = []
    # print(nb_batch)
    for i in range(nb_batch):
        img = np.transpose(img_batch[i].cpu().numpy(),[1,2,0])
        sigma2 = (sigma[i]/255)**2
        # print (sigma2)
        res.append(np.transpose(d2defend(img,sigma2),[2,0,1]))
    res = torch.from_numpy(np.asarray(res)).cuda()
    return res

def ddid_batch(img_batch,sigma2):
    '''
    :param img_batch: [batch, C,H,W ] | tensor.cuda()
    :param sigma2:  (xxx/255)**2
    :return: [batch,C,H,W] | tensor.cuda()
    '''
    nb_batch = img_batch.size()[0]
    res = []
    # print(nb_batch)
    for i in range(nb_batch):
        img = np.transpose(img_batch[i].cpu().numpy(),[1,2,0])
        # print("img.shape:{}".format(img.shape))
        # tmp  = ddid(img, sigma2)
        # print("tmp.shape:{}".format(tmp.shape))
        res.append(np.transpose(d2defend(img,sigma2),[2,0,1]))
    res = torch.from_numpy(np.asarray(res)).cuda()
    return res

def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))

# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(args.num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, args.num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, args.num_classes)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, args.num_classes)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name

#print args
def print_setting(args):
    import time
    print(args)
    time.sleep(5)

import glob
import re
import os
def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,"*model_*.pth"))
    epoch = []
    if file_list:
        for file in file_list:
            result = re.findall("model_(.*).pth.*",file)
            if result:
                epoch.append(int(result[0]))
        if epoch:
            return max(epoch)
    return 0



if __name__ == '__main__':
    print_setting(args)

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # get data
    test_adv_loader = get_test_adv_loader_adaptive(args.attack_method,args.epsilon)

    # load net
    save_dir = "checkpoint"
    # file_name = 'wide-resnet-' + str(args.depth) + 'x' + str(args.widen_factor) + '.t7'

    file_name = "cifar10_PGD_8_wideres_model_101.pth"
    wideRes = torch.load(os.path.join(save_dir, file_name))
    # wideRes = wideRes['net']
    wideRes = wideRes.to(device)

    vgg16 = torch.load(os.path.join(save_dir,'cifar10_vgg16_model_299.pth'))
    vgg16 = vgg16.to(device)

    vgg11 = torch.load(os.path.join(save_dir,'cifar10_vgg11_model_199.pth'))
    vgg11 = vgg11.to(device)

    resnet50 = torch.load(os.path.join(save_dir,'cifar10_resnet50_model_199.pth'))
    resnet50 = resnet50.to(device)

    #evaluate
    vgg16.eval()
    wideRes.eval()
    vgg11.eval()
    resnet50.eval()

    correct_wideRes = 0
    correct_vgg16 = 0
    correct_vgg11 = 0
    correct_resnet50 = 0

    count = 0
    for advdata, target,sigma in test_adv_loader:
        if count * 50 >= args.test_samples:
            break
        else:
            count += 1
        # load data
        advdata, target,sigma = advdata.to(device), target.to(device),sigma.to(device)

        # defence
        defence_data = adaptive_ddid_batch(advdata, sigma)
        if args.test_ssim:
            file_path = "data/test.h5"
            if os.path.exists(file_path):
                h5_store = h5py.File(file_path, 'r')
                cln_data = h5_store['data'][:]  
                h5_store.close()
            cln_data = cln_data[0:50]
            ssim_and_save(defence_data.cpu().numpy(),cln_data)
            break

        # test
        with torch.no_grad():
            output_wideRes = wideRes(defence_data.float())
            output_vgg16 = vgg16(defence_data.float())
            output_res50 = resnet50(defence_data.float())
            output_vgg11 = vgg11(defence_data.float())

        pred_wideRes = output_wideRes.max(1, keepdim=True)[1]
        pred_vgg16 = output_vgg16.max(1, keepdim=True)[1]
        pred_res50 = output_res50.max(1, keepdim=True)[1]
        pred_vgg11 = output_vgg11.max(1, keepdim=True)[1]

        correct_wideRes += pred_wideRes.eq(target.view_as(pred_wideRes)).sum().item()
        correct_vgg16 += pred_vgg16.eq(target.view_as(pred_vgg16)).sum().item()
        correct_resnet50 += pred_res50.eq(target.view_as(pred_res50)).sum().item()
        correct_vgg11 += pred_vgg11.eq(target.view_as(pred_vgg11)).sum().item()

        print("handled {} samples ~".format(count * 50))
    print('\nclean Test set: '
          ' wideRes_defence acc: {}/{} ({:.0f}%)  VGG16 defence acc: {}/{} ({:.0f}%)  res50_defence acc: {}/{} ({:.0f}%)  VGG11 defence acc: {}/{} ({:.0f}%)\n'.format(
               correct_wideRes, count*50,
              100. * correct_wideRes / (count*50),correct_vgg16,count*50,100. * correct_vgg16 / (count*50),
        correct_resnet50, count * 50, 100. * correct_resnet50 / (count * 50),
        correct_vgg11, count * 50, 100. * correct_vgg11 / (count * 50)
    ))

