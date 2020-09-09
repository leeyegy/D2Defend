

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
from ddid_lee import ddid
from config import  args


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
        print (sigma2)
        res.append(np.transpose(ddid(img,sigma2),[2,0,1]))
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
    for i in range(nb_batch):
        img = np.transpose(img_batch[i].cpu().numpy(),[1,2,0])
        res.append(np.transpose(ddid(img,sigma2),[2,0,1]))
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


def show(img_tensor):
    transforms.ToPILImage()(img_tensor).show()

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


    # check file
    # adv_data_path = os.path.join("data","test_adv_"+str(args.attack_method)+"_"+str(args.epsilon)+".h5")
    # assert os.path.exists(adv_data_path), "not found expected file : "+adv_data_path


    # get data
    test_adv_loader = get_test_adv_loader(args.attack_method,args.epsilon)
    # test_adv_loader = get_test_adv_loader_adaptive(args.attack_method,args.epsilon)


    # load net
    save_dir = "checkpoint"
    file_name = 'wide-resnet-' + str(args.depth) + 'x' + str(args.widen_factor) + '.t7'
    model = torch.load(os.path.join(save_dir, file_name))
    model = model['net']
    model = model.to(device)


    adv_model = torch.load(os.path.join(save_dir,'cifar10_pgd_8_model_120.pth'))
    adv_model = adv_model.to(device)

    time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    nb_epoch = 1


    for epoch in range(nb_epoch):

        #evaluate
        model.eval()
        adv_model.eval()

        clncorrect_nodefence = 0
        correct_defence = 0
        adv_defence = 0
        count = 0
        # for advdata, target,sigma in test_adv_loader:
        for advdata, target in test_adv_loader:

            if count * 50 >= args.test_samples:
                break
            else:
                count += 1
            # advdata, target,sigma = advdata.to(device), target.to(device),sigma.to(device)
            advdata, target = advdata.to(device), target.to(device)

            with torch.no_grad():
                output = model(advdata.float())
                adv_output = adv_model(advdata.float())

            pred = output.max(1, keepdim=True)[1]
            adv_pred = adv_output.max(1, keepdim=True)[1]

            clncorrect_nodefence += pred.eq(target.view_as(pred)).sum().item()
            adv_defence += adv_pred.eq(target.view_as(adv_pred)).sum().item()



            # defence
            # defence_data = ddid_batch(advdata,(args.sigma/255)**2)
            # defence_data = adaptive_ddid_batch(advdata, sigma)
            # with torch.no_grad():
            #     output = model(defence_data.float())
            # pred = output.max(1, keepdim=True)[1]
            # correct_defence += pred.eq(target.view_as(pred)).sum().item()
            # print("handled {} samples ~".format(count * 50))

        # print('\nclean Test set: '
        #       ' no_defence acc: {}/{} ({:.0f}%) defence acc: {}/{} ({:.0f}%) \n'.format(
        #            clncorrect_nodefence, count*50,
        #           100. * clncorrect_nodefence / (count*50),correct_defence,count*50,100. * correct_defence / (count*50)))

        print('\nclean Test set: '
              'no_defence acc: {}/{} ({:.0f}%)   adv_training_pgd_defence acc: {}/{} ({:.0f}%)  \n'.format(
                   clncorrect_nodefence, count*50,
                  100. * clncorrect_nodefence / (count*50), adv_defence, count*50,
                  100. * adv_defence / (count*50)))