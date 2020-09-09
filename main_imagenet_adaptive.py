

from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets,models,transforms


from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import TRAINED_MODEL_PATH
from advertorch.attacks import CarliniWagnerL2Attack,GradientSignAttack,L2PGDAttack,SpatialTransformAttack,JacobianSaliencyMapAttack,MomentumIterativeAttack



import numpy as np
import  time

from PIL import Image
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
import os, glob, datetime, time

import sys
from torch.autograd import Variable
from ddid_lee import ddid
from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack,GradientSignAttack,L2PGDAttack,SpatialTransformAttack,JacobianSaliencyMapAttack,MomentumIterativeAttack

from config import args

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
    print(nb_batch)
    for i in range(nb_batch):
        img = np.transpose(img_batch[i].cpu().numpy(),[1,2,0])
        # print("img.shape:{}".format(img.shape))
        # tmp  = ddid(img, sigma2)
        # print("tmp.shape:{}".format(tmp.shape))
        res.append(np.transpose(ddid(img,sigma2),[2,0,1]))
    res = torch.from_numpy(np.asarray(res)).cuda()
    return res

import glob
import re
import os




if __name__ == '__main__':
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # check file
    prefix = "data/threshold_20/"
    data_path = prefix + "test_tiny_ImageNet_"+str(1000)+"_adv_" + str(args.attack_method) + "_" + str(args.epsilon) + ".h5"

    # com_data_path = prefix + "test_ImageNet_"+str(args.set_size)+"_com_" + str(args.attack_method) + "_" + str(args.epsilon) + ".h5"
    assert os.path.exists(data_path), "not found expected file : "+data_path

    # get test data
    import  h5py
    h5_store = h5py.File(data_path,"r")
    data = torch.from_numpy(h5_store['data'][:])
    true_target = torch.from_numpy(h5_store['true_target'][:])
    sigma = torch.from_numpy(h5_store['sigma'][:])
    h5_store.close()

    #define batch_size
    batch_size = 50
    nb_steps= args.test_samples // batch_size

    #load net
    print('| Resuming from checkpoints...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    model = torch.load('./checkpoint/resnet50_epoch_22.pth')
    nb_epoch = 1
    model = model.to(device)

    for epoch in range(nb_epoch):
        #evaluate
        model.eval()
        clncorrect = 0

        for i in range(nb_steps):
            print("{}/{}".format(i,nb_steps))
            advdata = data[i*batch_size:(i+1)*batch_size,:,:,:].to(device)
            target = true_target[i*batch_size:(i+1)*batch_size].to(device)
            sigma_ = sigma[i*batch_size:(i+1)*batch_size].to(device)

            defence_data = adaptive_ddid_batch(advdata,sigma_)
            with torch.no_grad():
                output = model(defence_data.float())
            pred = output.max(1, keepdim=True)[1]
            pred = pred.double()
            target = target.double()
            clncorrect += pred.eq(target.view_as(pred)).sum().item()

        print('\nTest set with defence: '
              ' cln acc: {}/{} ({:.0f}%)\n'.format( clncorrect, args.test_samples,
                  100. * clncorrect / args.test_samples))
