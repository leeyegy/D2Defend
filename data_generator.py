import glob
import cv2
import os
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import h5py
import argparse
from networks import  *

from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import TRAINED_MODEL_PATH
from advertorch.attacks import GradientSignAttack,L2PGDAttack,SpatialTransformAttack,JacobianSaliencyMapAttack,MomentumIterativeAttack

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import  args
from attackers import *
import json

patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128

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

def _get_test_adv(attack_method,epsilon):
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load basic data
    # test_loader = get_handled_cifar10_test_loader(num_workers=4, shuffle=False, batch_size=50,nb_samples=10000)
    test_loader = get_handled_cifar10_test_loader(num_workers=4, shuffle=False, batch_size=50)

    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    file_name = 'wide-resnet-' + str(args.depth) + 'x' + str(args.widen_factor) + '.t7'
    wideRes = torch.load(os.path.join(save_dir, file_name))
    wideRes = wideRes['net']
    model = model.to(device)
    #model = torch.load('./checkpoint/cifar10_pgd_8_model_120.pth')
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # define adversary
    from advertorch.attacks import LinfPGDAttack
    if attack_method == "PGD":
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon,
            nb_iter=20, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
            targeted=False)
    elif attack_method == "FGSM":
        adversary = GradientSignAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            clip_min=0.0, clip_max=1.0, eps=epsilon, targeted=False) 
    elif attack_method == "Momentum":
        adversary = MomentumIterativeAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon,
            nb_iter=20, decay_factor=1.0, eps_iter=1.0, clip_min=0.0, clip_max=1.0,
            targeted=False, ord=np.inf)
    elif attack_method == "STA":
        adversary = SpatialTransformAttack(
            model, num_classes=args.num_classes, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            initial_const=0.05, max_iterations=500, search_steps=1, confidence=0, clip_min=0.0, clip_max=1.0,
            targeted=False, abort_early=True)
    elif attack_method == "DeepFool":
        adversary =  DeepFool(model,max_iter=20, clip_max=1.0, clip_min=0.0,epsilon=epsilon)
        # adversary =  DeepFool(model,max_iter=50, clip_max=1.0, clip_min=0.0,epsilon=epsilon)
        # adversary =  DeepFool(model,max_iter=500, clip_max=1.0, clip_min=0.0)

    elif attack_method == "CW":
        # adversary = CarliniWagnerL2Attack(
        #     model, num_classes=args.num_classes, epsilon=epsilon,loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        #     max_iterations=500, confidence=0, clip_min=0.0, clip_max=1.0,
        #     targeted=False, abort_early=False)
        adversary = CarliniWagnerL2Attack(
            model, num_classes=args.num_classes, epsilon=epsilon,loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            max_iterations=10, confidence=0, clip_min=0.0, clip_max=1.0,
            targeted=False, abort_early=True)
        # adversary = CarliniWagnerL2Attack(
        #     model, num_classes=args.num_classes, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        #     max_iterations=1000, confidence=0, clip_min=0.0, clip_max=1.0,
        #     targeted=False, abort_early=True)
    elif attack_method == "BIM":
        adversary = BIM(model, eps=epsilon, eps_iter=0.01, n_iter=20, clip_max=1.0, clip_min=0.0)
    elif attack_method == "JSMA":
        adversary = JacobianSaliencyMapAttack(model, num_classes=args.num_classes,
                 clip_min=0.0, clip_max=1.0, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                 theta=epsilon, gamma=1.0, comply_cleverhans=False)


    # generate for train.h5 | save as train_adv_attackMethod_epsilon
    test_adv = []
    test_true_target = []
    for clndata, target in test_loader:
        clndata, target = clndata.to(device), target.to(device)
        with ctx_noparamgrad_and_eval(model):
            advdata = adversary.perturb(clndata, target)
            test_adv.append(advdata.detach().cpu().numpy())
        test_true_target.append(target.cpu().numpy())
    test_adv = np.reshape(np.asarray(test_adv),[-1,3,32,32])
    test_true_target = np.reshape(np.asarray(test_true_target),[-1])
    print("test_adv.shape:{}".format(test_adv.shape))
    print("test_true_target.shape:{}".format(test_true_target.shape))
    del model

    return test_adv, test_true_target

def get_test_adv_loader_adaptive(attack_method,epsilon):
    #check file
    file_name = os.path.join("data","threshold_"+str(args.threshold),"new_"+str(attack_method)+"_"+str(epsilon)+".h5")
    assert  os.path.exists(file_name),"expected file not found in :{}".format(file_name)

    #save file
    h5_store = h5py.File(file_name, 'r')
    test_data = h5_store['data'][:] 
    try:
        test_true_target=h5_store['true_target'][:]
    except:
        test_true_target=h5_store['target'][:]
    sigma = h5_store['sigma'][:]
    h5_store.close()

    train_data = torch.from_numpy(test_data)
    train_target = torch.from_numpy(test_true_target) 
    sigma = torch.from_numpy(sigma)
    train_dataset = CIFAR10Dataset_ada(train_data, train_target,sigma)
    del train_data,train_target,sigma
    return DataLoader(dataset=train_dataset, num_workers=2, drop_last=True, batch_size=50,
                  shuffle=False)

def get_test_adv_loader(attack_method,epsilon):
    #save file
    if os.path.exists("data/test_adv_"+str(attack_method)+"_"+str(epsilon)+".h5"):
        h5_store = h5py.File("data/test_adv_"+str(attack_method)+"_"+str(epsilon)+".h5", 'r')
        test_data = h5_store['data'][:]
        try:
            test_true_target=h5_store['true_target'][:]
        except:
            test_true_target=h5_store['target'][:]
        h5_store.close()
    else:

        test_data,test_true_target = _get_test_adv(attack_method,epsilon)
        h5_store = h5py.File("data/test_adv_"+str(attack_method)+"_"+str(epsilon)+".h5", 'w')
        h5_store.create_dataset('data' ,data= test_data)
        h5_store.create_dataset('true_target',data=test_true_target)
        h5_store.close()

    train_data = torch.from_numpy(test_data)
    train_target = torch.from_numpy(test_true_target)  # numpy转Tensor
    train_dataset = CIFAR10Dataset(train_data, train_target)
    del train_data,train_target
    return DataLoader(dataset=train_dataset, num_workers=2, drop_last=True, batch_size=50,
                  shuffle=False)

#generate h5file for test data regarding specific attack
def generate_attackh5(save_dir="data",attack_method="PGD",epsilon=8/255):
    '''
    :param attack_method:
    :param epsilon:
    :return: the name of file where (test_adv_data, test_true_lable) is stored
    '''
    file_name = "test_"+attack_method+"_"+epsilon+".h5"
    file_path = os.path.join(save_dir,file_name)
    if not os.exists(save_dir):
        os.mkdir(save_dir)
    else:
        # get raw test data
        data,target = get_test_raw_data()

class CIFAR10Dataset_ada(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """

    def __init__(self, data, target,sigma):
        super(CIFAR10Dataset_ada, self).__init__()
        self.data = data
        self.target = target
        self.sigma=sigma

    def __getitem__(self, index): 
        batch_x = self.data[index]
        batch_y = self.target[index]
        batch_s = self.sigma[index]

        return batch_x, batch_y,batch_s

    def __len__(self):
        return self.data.size(0)

class CIFAR10Dataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """

    def __init__(self, data, target):
        super(CIFAR10Dataset, self).__init__()
        self.data = data
        self.target = target

    def __getitem__(self, index): 
        batch_x = self.data[index]
        batch_y = self.target[index]
        return batch_x, batch_y

    def __len__(self):
        return self.data.size(0)

def get_raw_cifar10_data(loader):
    train_data = []
    train_target = []

    # load data
    for batch_idx, (data, target) in enumerate(loader):
        train_data.append(data.numpy())
        train_target.append(target.numpy())
    train_data = np.asarray(train_data)
    train_target = np.asarray(train_target)
    train_data = train_data.reshape([-1, 3, 32, 32])
    train_target = np.reshape(train_target, [-1])

    return train_data, train_target


def get_handled_cifar10_train_loader(batch_size, num_workers, shuffle=True):
    if os.path.exists("data/train.h5"):
        h5_store = h5py.File("data/train.h5", 'r')
        train_data = h5_store['data'][:]
        train_target = h5_store['target'][:]
        h5_store.close()
        print("^_^ data loaded successfully from train.h5")
    else:
        h5_store = h5py.File("data/train.h5", 'w')

        transform = transforms.Compose([transforms.ToTensor()])
        trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
        train_data, train_target = get_raw_cifar10_data(train_loader)


        h5_store.create_dataset('data' ,data= train_data)
        h5_store.create_dataset('target',data = train_target)
        h5_store.close()

    train_data = torch.from_numpy(train_data)
    train_target = torch.from_numpy(train_target) 
    train_dataset = CIFAR10Dataset(train_data, train_target)
    del train_data,train_target
    return DataLoader(dataset=train_dataset, num_workers=num_workers, drop_last=True, batch_size=batch_size,
                  shuffle=shuffle)

def get_handled_cifar10_test_loader(batch_size, num_workers, shuffle=True):
    if os.path.exists("data/test.h5"):
        # h5_store = pd.HDFStore("data/train.h5", mode='r')
        h5_store = h5py.File("data/test.h5", 'r')
        train_data = h5_store['data'][:] 
        train_target = h5_store['target'][:]
        h5_store.close()
        print("^_^ data loaded successfully from test.h5")

    else:
        h5_store = h5py.File("data/test.h5", 'w')

        transform = transforms.Compose([transforms.ToTensor()])
        trainset = CIFAR10(root="./data", train=False, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
        train_data, train_target = get_raw_cifar10_data(train_loader)


        h5_store.create_dataset('data' ,data= train_data)
        h5_store.create_dataset('target',data = train_target)
        h5_store.close()


    train_data = torch.from_numpy(train_data)
    train_target = torch.from_numpy(train_target)  
    train_dataset = CIFAR10Dataset(train_data, train_target)
    del train_data,train_target
    return DataLoader(dataset=train_dataset, num_workers=num_workers, drop_last=True, batch_size=batch_size,
                      shuffle=shuffle)

def get_test_raw_data():
    '''
    :return: train_image ,  train_target  | tensor
    '''
    if os.path.exists("data/test.h5"):
        h5_store = h5py.File("data/test.h5", 'r')
        train_data = h5_store['data'][:]
        train_target = h5_store['target'][:]
        h5_store.close()
    else:
        h5_store = h5py.File("data/test.h5", 'w')

        transform = transforms.Compose([transforms.ToTensor()])
        trainset = CIFAR10(root="./data", train=False, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
        train_data, train_target = get_raw_cifar10_data(train_loader)


        h5_store.create_dataset('data' ,data= train_data)
        h5_store.create_dataset('target',data = train_target)
        h5_store.close()

    train_data = torch.from_numpy(train_data)
    train_target = torch.from_numpy(train_target)  # numpy转Tensor
    return train_data,train_target

def get_train_raw_data():
    '''
    :return: train_image ,  train_target  | tensor
    '''
    if os.path.exists("data/train.h5"):
        # h5_store = pd.HDFStore("data/train.h5", mode='r')
        h5_store = h5py.File("data/train.h5", 'r')
        train_data = h5_store['data'][:] 
        train_target = h5_store['target'][:]
        h5_store.close()
        # print("^_^ data loaded successfully from train.h5")
    else:
        h5_store = h5py.File("data/train.h5", 'w')

        transform = transforms.Compose([transforms.ToTensor()])
        trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
        train_data, train_target = get_raw_cifar10_data(train_loader)


        h5_store.create_dataset('data' ,data= train_data)
        h5_store.create_dataset('target',data = train_target)
        h5_store.close()

    train_data = torch.from_numpy(train_data)
    train_target = torch.from_numpy(train_target)  # numpy转Tensor
    return train_data,train_target

def h52image(h5_path,save_path,dataset="cifar10"):
    # path check
    assert  os.path.exists(h5_path),"expected file not found:{}".format(h5_path)
    assert dataset=="tiny_imagenet" or dataset == "cifar10", "only support cifar10 and tiny_imagenet"
    if not  os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = np.zeros([200]) if dataset=="tiny_imagenet" else  np.zeros([10])

    # load data
    h5_store = h5py.File(h5_path,"r")
    data = h5_store['data'][:]
    try :
        target = h5_store['true_target'][:]
    except:
        target = h5_store['target'][:]
    data = torch.from_numpy(data).float()

    # save image
    for i in range(data.size()[0]):
        # img = (data[i] * 255).astype(np.int16)
        img = transforms.ToPILImage()(data[i])
        img_save_dir = os.path.join(save_path,str(int(target[i])))
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        img_save_name = os.path.join(img_save_dir,str(file_name[int(target[i])])+".png")
        # cv2.imwrite(img_save_name,img)
        img.save(img_save_name)
        file_name[int(target[i])] += 1

def _generate_adaptive_sigma_h5(h5_path,save_path,sigma_path,dataset="cifar10"):
    # path check
    assert  os.path.exists(h5_path),"expected file not found:{}".format(h5_path)
    if not  os.path.exists(save_path):
        os.makedirs(save_path)

    file_name = np.zeros([200]) if dataset=="tiny_imagenet" else  np.zeros([10])

    save_file_name = os.path.join(save_path,"test_tiny_ImageNet_1000_adv_"+args.attack_method+"_"+str(args.epsilon)+".h5") if dataset=="tiny_imagenet" else os.path.join(save_path,"new_"+args.attack_method+"_"+str(args.epsilon)+".h5")


    # load data
    h5_store = h5py.File(h5_path,"r")
    data = h5_store['data'][:]
    try :
        target = h5_store['true_target'][:]
    except:
        target = h5_store['target'][:]

    with open(sigma_path,"r") as sigma:
        sigma_dict = json.load(sigma)

    # print("target.shape:{}".format(target.shape))
    # assign adaptive_sigma
    adaptive_sigma = np.zeros(target.shape)
    for i in range(data.shape[0]):
        key = str(int(target[i]))
        # print(sigma_dict[key])
        # print("key:{}".format(key))
        try:
            adaptive_sigma[i] = sigma_dict[key][int(file_name[int(target[i])])]
        except:
            adaptive_sigma[i] = sigma_dict[key]
        file_name[int(target[i])] += 1

    # save file
    h5_store = h5py.File(save_file_name, 'w')
    h5_store.create_dataset('data' ,data= data)
    h5_store.create_dataset('true_target',data = target)
    h5_store.create_dataset('sigma',data = adaptive_sigma)
    h5_store.close()
    print("adaptive sigma file saved in {} with data and target ~".format(save_path))

   
if __name__ == '__main__':
    if args.task == "g_adv":
        get_test_adv_loader(args.attack_method,args.epsilon)
    elif args.task == "g_img":
        h5_path = os.path.join("data","test_adv_"+args.attack_method+"_"+str(args.epsilon)+".h5")
        save_path = os.path.join("data","cifar10_img","val",args.attack_method+"_"+str(args.epsilon))
        h52image(h5_path,save_path,dataset="cifar10")
    elif args.task == "g_adaptive_sigma":
        h5_path = os.path.join("data","test_adv_"+args.attack_method+"_"+str(args.epsilon)+".h5")
        save_path = os.path.join("data","threshold_"+str(args.threshold))
        sigma_path = os.path.join("data","threshold_"+str(args.threshold),args.attack_method+"_"+str(args.epsilon)+".json")
        _generate_adaptive_sigma_h5(h5_path,save_path,sigma_path,dataset="cifar10")        

    # h52image
    # h5_path = os.path.join("data","new_test_adv_"+args.attack_method+"_"+str(args.epsilon)+".h5")
    # # h5_path = os.path.join("data","train.h5")
    # save_path = os.path.join("data","cifar10_img","val","new",args.attack_method+"_"+str(args.epsilon))
    # # save_path = os.path.join("data","cifar10_img","train","NONE_0.0")
    # # h5_path = os.path.join("data","test_tiny_ImageNet_1000_adv_"+args.attack_method+"_"+str(args.epsilon)+".h5")
    # # save_path = os.path.join("data","tiny_imagenet_img",args.attack_method+"_"+str(args.epsilon))
    # h52image(h5_path,save_path,dataset="cifar10")

#     #_generate_adaptive_sigma_h5
#     # h5_path = os.path.join("data","test_tiny_ImageNet_1000_adv_"+args.attack_method+"_"+str(args.epsilon)+".h5")
#     h5_path = os.path.join("data","new_test_adv_"+args.attack_method+"_"+str(args.epsilon)+".h5")
#     save_path = os.path.join("data","threshold_"+str(args.threshold))
#     sigma_path = os.path.join("data","threshold_"+str(args.threshold),"new_"+args.attack_method+"_"+str(args.epsilon)+".json")
#     _generate_adaptive_sigma_h5(h5_path,save_path,sigma_path,dataset="cifar10")
