'''
@function: pixel deflection
@function: wavelet transformation | or other conventional denoiser
'''

from __future__ import division
from random import randint, uniform
import os, sys, argparse, matplotlib, multiprocessing
from imageio import imread
from joblib import Parallel, delayed
from glob import glob
import numpy as np

def wrapper_pd_block(args, image, defend=True):
    '''
    :function : multi samples version for pd_block
    :param args:
    :param image: (batch_size,C,H,W) | [0,1]
    :param defend: if false then do nothing
    :return: (batch_size,C,H,W) | [0,1]
    '''
    res = []
    for i in range(image.shape[0]):
        res.append(np.transpose(pd_block(args,np.transpose(image[i],[1,2,0]),defend),[2,0,1]))
    res = np.asarray(res)
    return res


def denoiser(denoiser_name, img,sigma):
    '''
    :param denoiser_name: str| 'wavelet' or 'TVM' or 'bilateral' or 'deconv' or 'NLM'
    :param img: (H,W,C) | np.array | [0,1]
    :param sigma:  for wavelet
    :return:(H,W,C) | np.array | [0,1]
    '''
    from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, denoise_nl_means, wiener)
    if denoiser_name == 'wavelet':
        return denoise_wavelet(img,sigma=sigma, mode='soft', multichannel=True,convert2ycbcr=True, method='BayesShrink')
    elif denoiser_name == 'TVM':
        return denoise_tv_chambolle(img, multichannel=True)
    elif denoiser_name == 'bilateral':
        return denoise_bilateral(img, bins=1000, multichannel=True)
    elif denoiser_name == 'deconv':
        return wiener(img)
    elif denoiser_name == 'NLM':
        return denoise_nl_means(img, multichannel=True)
    else:
        raise Exception('Incorrect denoiser mentioned. Options: wavelet, TVM, bilateral, deconv, NLM')

def pixel_deflection(img, rcam_prob, deflections, window, sigma):
    '''
    :param img: (H,W,C)| np.array | [0,255]
    :param rcam_prob: (H,W) | np.array| either get map matrix or the zero matrix (without map)
    :param deflections: the number of pixels that will be deflected | non-negative int
    :param window: window size around target pixel for random selecting a pixel to replace the target pixel
    :param sigma:  useless in this version
    :return: image data matrix after adding noise by pixel-deflection procedure  | (H,W,C) | np.array | [0,255]
    '''
    H, W, C = img.shape
    while deflections > 0:
        for c in range(C):
            x,y = randint(0,H-1), randint(0,W-1)

            if uniform(0,1) < rcam_prob[x,y]:
                continue

            while True: #this is to ensure that PD pixel lies inside the image
                a,b = randint(-1*window,window), randint(-1*window,window)
                if x+a < H and x+a > 0 and y+b < W and y+b > 0: break
            img[x,y,c] = img[x+a,y+b,c]
            deflections -= 1
    return img

def pd_block(args, image, defend=True):
    '''
    :param args:  model setting | parser
    :param image: (H,W,C) | np.array | [0,1]
    :param defend: whether use pixel_deflection & wavelet | if defend is false then do nothing
    :return: (H,W,C) | [0,1]
    '''

    # assumes map is same name as image but inside maps directory
    if not args.disable_map:
        # map   = get_map('./maps/'+image_name.split('/')[-1])
        pass
    else:
        map  = np.zeros((image.shape[0],image.shape[1]))


    if defend:
        img = pixel_deflection(image, map, args.deflections, args.window, args.sigma_pixel_deflection)
        return denoiser(args.denoiser, img, args.sigma_pixel_deflection)
    else:
        return image
