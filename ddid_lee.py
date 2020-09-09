import numpy as np
import cv2
from scipy.fftpack import fft, ifft
from config import  args
from time import *


def ddid(y, sigma2):
    '''
    :@y : [H,W,C] | [0,1]
    :@sigma2: (sigma/255)^2
    '''
    height, width, depth = np.shape(y)[0], np.shape(y)[1], np.shape(y)[2]
    s = [height * width, depth]
    M = np.zeros([depth, depth])
    M[0, :] = 1 * np.sqrt(1 / depth)
    for i in range(1, depth):
        for j in range(depth):
            M[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * depth)) * np.sqrt(2 / depth)
    y = np.reshape(np.dot(np.reshape(y, s), M), np.shape(y))

    for i in range(args.nb_iters):
        if i ==0:
            x = step(y, y, sigma2, args.r, args.sigma_s, 100, args.gamma_f[i])
        elif i ==1:
            x = step(x, y, sigma2, args.r, args.sigma_s, 8.7, args.gamma_f[i])
        elif i == 2:
            x = step(x, y, sigma2, args.r, args.sigma_s, 0.7, args.gamma_f[i])
        else:
            print("error: nb_iters is greater than 3!")
            raise

    x = np.reshape(np.asarray(np.asmatrix(np.reshape(x, s)) * np.asmatrix(M).I), np.shape(x))
    return x


def step(x, y, sigma2, r, sigma_s, gamma_r, gamma_f):
    begin_time = time()
    height, width, depth = np.shape(y)[0], np.shape(y)[1], np.shape(y)[2]
    xx = np.array(range(-r, r + 1))
    [dx, dy] = np.meshgrid(xx, xx)
    h = np.exp(-(dx ** 2 + dy ** 2) / (2 * (sigma_s ** 2)))
    xp = []
    yp = []
    for i in range(depth):
        xp.append(np.pad(x[:, :, i], (r, r), 'symmetric'))
        yp.append(np.pad(y[:, :, i], (r, r), 'symmetric'))

    xp = np.transpose(np.asarray(xp), [1, 2, 0])  # bug one
    yp = np.transpose(np.asarray(yp), [1, 2, 0])

    xt = np.zeros(np.shape(x))
    end_time = time()
    run_time = end_time - begin_time
    # print('step函数预处理时间：', run_time)

    for i in range(0, height):
        for j in range(0, width):

            # BF
            begin_time = time()
            g = xp[i:i + 2 * r + 1, j:j + 2 * r + 1, :]
            y = yp[i:i + 2 * r + 1, j:j + 2 * r + 1, :]
            d = g - g[r, r, :]
            d1 = np.power(d, 2)[:, :, 0] + np.power(d, 2)[:, :, 1] + np.power(d, 2)[:, :, 2]
            k = np.exp(- d1 / (gamma_r * sigma2)) * h  # Eq. 4

            gt = []
            st = []
            for c in range(depth):
                gt.append((g[:, :, c] * k).sum() / k.sum())
                st.append((y[:, :, c] * k).sum() / k.sum())
            gt = np.reshape(np.asarray(gt), [1, 1, -1])
            st = np.reshape(np.asarray(st), [1, 1, -1])

            end_time = time()
            run_time = end_time - begin_time
            # print('BF时间：', run_time)

            # Fourier Domain: Wavelet Shrinkage
            begin_time = time()

            V = sigma2 * ((k ** 2).sum())
            Gt = []
            St = []
            minus_g = g - gt
            minus_y = y - st
            for c in range(depth):
                Gt.append(np.roll(np.roll(minus_g[:, :, c] * k, shift=-r, axis=1), shift=-r, axis=0))
                St.append(np.roll(np.roll(minus_y[:, :, c] * k, shift=-r, axis=1), shift=-r, axis=0))

            Gt = np.transpose(np.asarray(Gt), [1, 2, 0])
            St = np.transpose(np.asarray(St), [1, 2, 0])

            G = []
            S = []
            for c in range(depth):
                G.append(np.fft.fft2(Gt[:, :, c]))
                S.append(np.fft.fft2(St[:, :, c]))
            S = np.transpose(np.asarray(S), [1, 2, 0])
            G = np.transpose(np.asarray(G), [1, 2, 0])
            K = np.exp((- gamma_f * V) / (G * np.conj(G)))

            St = []
            for c in range(depth):
                St.append((S[:, :, c] * K[:, :, c]).sum() / np.power(2 * r + 1, 2))
            St = np.asarray(St)
            end_time = time()
            run_time = end_time - begin_time
            # print('小波时间：', run_time)
            xt[i, j, :] = st + St.real
    return xt


if __name__ == "__main__":
    # sigma=30/255
    sigma = 50 / 255  # for test
    from PIL import Image

    # img = Image.fromarray(img.astype('uint8')).convert('RGB')
    # img.save('pil_house.png')
    # img=cv2.imread('Images/House256rgb.png')
    img = np.asarray(Image.open('Images/House256rgb.png'))
    # print(img)
    # print(np.max(img))
    # print(img.shape)
    # img = np.reshape(img,[np.shape(img)[0],np.shape(img)[1],-1])
    # print(img.shape)
    # print(img)
    img = img / 255
    # img = img *255
    # print(img)

    # cv2.imwrite('pure_house.png',img)
    xt = ddid(img, sigma ** 2)
    # xt=ddid(img,0.0384)

    # print(np.max(xt))
    # xt = xt*255
    # cv2.imwrite('python_Peppers512rgb.png',xt)
    # import matplotlib.image as mp
    # mp.imsave('mp_house.png',xt)
    img = xt * 255

    from PIL import Image

    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    img.save('pil_house.png')

