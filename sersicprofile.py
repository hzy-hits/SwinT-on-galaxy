# %%
from astropy.io import fits

import glob
import sys
import os
import shutil
import csv
import random
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import gamma
from multiprocessing import Pool
import multiprocessing as mp
from sklearn.metrics import mean_squared_error

# %%
#climit=int(100000)
#sys.setrecursionlimit(climit)
np.set_printoptions(threshold=1e6)
pixs = int(224)
#pixs=int(61)
lenth = int(pixs) * (0.074)
num_need = 1000
mode = "test"

# %%

if mode == "train":

    galaxy_dir = './data/train'
    if os.path.exists(galaxy_dir):
        shutil.rmtree(galaxy_dir)
    os.mkdir(galaxy_dir)

    csv_file = './data/tain/train_data.csv'
elif mode == "test":

    galaxy_dir = './data/test'
    if os.path.exists(galaxy_dir):
        shutil.rmtree(galaxy_dir)
    os.mkdir(galaxy_dir)
    csv_file = './data/test/test_data.csv'

print(csv_file)
"""
def noise_catalog(csv_file):
    noise_file=[]
    with open (csv_file) as f:
        reader=csv.DictReader(f)
        for row in reader:
            noise_file.append(row['name'])
    return noise_file
       """


def read_noise(noise_file):
    #noise_dir='/Raid0/lirui/cutout/noise/data/fits/r/'
    #noise_dir='./data/r/'
    with fits.open(noise_file, memmap=False) as f:
        hdr = f[1].header
        noise = f[1].data

        x = random.randint(1, 8924)
        y = random.randint(1, 8924)
        noise = noise[x:x + 224, y:y + 224]
        f.close()

        #pix1=int(75-(pixs-1)/2)
        #pix2=int(75+(pixs-1)/2+1)
        #noise=noise[pix1:pix2,pix1:pix2]

        if random.choice([0, 1]) == 0:
            noise = np.flip(noise, 1)  #水平镜像
        if random.choice([0, 1]) == 0:
            noise = np.flip(noise, 0)  #垂直镜像
    return noise, hdr


def read_PSF(psf_fits):
    with fits.open(psf_fits, memmap=False) as f:
        PSF = f[0].data

    PSF = PSF / np.sum(PSF)
    return PSF


def csv_header():
    fname = csv_file
    headers = [
        'name', 'mag', 'phot_xcen', 'phot_ycen', 'phot_R_eff', 'phot_q',
        'phot_pa', 'phot_nser', 'SNR'
    ]
    with open(fname, 'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(headers)
    f.close()


def save_csv(paras):
    fname = csv_file
    data = paras
    with open(fname, 'a+', newline="") as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data)
    f.close()


csv_header()

# %%


class SersicProfile:

    def __init__(self, i, psf, noise, hdr):
        self.a = 0
        self.name = i
        self.n = self.func_nser()
        self.xcen = self.func_position()
        self.ycen = self.func_position()
        self.q = np.random.uniform(0.2, 1)
        self.pa = np.random.uniform(0, 180) / 180 * np.pi
        self.mag = self.func_mag()
        self.reff = self.func_Reff()
        self.x, self.y = self.coordinate()  #np.meshgrid()
        self.flux = np.power(10,
                             (25.94 - self.mag) / 2.5) * 0.074 * 0.074 * 150

        self.r = self.func_r()
        self.phot = self.func_phot(psf, hdr)
        self.galaxy = self.phot + noise
        self.snr = self.func_SNR(noise)

    def get_para(self):  #把参数打包出来
        return [
            self.name,
            round(self.mag, 6), self.xcen, self.ycen, self.reff, self.q,
            self.pa, self.n,
            round(self.snr, 6)
        ]

    def coordinate(self):

        nx = pixs
        ny = pixs
        xhilo = [-lenth / 2, lenth / 2]
        yhilo = [-lenth / 2, lenth / 2]
        x = (xhilo[1] - xhilo[0]) * np.outer(
            np.ones(ny), np.arange(nx)) / float(nx) + xhilo[0]
        y = (yhilo[1] - yhilo[0]) * np.outer(
            np.arange(ny), np.ones(nx)) / float(ny) + yhilo[0]
        return x, y

    def func_position(self):  #位置初始化方法
        posi = 0
        while True:
            posi = np.random.normal(0, 0.15)
            if -0.4 < posi < 0.4:
                break
        return posi

    def func_Reff(self):  #Reff初始化方法
        #Reff_range=[0.3,4]
        #Reff=np.random.normal(0.5,1.8)-0.1
        while True:
            r = np.random.normal(-0.1, 0.4)
            Reff = np.power(10, r) + 0.1
            if 0.2 < Reff < 4:
                break

        return Reff

    def func_mag(self):  #mag初始化方法
        mag_range = [17, 22]
        #mag=mag_range[0]+mag_range[1]-expon.rvs(loc=mag_range[0])
        while True:
            mag = mag_range[0] + mag_range[1] - (
                np.random.exponential(scale=1) + 17)
            if 17 < mag < 22:
                break
        return mag

    def func_nser(self):  #n_index初始化方法

        while True:
            n = np.random.f(30, 5) * 2 - 0.3
            if 0.1 < n < 8.0:
                break
        return n

    def func_r(self):  #算半径
        xnew = (self.x - self.xcen) * np.cos(
            self.pa) + (self.y - self.ycen) * np.sin(self.pa)
        ynew = -(self.x - self.xcen) * np.sin(
            self.pa) + (self.y - self.ycen) * np.cos(self.pa)
        return np.sqrt(xnew * xnew / self.q + ynew * ynew * self.q)

    def func_phot(self, psf, hdr):  #算算sersicprofile
        n = self.n
        #gain=hdr["gain"]
        if n >= 0.36:  # from Ciotti & Bertin 1999, truncated to n^-3
            k = 2.0 * n - 1. / 3 + 4. / (405. * n) + 46. / (
                25515. * n * n) + 131. / (1148175. * n * n * n)
        else:  # from MacArthur et al. 2003
            k = 0.01945 - 0.8902 * n + 10.95 * n * n - 19.67 * n * n * n + 13.43 * np.power(
                n, 4)
        phot = fftconvolve(
            self.flux * k**(2.0 * n) /
            (np.pi * self.reff * self.reff * gamma(2.0 * n + 1)) *
            np.exp(-k * (self.r / self.reff)**(1. / n)),
            psf,
            mode="same")
        phot = np.where(phot < 0, 0, phot)
        return phot

    def func_SNR(self, noise):
        x_max, y_max = self.xcen, self.ycen
        x_max_pix = np.where(x_max >= 0, int(x_max * 5 + 1),
                             int(x_max * 5 - 1))
        y_max_pix = np.where(y_max >= 0, int(y_max * 5 + 1),
                             int(y_max * 5 - 1))

        img1 = self.galaxy
        img1 = img1[112 - int(self.reff / 0.074) - 15:112 +
                    int(self.reff / 0.074) + 15, 112 - int(self.reff / 0.074) -
                    15:112 + int(self.reff / 0.074) + 15]

        noise = noise[112 - int(self.reff / 0.074) - 15:112 +
                      int(self.reff / 0.074) + 15,
                      112 - int(self.reff / 0.074) - 15:112 +
                      int(self.reff / 0.074) + 15]
        k2 = np.mean(np.array(noise))
        print(self.flux)
        img_one = np.ones(img1.shape) * k2
        mse1 = mean_squared_error(img1, img_one)
        mse2 = mean_squared_error(noise, img_one)

        if mse2 == 0:
            return 10000
        else:
            return mse1 / mse2

    def save_galaxy(self, fname, hdr, psf):

        hdu = fits.PrimaryHDU(self.galaxy)  #,header=hdr
        hdu.writeto(fname, overwrite=True)
        #fits.append(fname,np.array([psf]))
    def checksave(self, psf, hdr):

        if self.snr > 50:
            fname = galaxy_dir + '/' + str(self.name) + '.fits'
            self.save_galaxy(fname, psf, hdr)
            save_csv(self.get_para())
            return 1
        else:
            #print("Can not get a galaxy with high SNR,niose number is:",self.name)
            return 0


# %%
def mainfunc(num_need):
    noise_file = glob.glob('.\\data\\r\\noise\\*fits')
    psf_files = glob.glob('.\\data\\PSF\\*bz2')
    #print("ok")
    #num_noise=len(noise_file)
    num_psf = len(psf_files)
    i = 0
    data0 = []
    while i < num_need:
        noise, hdr = read_noise(noise_file[0])
        psf = read_PSF(psf_files[np.random.randint(0, num_psf)])
        #if int(i)%1000==0:
        #print ('galaxy simulation finished:', np.round(100.0*i/num_need,5),"%")
        x = SersicProfile(i, psf, noise, hdr)
        if x.checksave(psf, hdr) == 1:
            i += 1
            #data0.append(x.get_para())
    #data0=np.array(data0)

    return data0


if __name__ == '__main__':
    #pool = mp.Pool(6)
    mainfunc(num_need)

    #pool.apply_async(func=mainfunc, args=(num_need,), callback=save_csv)
    #pool.close()
    #pool.join()
