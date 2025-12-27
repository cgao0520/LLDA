#! /usr/bin/env python
#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm, norm, inv, pinv
from math import sqrt
from genCov import genCov
from CovMean import CovMean
from Mult import Mult
#import skimage as ski


#########################################
## some functions
#########################################
"""
def GetCovMean(cov_list):
    #global train_data
    cov_img = []#np.array([])
    for img in cov_list:#train_data[which_digit][:how_many]:
        img = np.array(img)
        img.shape = 28, 28
        c = genCov(img)
        cov_img.append(c)
    m = CovMean(cov_img)
    return m
"""

def CovDist(a,b):
    dist = norm(logm(a)-logm(b))
    return dist

def ReadData(train, num_of_obj = -1):
    #########################################
    ## read image label
    filename = '../dataset/train-labels.idx1-ubyte' if train else '../dataset/t10k-labels.idx1-ubyte'
    fd = open(filename,'rb')
    h = fd.read(4) # magic number
    assert(h == b'\x00\x00\x08\x01')
    h = fd.read(4) # number of items
    lbs = h[0]*256**3+h[1]*256**2+h[2]*256+h[3]
    if num_of_obj > 0:
        lbs = num_of_obj
    #assert(lbs == 60000)

    label=list(fd.read(lbs))
    fd.close()

    label = np.array(label)

    #########################################
    ## read image data
    filename = '../dataset/train-images.idx3-ubyte' if train else '../dataset/t10k-images.idx3-ubyte'
    fd = open(filename,'rb')
    h = fd.read(4) # magic number
    assert(h == b'\x00\x00\x08\x03')
    h = fd.read(4) # number of images
    imgs = h[0]*256**3+h[1]*256**2+h[2]*256+h[3]
    if num_of_obj > 0:
        imgs = num_of_obj
    #assert(imgs == 60000)

    h = fd.read(4) # number of rows
    rows = h[0]*256**3+h[1]*256**2+h[2]*256+h[3]
    assert(rows == 28)
    h = fd.read(4) # number of columns
    cols = h[0]*256**3+h[1]*256**2+h[2]*256+h[3]
    assert(cols == 28)

    data=[[],[],[],[],[],[],[],[],[],[]]
    feat=[[],[],[],[],[],[],[],[],[],[]]
    for i in range(imgs):
        dat = list(fd.read(rows*cols))
        fea = np.array(dat)
        fea.shape = rows, cols
        fea = genCov(fea) # generate covariance feature
        feat[label[i]].append(fea)
        data[label[i]].append(dat)

    fd.close()
    #data = np.array(data)
    #feat = np.array(feat)

    return data, feat, label, rows, cols




#########################################
## show image
#########################################

def ShowImage(img_list, label_list):
    rn = len(img_list)
    n = int(sqrt(rn))
    n = n+1 if n*n < rn else n
    plt.set_cmap('gray')
    j=1
    for img in img_list:
        #img = np.reshape(train_data[i,:],(rows,cols))
        img = np.array(img)
        img.shape = 28,28
        plt.subplot(n,n,j)
        plt.imshow(img)
        plt.title(label_list[j-1])
        j += 1
    plt.show()


train_data, train_feat, train_label, rows, cols = ReadData(True,100) # read train data
#print(data.shape)
#print(label.shape)

c1, c2 = 1,5 # class 1 and class 2 for binary classification
c1_name = str(c1)
c2_name = str(c2)
m1 = CovMean(train_feat[c1])
m2 = CovMean(train_feat[c2])
train_set = np.concatenate((train_feat[c1],train_feat[c2]), axis=0)
m_all = CovMean(train_set)
#print("mean of cov images of class 1: ", m1)
#print("mean of cov images of class 2: ", m2)
#print("mean of cov images of whole set: ", m_all)

#la_train_feat_c1 = [logm(m) for m in train_feat[c1]]
#la_train_feat_c2 = [logm(m) for m in train_feat[c2]]

# compute the Sw, i.e., the sactter within each class
inv_m1 = inv(m1)
inv_m2 = inv(m2)
inv_m = inv(m_all)
Sw = np.zeros(inv_m1.shape)
for c in train_feat[c1]:
    m = logm(Mult(inv_m1, c))
    Sw = Sw + np.matmul(m, m.T)
for c in train_feat[c2]:
    m = logm(Mult(inv_m2, c))
    Sw = Sw + np.matmul(m, m.T)

v = np.matmul(pinv(Sw), logm(Mult(inv_m1, m2)))

vm1 = np.matmul(v.T, logm(m1))
vm2 = np.matmul(v.T, logm(m2))

test_data, test_feat, test_label, rows, cols = ReadData(False,3000) # read test data

mis=[]
mis_lbl=[]
correct=0
#########################################
## test part for LLDA
#########################################
i=0
print("Misclassified sample indices of {0}:".format(c1_name))
for c in test_feat[c1]:
    t = np.matmul(v.T, logm(c)) # projection
    d1 = norm(t - vm1)
    d2 = norm(t - vm2)
    if d1<d2:
        correct += 1
    else:
        mis.append(test_data[c1][i])
        mis_lbl.append(c1_name)
        print(i,end=',')
    i += 1

i=0
print("Misclassified sample indices of {0}:".format(c2_name))
for c in test_feat[c2]:
    t = np.matmul(v.T, logm(c)) # projection
    d1 = norm(t - vm1)
    d2 = norm(t - vm2)
    if d2<d1:
        correct += 1
    else:
        mis.append(test_data[c2][i])
        mis_lbl.append(c2_name)
        print(i,end=',')
    i += 1    

print("accuracy: ",correct/(len(test_feat[c1])+len(test_feat[c2])) )    
#########################################
## test part for LieMean
#########################################

"""
i=0
#n=9
print("misclassified images of class 1: ")
for img in test_feat[c1]:
    d1 = CovDist(img,m1)
    d2 = CovDist(img,m2)
    if d1<d2:
        correct += 1
    else:
        mis.append(test_data[c1][i])
        mis_lbl.append('c1')
        print(i,end=',')
    i += 1

print("\nmisclassified images of class 2: ")
i = 0
for img in test_feat[c2]:
    d1 = CovDist(img,m1)
    d2 = CovDist(img,m2)
    if d2<d1:
        correct += 1
    else:
        mis.append(test_data[c2][i])
        mis_lbl.append('c2')
        print(i,end=',')
    i += 1

print("accuracy: ",correct/(len(test_data[c1])+len(test_data[c2])) )
"""
#349 442 933 of image 1
#img_list = [test_data[1][349], test_data[1][442], test_data[1][933]]

ShowImage(mis,mis_lbl)
