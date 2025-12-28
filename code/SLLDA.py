#! /usr/bin/env python
#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm, norm, inv, pinv
from math import sqrt
from genCov import genCov
from CovMean import CovMean
from Mult import Mult
from MnistLabels import count_mnist_labels
import sys
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

def ReadData(train, class1, class2, num_of_obj = -1):
    #########################################
    ## read image label
    filename = '../dataset/train-labels.idx1-ubyte' if train else '../dataset/t10k-labels.idx1-ubyte'
    fd = open(filename,'rb')
    h = fd.read(4) # magic number
    assert(h == b'\x00\x00\x08\x01')
    h = fd.read(4) # number of items
    lbs = h[0]*256**3+h[1]*256**2+h[2]*256+h[3]
    assert(lbs >= num_of_obj)
    # Find all matching positions
    labels = np.frombuffer(fd.read(), dtype=np.uint8, offset=0)
    indices1 = np.where(labels == class1)[0]
    labels1 = indices1[:num_of_obj]
    indices2 = np.where(labels == class2)[0]
    labels2 = indices2[:num_of_obj]
    fd.close()

    #########################################
    ## read image data
    filename = '../dataset/train-images.idx3-ubyte' if train else '../dataset/t10k-images.idx3-ubyte'
    fd = open(filename,'rb')
    h = fd.read(4) # magic number
    assert(h == b'\x00\x00\x08\x03')
    h = fd.read(4) # number of images
    imgs = h[0]*256**3+h[1]*256**2+h[2]*256+h[3]
    assert(imgs >= num_of_obj)

    h = fd.read(4) # number of rows
    rows = h[0]*256**3+h[1]*256**2+h[2]*256+h[3]
    assert(rows == 28)
    h = fd.read(4) # number of columns
    cols = h[0]*256**3+h[1]*256**2+h[2]*256+h[3]
    assert(cols == 28)

    all_data = np.frombuffer(fd.read(), dtype=np.uint8, offset=0)
    # Reshape to (N, 784) so we can easily index into it
    all_images = all_data.reshape(-1, rows * cols)

    image_list1 = []
    image_list2 = []
    feat_list1 = []
    feat_list2 = []
    for idx in labels1:
        # Extract the specific image row and reshape to 28x28 matrix
        matrix = all_images[idx].reshape(rows, cols)
        image_list1.append(matrix)
        feat_list1.append(genCov(matrix))

    for idx in labels2:
        # Extract the specific image row and reshape to 28x28 matrix
        matrix = all_images[idx].reshape(rows, cols)
        image_list2.append(matrix)
        feat_list2.append(genCov(matrix))

    fd.close()

    return image_list1, image_list2, feat_list1, feat_list2, labels1, labels2, rows, cols

#########################################
## show image
#########################################

def ShowImage(img_list, label_list, title):
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
    plt.suptitle(title)
    plt.show()



def print_usage():
    print("Usage: python SLLDA.py <0-9> <0-9> <m> <n>\n")
    print("Where \n\t<0-9> indicates digit number between 0 and 9 for a class")
    print("\tm in <m> indicates instances for each training class")
    print("\tn in <n> indicates instances for each testing class\n")
    sys.exit(0)

def validate_args(train_label_counts, test_label_counts):
# Check if run this program in expected format
    if len(sys.argv) != 5:
        print_usage()

    try:
        # Convert strings to integers
        class1 = int(sys.argv[1])
        class2 = int(sys.argv[2])
        
        # Range check
        if not (0 <= class1 <= 9) or not (0 <= class2 <= 9):
            raise ValueError("Digits must be between 0 and 9")
        
        num_obj_for_train = int(sys.argv[3])
        num_obj_for_test  = int(sys.argv[4])
        
        if (num_obj_for_train > train_label_counts[class1] or num_obj_for_train > train_label_counts[class2]):
            print("Requested number of training instances is too big, specifically, total instances of class 1: %d and class 2: %d" % \
                  (train_label_counts[class1], train_label_counts[class2]))
            raise ValueError("Requested training instance is too big")
            
        if (num_obj_for_test > test_label_counts[class1] or num_obj_for_test > test_label_counts[class2]):
            print("Requested number of testing instances is too big, specifically, total instances of class 1: %d and class 2: %d" % \
                  (test_label_counts[class1], test_label_counts[class2]))
            raise ValueError("Requested testing instance is too big")
            
        return class1, class2, num_obj_for_train, num_obj_for_test

    except ValueError as e:
        sys.exit(1)

####################################################################################################
# Code Entry starts from here
####################################################################################################

#Analyze original MNIST dataset to get numbers of instance/objects of each digit
train_label_counts = count_mnist_labels("../dataset/t10k-labels.idx1-ubyte") # use "t10k-images.idx3-ubyte", the small dataset for training
test_label_counts  = count_mnist_labels("../dataset/train-labels.idx1-ubyte") # for testing, even though file was named for train

class1, class2, num_obj_for_train, num_obj_for_test = validate_args(train_label_counts, test_label_counts)

#print(class1, class2, num_obj_for_train, num_obj_for_test)

image_list1, image_list2, feat_list1, feat_list2, labels1, labels2, rows, cols = ReadData(True, class1, class2, num_obj_for_train) # read train data

c1, c2 = class1,class2 # class 1 and class 2 for binary classification
c1_name = str(c1)
c2_name = str(c2)
m1 = CovMean(feat_list1)
m2 = CovMean(feat_list2)
train_set = np.concatenate((feat_list2,feat_list2), axis=0)
m_all = CovMean(train_set)

# compute the Sw, i.e., the sactter within each class
inv_m1 = inv(m1)
inv_m2 = inv(m2)
inv_m = inv(m_all)
Sw = np.zeros(inv_m1.shape)

for c in feat_list1:
    m = logm(Mult(inv_m1, c))
    Sw = Sw + np.matmul(m, m.T)
for c in feat_list2:
    m = logm(Mult(inv_m2, c))
    Sw = Sw + np.matmul(m, m.T)

v = np.matmul(pinv(Sw), logm(Mult(inv_m1, m2)))

vm1 = np.matmul(v.T, logm(m1))
vm2 = np.matmul(v.T, logm(m2))

image_list1, image_list2, feat_list1, feat_list2, labels1, labels2, rows, cols = ReadData(False, class1, class2, num_obj_for_test) # read train data

mis=[]
mis_lbl=[]
correct=0
#########################################
## test part for LLDA
#########################################
i=0
print("Misclassified sample indices of {0}:".format(c1_name))
for c in feat_list1:
    t = np.matmul(v.T, logm(c)) # projection
    d1 = norm(t - vm1)
    d2 = norm(t - vm2)
    if d1<d2:
        correct += 1
    else:
        mis.append(image_list1[i])
        mis_lbl.append(c1_name)
        print(i,end=',')
    i += 1

i=0
print("Misclassified sample indices of {0}:".format(c2_name))
for c in feat_list2:
    t = np.matmul(v.T, logm(c)) # projection
    d1 = norm(t - vm1)
    d2 = norm(t - vm2)
    if d2<d1:
        correct += 1
    else:
        mis.append(image_list2[i])
        mis_lbl.append(c2_name)
        print(i,end=',')
    i += 1    

acc = correct/(len(feat_list1)+len(feat_list2))
print("accuracy: ", acc )

acc *= 100
title = f"Classification Accuracy: {acc:.2f}%, misclassified images are shown below"

ShowImage(mis, mis_lbl, title)
