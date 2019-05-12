import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
#from __future__ import print_function
import csv
import urllib3
from PIL import Image
from io import BytesIO
import math
#load in dataset
imageset = []
cannys = []

api_key = 'AIzaSyAM-dtPLIdddJwgcC8UXMRzSeVzsXqV8Z0'


def getImage(lat,long,zoom,api,keypoints,projW,projH):
    s = "https://maps.googleapis.com/maps/api/staticmap?center="+str(lat)+","+str(long)+"&size=1600x1600&zoom="+str(zoom)+"&maptype=satellite"
    for d in keypoints:
        s = s+"&markers=color:green%7C"+str(d[0])+","+str(d[1])
    s = s+"&key="+api
    http = urllib3.PoolManager()
    print(s)
    r = http.request('GET', s)
    bg = Image.open(BytesIO(r.data)).resize((projW,projH))
    return bg

def get_kmeans_labels(image, K, random_state=0):
    height = len(image)
    width = len(image[0])
    pixs = [image[i][j] for i in range(height) for j in range(width)]
    #image[y][x] = pixs[540*y + x]
    kclass = KMeans(n_clusters=K,random_state=0)
    kclass.fit(pixs)
    labs = kclass.labels_
    truelabels = np.empty((height,width))
    for i in range(height):
        for j in range(width):
            truelabels[i][j] = labs[width*i + j]
    return truelabels

def colorzones(image,labels,toplabel):
    mask = np.zeros_like(image)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if labels[i][j] == toplabel:
                mask[i][j] = 255
    return mask

def mean_color(image, labels):
    ''' Function to map labels to average color of the segment.
        You do not need to change this function.
        Args:
            image: [ndarray (M x N x 3)] RGB image
            labels: [ndarray (M x N)] segmentation labels
        Returns:
            out: [ndarray (M x N x 3)] Mean color image
    '''
    out = np.zeros_like(image)
    labelcolors = []
    for label in np.unique(labels):
        indices = np.nonzero(labels == label)
        color = np.mean(image[indices], axis=0)
        out[indices] = color
        labelcolors.append(color)
    return out,labelcolors

def hsvdistance(colorlist,fromidx):
    dists = []
    h0,s0,v0 = colorlist[fromidx]
    for c in colorlist:
        h1,s1,v1 = c
        dh = min(abs(h1-h0), 360-abs(h1-h0)) / 180.0
        ds = abs(s1-s0)
        dv = abs(v1-v0) / 255.0
        #distance = np.sqrt(dh*dh+ds*ds+dv*dv)
        dists.append(dh)
    print(dists)
    sortd = np.argsort(dists)
    print(sortd)
    idx = sortd[1]
    idx2 = sortd[2]

    print(colorlist[fromidx])
    print(colorlist[idx])
    print(colorlist[idx2])
    print(dists[idx])
    print(dists[idx2])
    return idx,idx2
def getmask(image):
    #blur image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blur = cv2.bilateralFilter(image,9,250,250)
    #kmeans image
    kmeans_8_labels  = get_kmeans_labels(blur,  8, 0)
    labels,colorlist = mean_color(image,kmeans_8_labels)
    bgrlabels = cv2.cvtColor(labels, cv2.COLOR_HSV2BGR)

    cv2.imshow('kmeans',bgrlabels)
    cv2.waitKey()
    cv2.imwrite('aerials2/kmeans8.png',bgrlabels)
    print(colorlist)
    unique, counts = np.unique(kmeans_8_labels, return_counts=True)
    top = np.argsort(counts)[-1]
    idx2,idx3 = hsvdistance(colorlist,top)
    kmeans_8_labels[kmeans_8_labels == idx2] = top
    kmeans_8_labels[kmeans_8_labels == idx3] = top

    labels,colorlist = mean_color(image,kmeans_8_labels)
    bgrlabels = cv2.cvtColor(labels, cv2.COLOR_HSV2BGR)
    cv2.imshow('kmeans',bgrlabels)
    cv2.imwrite('aerials2/kmeans8after.png',bgrlabels)

    mask = colorzones(blur,kmeans_8_labels,top)
    mask[np.where((mask==255).all(axis=2))] = [255,255,255]
    mask = mask[247:1316, 113:1546]
    mask[60:-60,60:-60] = [0,0,0]
    #final = cv2.addWeighted(image, 1.0, mask,0.25, 0)
    return mask

lat,long = 40.113936, -88.231303
img = getImage(lat,long,19,api_key,[],1600,1600)
impath = 'aerialslive/kmeansmask.png'
img.save(impath)

imageset.append(cv2.imread(impath))

for image in imageset:
    #img = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
    m = getmask(image)
    cv2.imwrite(impath,m)
cv2.destroyAllWindows()
