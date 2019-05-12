import cv2
import numpy as np
from PIL import Image
import math

import os
relevant_path = "vidtest"
included_extensions = ['jpg','jpeg']
file_names = [fn for fn in sorted(os.listdir(relevant_path))
              if any(fn.endswith(ext) for ext in included_extensions)]

fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
video = cv2.VideoWriter('OUTPUT.mp4',fourcc,2,(756,1008))

numframes = len(file_names)
i = 0
for fln in file_names:
    i += 1
    bg = Image.open(os.path.join(relevant_path,fln))
    bg = bg.rotate(-90)
    bg = bg.resize((756,1008))
    bg.save('laneresized.png')

    img = cv2.imread('laneresized.png')
    origimg = img
    img = img[int(img.shape[0]/2):int(img.shape[0])]
    # cv2.imshow('orig',img)
    # cv2.waitKey()

    blur = cv2.bilateralFilter(img,9,100,100)
    # cv2.imshow('blur',blur)
    # cv2.waitKey()

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray',gray)
    # cv2.waitKey()

    ret,thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    # cv2.imshow('thresh',thresh)
    # cv2.waitKey()

    hkernel = np.ones((1,20), np.uint8)  # note this is a horizontal kernel
    vkernel = np.ones((20,1), np.uint8)  # note this is a vertical kernel
    d_im = cv2.dilate(thresh, vkernel, iterations=1)
    e_im = cv2.erode(d_im,hkernel,iterations = 4)
    d_im2 = cv2.dilate(e_im, vkernel, iterations=3)

    # cv2.imshow('1x vertical dilation',d_im)
    # cv2.waitKey()
    #
    # cv2.imshow('4x horiz dilation',e_im)
    # cv2.waitKey()
    #
    # cv2.imshow('3x vertical dilation',d_im2)
    # cv2.waitKey()

    sobelx = np.uint8(np.absolute(cv2.Sobel(d_im2,cv2.CV_64F,1,0,ksize=5)))
    sobely = np.uint8(np.absolute(cv2.Sobel(d_im2,cv2.CV_64F,0,1,ksize=5)))

    edges = cv2.bitwise_or(sobelx,sobely)
    minrow = 0
    for rowidx in range(len(edges)):
        if sum(edges[rowidx]) > 0:
            minrow = rowidx
            break;
    nonz = np.average(np.nonzero(edges[minrow]))
    if math.isnan(nonz):
        midpoint = int(edges.shape[1]/2)
    else:
        midpoint = int(nonz)
    edgecolor = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edgecolor[np.where((edgecolor != [0,0,0]).all(axis=2))] = [0,255,0]
    cv2.arrowedLine(edgecolor,(int(edges.shape[1]/2),int(3*edges.shape[0]/4)),(midpoint,minrow),(0,0,255),2)

    pad = np.zeros((edgecolor.shape),dtype='uint8')

    edgewithframe = np.concatenate((pad,edgecolor))
    #edgeframe =
    # cv2.imshow('edges',edgecolor)
    # cv2.waitKey()
    print("Frame: ",i,"/",numframes)
    edged = cv2.addWeighted(origimg,0.8,edgewithframe,0.8,0)
    print(edged.shape)

    #edged = cv2.addWeighted(img,0.8,edgecolor,0.8,0)
    video.write(edged)

video.release()
cv2.destroyAllWindows()
