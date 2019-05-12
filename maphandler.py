#from __future__ import print_function
import cv2
import csv
import urllib3
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
import math

api_key = 'AIzaSyAM-dtPLIdddJwgcC8UXMRzSeVzsXqV8Z0'


def getImage(lat,long,zoom,api,keypoints,projW,projH):
    s = "https://maps.googleapis.com/maps/api/staticmap?center="+str(cenLat)+","+str(cenLong)+"&size=1600x1600&zoom="+str(zoom)+"&maptype=satellite"
    for d in keypoints:
        s = s+"&markers=color:green%7C"+str(d[0])+","+str(d[1])
    s = s+"&key="+api
    http = urllib3.PoolManager()
    print(s)
    r = http.request('GET', s)
    bg = Image.open(BytesIO(r.data)).resize((projW,projH))
    return bg

def getPointLatLng(x, y, zoom, cenLat, cenLong):
    parallelMultiplier = math.cos(cenLat * math.pi / 180)
    degreesPerPixelX = 360 / math.pow(2, zoom + 8)
    degreesPerPixelY = 360 / math.pow(2, zoom + 8) * parallelMultiplier
    pointLat = cenLat - degreesPerPixelY * ( y - imH / 2)
    pointLng = cenLong + degreesPerPixelX * ( x  - imW / 2)

    return (pointLat, pointLng)

def threshold(impath):
    img = cv2.imread(impath)
    blur = cv2.bilateralFilter(img,9,100,100)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
    thresh[60:-60,60:-60] = [0]
    return thresh

def dilerod(impath):
    img = cv2.imread(impath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,bw = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    hkernel = np.ones((1,20), np.uint8)  # note this is a horizontal kernel
    vkernel = np.ones((20,1), np.uint8)  # note this is a vertical kernel
    d_im = cv2.dilate(bw, hkernel, iterations=1)
    e_im = cv2.erode(d_im, hkernel, iterations=1)
    d_im = cv2.dilate(e_im, vkernel, iterations=1)
    e_im = cv2.erode(d_im, vkernel, iterations=1)
    bgr = cv2.cvtColor(e_im, cv2.COLOR_GRAY2BGR)
    return bgr

def dilerod2(impath):
    img = cv2.imread(impath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,bw = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    hkernel = np.ones((1,15), np.uint8)  # note this is a horizontal kernel
    vkernel = np.ones((15,1), np.uint8)  # note this is a vertical kernel
    d_im = cv2.dilate(bw, hkernel, iterations=2)
    d_im = cv2.dilate(d_im, vkernel, iterations=2)

    bgr = cv2.cvtColor(d_im, cv2.COLOR_GRAY2BGR)
    return bgr


def hough(impath):
    img = cv2.imread(impath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #edges = cv2.Canny(gray,140,200)
    sobelx = np.uint8(np.absolute(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)))
    sobely = np.uint8(np.absolute(cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)))
    edges = cv2.bitwise_or(sobelx,sobely)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),1)
    return line_image

def connectcommonlines(impath):
    lineimg = cv2.imread(impath)
    gray = cv2.cvtColor(lineimg, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    rowprops = []
    colprops = []

    for rowidx in range(gray.shape[0]):
        row = gray[rowidx]
        propcolored = (sum(row)/150)/len(row) #proportion of row that contains a line
        rowprops.append(propcolored)

    halfH = int(len(rowprops)/2)
    tophalf = rowprops[:halfH]
    bothalf = rowprops[halfH:]
    maxrowstop = np.argpartition(tophalf, -4)[-4:]
    maxrowstop = [min(maxrowstop),max(maxrowstop)]
    maxrowsbot = halfH + np.argpartition(bothalf, -5)[-5:]
    maxrowsbot = [min(maxrowsbot),max(maxrowsbot)]
    print(maxrowstop)
    print(maxrowsbot)

    #print(rowprops[maxrows])
    for colidx in range(gray.shape[1]):
        col = gray[:,colidx]
        propcolored = (sum(col)/150)/len(col) #proportion of row that contains a line
        colprops.append(propcolored)

    halfW = int(len(colprops)/2)
    Lhalf = colprops[:halfW]
    Rhalf = colprops[halfW:]
    maxcolsleft = np.argpartition(Lhalf, -4)[-4:]
    maxcolsleft = [min(maxcolsleft),max(maxcolsleft)]
    maxcolsright = halfW + np.argpartition(Rhalf, -2)[-2:]
    maxcolsright = [min(maxcolsright),max(maxcolsright)]

    print(maxcolsleft)
    print(maxcolsright)

    line_image = np.copy(lineimg) * 0  # creating a blank to draw lines on
    lines = []

    for col in np.concatenate((maxcolsleft,maxcolsright),axis=0):
        lines.append([col,0,col,len(rowprops)])
    for row in np.concatenate((maxrowstop,maxrowsbot),axis=0):
        lines.append([0,row,len(colprops),row])
    for l in lines:
        x1,y1,x2,y2 = l
        cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),1)
    return line_image

def getContours(impath):
    img = cv2.imread(impath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    contours,hierarchy  = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #hull = []
    # calculate points for each contour
    #for i in range(len(contours)):
        # creating convex hull object for each contour
        #hull.append(cv2.convexHull(contours[i], False))
    # create an empty black image
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (0, 255, 0) # green - color for contours
        #color = (255, 0, 0) # blue - color for convex hull
        # draw ith contour
        cv2.drawContours(drawing, contours, i, color_contours, 15, 8, hierarchy)
        # draw ith convex hull object
        #cv2.drawContours(drawing, hull, i, color, -1, 8)
    #print(contours)
    return drawing


imW = 1600
imH = 1600
zoom = 19
cenLat,cenLong = 40.113936, -88.231303
iter = 3
impath = 'aerials'+str(iter)+'/im'+str(iter)+'.png'
croppath = 'aerials'+str(iter)+'/crop'+str(iter)+'.png'
threshpath = 'aerials'+str(iter)+'/thresh'+str(iter)+'.png'
comppath = 'aerials'+str(iter)+'/comppath'+str(iter)+'.png'
complinespath = 'aerials'+str(iter)+'/complines'+str(iter)+'.png'
hpath = 'aerials'+str(iter)+'/hough'+str(iter)+'.png'
hlinespath = 'aerials'+str(iter)+'/houghlines'+str(iter)+'.png'
contpath = 'aerials'+str(iter)+'/contpath'+str(iter)+'.png'
contlinespath = 'aerials'+str(iter)+'/contlines'+str(iter)+'.png'
maskedpath = 'aerials'+str(iter)+'/MASKED'+str(iter)+'.png'
im = getImage(cenLat,cenLong,19,api_key,[],imW,imH)
im.save(impath)

#crop the image to the block that we care about
img = cv2.imread(impath)
crop_img = img[247:1316, 113:1546]
cv2.imwrite(croppath,crop_img)

#threshold the image to give us
thresh = threshold(croppath)
cv2.imwrite(threshpath,thresh)

de = dilerod('aerials2/kmeansmask.png')
cv2.imwrite('aerials3/dilerosion.png',de)

hlines = hough('aerials3/dilerosion.png')
cv2.imwrite(hlinespath,hlines)

contlines = getContours(hlinespath)
cv2.imwrite(contlinespath,contlines)

comp = connectcommonlines(hlinespath)
cv2.imwrite(complinespath,comp)

compdilate = dilerod2(complinespath)
gray = cv2.cvtColor(de, cv2.COLOR_BGR2GRAY)
ret,bw = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
safetygreen = np.zeros((bw.shape[0],bw.shape[1],3),'uint8')
safetygreen[...,1] = bw
cv2.imwrite('aerials3/safetymargin.png',compdilate)

comp = cv2.addWeighted(crop_img,0.8,comp,1,0)
h = cv2.addWeighted(crop_img, 0.8, contlines, 0.5, 0)
masked = cv2.addWeighted(crop_img,0.8,compdilate,0.8,0)
masked = cv2.addWeighted(masked,0.2,safetygreen,1,0)
cv2.imwrite(hpath,h)
cv2.imwrite(comppath,comp)
cv2.imwrite(maskedpath,masked)
safety = cv2.addWeighted(de,0.8,safetygreen,0.5,0)
cv2.imwrite('safety.png',safety)
cv2.destroyAllWindows()
