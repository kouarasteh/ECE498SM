#from __future__ import print_function
import cv2
import csv
import urllib3
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
import math

imW = 1600
imH = 1600

def crop2origPIX(x,y):
    return(x+113,y+247)
    #crop_img = img[247:1316, 113:1546]

def orig2cropPIX(x,y):
    if y<247 or y>1315 or x<113 or x>1545:
        return (-1,-1)
    else:
        return(x-113,y-247)


def getPointLatLng(x, y, zoom=19, cenLat=40.113936, cenLong=-88.231303):
    parallelMultiplier = math.cos(cenLat * (math.pi / 180))
    degreesPerPixelX = (360 / (2**(zoom + 8)))
    degreesPerPixelY = ((360 / (2**(zoom + 8))) * parallelMultiplier)
    pointLat = (cenLat - (degreesPerPixelY * ( y - imH / 2)))
    pointLng = (cenLong + (degreesPerPixelX * ( x  - imW / 2)))
    return (pointLat, pointLng)

def getLatLngPoint(lat,long,zoom=19,cenLat=40.113936,cenLong=-88.231303):
    parallelMultiplier = math.cos(cenLat * (math.pi / 180))
    degreesPerPixelX = (360 / 2**(zoom + 8))
    degreesPerPixelY = ((360 / 2**(zoom + 8)) * parallelMultiplier)
    y = int(( ((cenLat - lat)/degreesPerPixelY) +(imH/2)))
    x = int(( ((long - cenLong)/degreesPerPixelX) +(imW/2)))
    return x,y

def islatlonglegal(lat,long,zoom,cenLat,cenLong,mask):
    x,y = getLatLngPoint(lat,long,zoom,cenLat,cenLong)
    if x<0 or x>=1600 or y<0 or y>1600:
        print("out of bounds")
        return False
    else:
        print("in bounds")
        cropx,cropy = orig2cropPIX(x,y)
        return (mask[cropy][cropx] == 255)


zoom = 19
cenLat,cenLong = 40.113936, -88.231303

OFFSET = 268435456
RADIUS = (OFFSET/math.pi)
def get_pixel(x, y, x_center, y_center, zoom_level):
    """
    x, y - location in degrees
    x_center, y_center - center of the map in degrees (same value as in the google static maps URL)
    zoom_level - same value as in the google static maps URL
    x_ret, y_ret - position of x, y in pixels relative to the center of the bitmap
    """
    x_ret = (l_to_x(x) - l_to_x(x_center)) >> (21 - zoom_level)
    y_ret = (l_to_y(y) - l_to_y(y_center)) >> (21 - zoom_level)
    return (800 + x_ret),(800 + y_ret)

def l_to_x(x):
    return int(round(OFFSET + RADIUS * x * math.pi / 180))

def l_to_y(y):
    return int(round(OFFSET - RADIUS * math.log((1 + math.sin(y * math.pi / 180)) / (1 - math.sin(y * math.pi / 180))) / 2))

mask = cv2.imread('aerials3/dilerosion.png')
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)



Y,X =  40.114356, -88.231898

x,y = get_pixel(X,Y,cenLong,cenLat,19)
print(orig2cropPIX(x,y))
#142,54
#print(islatlonglegal(cenLat,cenLong,zoom,cenLat,cenLong,mask))
#40.114356, -88.231898

# lat,long = getPointLatLng(255,301,19,40.113936, -88.231303)
# print(lat,long)
# x,y = getLatLngPoint(lat,long,19,40.113936, -88.231303)
# print(x,y)
# print(orig2cropPIX(x,y))

m = cv2.imread('aerials1/crop1.png')
plt.imshow(m)
plt.show()
# islegal = set()
# for rowidx in range(mask.shape[0]):
#     for colidx in range(mask.shape[1]):
#         origx,origy = crop2origPIX(colidx,rowidx)
#         if origx != -1:
#             if ((mask[rowidx][colidx]== 255)):
#                 plat,plong = getPointLatLng(origx,origy,zoom,cenLat,cenLong)
#                 print(plat,plong)
#                 islegal.add((plat,plong))
