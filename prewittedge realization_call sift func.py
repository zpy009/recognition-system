import numpy as np
import cv2
import copy
import math

def myprewittedge(Im,T,direction):
    if T is None:
        T = (float(np.max(Im)) + float(np.min(Im)))/2
        G1, G2 = copy.deepcopy(Im), copy.deepcopy(Im)
        for i in range(10):
            G1 = np.where(Im >= T, Im, 0)
            G2 = np.where(Im < T, Im, 0)
            m1 = np.sum(G1)/np.sum(G1 != 0)
            m2 = np.sum(G2)/np.sum(G2 != 0)
            T = (m1 + m2) / 2

    Im = Im.astype(np.int16)
    prewittNum, new_Im = 0, np.zeros((Im.shape[0],Im.shape[1]),np.uint8)
    for j in range(1,Im.shape[0]-1):
        for k in range(1,Im.shape[1]-1):
            prewittNum = max(abs(Im[j][k+1]+Im[j+1][k+1]+Im[j+1][k]-Im[j-1][k]-Im[j-1][k-1]-Im[j][k-1]), abs(Im[j-1][k]+Im[j-1][k+1]+Im[j][k+1]-Im[j][k-1]-Im[j+1][k-1]-Im[j+1][k]), abs(Im[j-1][k+1]+Im[j-1][k]+Im[j-1][k-1]-Im[j+1][k+1]-Im[j+1][k]-Im[j+1][k-1]), abs(Im[j-1][k+1]+Im[j][k+1]+Im[j+1][k+1]-Im[j-1][k-1]-Im[j][k-1]-Im[j+1][k-1]))
            if prewittNum > T:
                new_Im[j][k] = 255
    return new_Im

def mylineextraction(f):
    img = cv2.imread('fig.tif')
    max_line_value, max_line_index = 0, []
    lines = cv2.HoughLinesP(f,1,np.pi/180,threshold=30, minLineLength=350, maxLineGap=12)
    for line in lines:
        if (math.sqrt(math.pow(line[0][2]-line[0][0],2)+math.pow(line[0][3]-line[0][1],2)) > max_line_value):
            max_line_value = math.sqrt(math.pow(line[0][2]-line[0][0],2)+math.pow(line[0][3]-line[0][1],2))
            max_line_index = line[0]
    cv2.line(img,(max_line_index[0],max_line_index[1]),(max_line_index[2],max_line_index[3]),(255,0,0),2)
    return img

# task 1
Im = cv2.imread('fig.tif',0)
cv2.imshow('image01',Im)
k = cv2.waitKey(0)
if k == 27:         # 按下esc时，退出
    cv2.destroyAllWindows()
cv2.imwrite('01original.png',Im)
print('Original image is read and displayed successfully.')

# task 2
T = float(np.max(Im))*0.2
direction = 'all'
g = myprewittedge(Im,T,direction)
cv2.imshow('image02',g)
k = cv2.waitKey(0)
if k == 27:         # 按下esc时，退出
    cv2.destroyAllWindows()
cv2.imwrite('02boundary1.png',g)
print('The corresponding binary edge image is computed and displayed successfully.')

# task 3
direction = 'all'
f = myprewittedge(Im,None,direction)
cv2.imshow('image03',f)
k = cv2.waitKey(0)
if k == 27:         # 按下esc时，退出
    cv2.destroyAllWindows()
cv2.imwrite('03boundary2.png',f)
print('The corresponding binary edge image is computed and displayed successfully.')

# task 4
img = mylineextraction(f)
cv2.imshow('image04',img)
k = cv2.waitKey(0)
if k == 27:         # 按下esc时，退出
    cv2.destroyAllWindows()
cv2.imwrite( '04longestline.png',img)
print('The longest line in image is computed and displayed successfully.')

# task 5
I = cv2.imread('QR-Code.png',0)
I1 = cv2.imread('image1.png',0)
I2 = cv2.imread('image2.png',0)
I3 = cv2.imread('image3.png',0)

sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(I, None)
kp1, des1 = sift.detectAndCompute(I1, None)
kp2, des2 = sift.detectAndCompute(I2, None)
kp3, des3 = sift.detectAndCompute(I3, None)

bf = cv2.BFMatcher()
matches1 = bf.knnMatch(des, des1, k=2)
matches2 = bf.knnMatch(des, des2, k=2)
matches3 = bf.knnMatch(des, des3, k=2)

good = [[], [], []]
good[0] = [[m] for m, n in matches1 if m.distance < 0.9 * n.distance]
good[1] = [[m] for m, n in matches2 if m.distance < 0.6 * n.distance]
good[2] = [[m] for m, n in matches3 if m.distance < 0.6 * n.distance]

img1 = cv2.drawMatchesKnn(I, kp, I1, kp1, good[0], None, flags=2)
img2 = cv2.drawMatchesKnn(I, kp, I2, kp2, good[1], None, flags=2)
img3 = cv2.drawMatchesKnn(I, kp, I3, kp3, good[2], None, flags=2)

if len(good[0]) > len(good[1]):
    if len(good[2]) > len(good[0]):
        optimal_index = 2
    else:
        optimal_index = 0
else:
    if len(good[2]) > len(good[1]):
        optimal_index = 2
    else:
        optimal_index = 1

cv2.imshow('image1',img1)
cv2.imshow('image2',img2)
cv2.imshow('image3',img3)


cv2.imwrite( '06QR_img2.png',img2)
cv2.imwrite( '07QR_img3.png',img3)

print('image'+str(optimal_index+1)+' matches “QR-Code.png” best')
input()