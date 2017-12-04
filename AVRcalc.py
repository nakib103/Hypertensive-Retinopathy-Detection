import cv2
import numpy as np
from math import sqrt 
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage.feature import canny

## reading the image
image = cv2.imread('dataDRIVE\\21_training.tif')
_,imggreen,imgred = cv2.split(image)

cv2.imshow('green channel image',imggreen)

# average filtering
kernel = np.ones((31,31))/961
imgavg = cv2.filter2D(imggreen,-1,kernel)
cv2.imshow('averaged image',imgavg)

#max intensity level
hist = cv2.calcHist([imggreen],[0],None,[256],[0,256])
maxVal = -1
maxLoc = [(0,0)]
for x in range(imgavg.shape[0]):
    for y in range(imgavg.shape[1]):
        #print(x,y)
        if imgavg[x][y] >= maxVal:
            maxVal = imgavg[x][y]
            maxLoc[0] = (y,x) 
            
cv2.imwrite('c:\\users\\Snakib\\Desktop\\imgag.tiff',imgavg)
print(maxLoc)
loc = len(maxLoc)
imgtemp = imgavg.copy()
print(imgavg[maxLoc[0][1]][maxLoc[0][0]])
cv2.circle(imgtemp,maxLoc[loc-1],5,(0),-1)
cv2.imshow('localized OD',imgavg)

# thresholding
thresh, imgthresh = cv2.threshold(imgavg,180,255,cv2.THRESH_BINARY)
cv2.imshow('thrshholded image',imgthresh)

# contour
imageCont, contours,hierarchy = cv2.findContours(imgthresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(contours)
cv2.drawContours(imggreen, contours, contourIdx=-1, color=0, thickness=1)    
cv2.imshow('contour detected image 2', imggreen)

# radius
length = len(contours[0])
radius = 0
for i in range(length):
    radius = radius + sqrt((contours[0][i][0][0]-maxLoc[0][0])**2 + (contours[0][i][0][0]-maxLoc[0][0])**2)
radius = int(radius/length)
print(radius)

imgtemp2 = imgavg.copy()
cv2.circle(imgtemp2,maxLoc[loc-1],int(radius),(0),2)
cv2.imshow('Detected OD',imgtemp2)
cv2.imwrite('C:\\users\\Snakib\\Desktop\\imgODdetect.tiff',imgtemp2)

# ROI
ROIrad = radius*4
mask = np.zeros((imgthresh.shape[0],imgthresh.shape[1]), dtype=np.uint8)
cv2.circle(mask,maxLoc[loc-1],ROIrad,(255),-1)
cv2.imshow('mask',mask)
cv2.imwrite('C:\\users\\Snakib\\Desktop\\ROI.tiff',mask)

# detecting artery and vein
mask2 = cv2.imread('C:\\users\\Snakib\\Desktop\\image.tiff')
imgseg =  cv2.imread('C:\\users\\Snakib\\Desktop\\inverted.tiff')
_,mask2,_ = cv2.split(mask2)
cv2.imshow('segmented image',mask2)

print(mask.shape,mask2.shape)
count = 0
intensity = 0
for x in range(mask.shape[0]):
    for y in range(mask.shape[1]):       
        if mask[x][y] and mask2[x][y]:
            intensity = intensity + imgseg[x][y][1]
            count = count + 1
print('intensity sum {}'.format(intensity))       
intensitymean = int(intensity/count)
print('intensity mean {} and count {}'.format(intensitymean,count))
artery = mask.copy()
vein = mask.copy()
cv2.imshow('artery before',artery)                
cv2.imshow('veins before',vein)
for x in range(mask.shape[0]):
    for y in range(mask.shape[1]):
        if mask[x][y] and mask2[x][y]:
            if imgseg[x][y][1] >= intensitymean-15:
                artery[x][y] = 0
            else:
                vein[x][y] = 0
            
cv2.imshow('artery',artery)                
cv2.imshow('veins',vein)
cv2.imshow('whats this',vein-artery)

# arteriovenus ratio

# invert
artery = 255 - artery
vein = 255 - vein
cv2.imshow('inverted artery', artery)
cv2.imshow('inverted vein', vein)
cv2.imwrite('C:\\users\\Snakib\\Desktop\\artery.tiff',artery)
cv2.imwrite('C:\\users\\Snakib\\Desktop\\vein.tiff',vein)

# distance transform
print(artery.dtype)
arteryDist = cv2.distanceTransform(artery,cv2.DIST_L1, cv2.DIST_MASK_PRECISE)
veinDist = cv2.distanceTransform(vein,cv2.DIST_L1, cv2.DIST_MASK_PRECISE)
cv2.imshow('distance transformed artery', arteryDist)
cv2.imshow('distance transformed vein', veinDist)
cv2.imwrite('c:\\users\\Snakib\\Desktop\\veinDist.png', veinDist)


# thinning 
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
size = np.size(arteryDist)
arteryskel = np.zeros(arteryDist.shape,np.uint8)
done = False
while( not done):
    eroded = cv2.erode(arteryDist,kernel)
    temp = cv2.dilate(eroded,kernel)
    temp = cv2.subtract(arteryDist,temp)
    arteryskel = temp.copy()
    arteryDist = eroded.copy()
 
    if cv2.countNonZero(arteryDist) == 0:  
        done = True
print(cv2.countNonZero(arteryskel))
cv2.imshow('thinned image artery',arteryskel)
                       
veinskel = np.zeros(veinDist.shape,np.uint8)
done = False
while( not done):
    eroded = cv2.erode(veinDist,kernel)
    temp = cv2.dilate(eroded,kernel)
    temp = cv2.subtract(veinDist,temp)
    veinskel = temp.copy()
    veinDist = eroded.copy()
 
    if cv2.countNonZero(arteryDist) == 0:  
        done = True

print(cv2.countNonZero(veinskel))
cv2.imshow('thinned image vein',veinskel)

venuole = []
arteriole = []
for x in range(veinskel.shape[0]):
    for y in range(veinskel.shape[1]):
        if veinskel[x][y] > 0:
            venuole.append(veinskel[x][y])
        if arteryskel[x][y] > 0:
            arteriole.append(arteryskel[x][y])

venuole = sorted(venuole)
arteriole = sorted(arteriole)

lenven = len(venuole)
print('length of lenven {} {} {}'.format(lenven,lenven//2,lenven//2-1))
if lenven%2 == 1:
    Wa = venuole[lenven//2]
    if lenven//2 == 0:
        Wb = venuole[0]
    else:
        Wb = venuole[lenven//2 - 1]
else:
    Wa = (venuole[lenven//2 - 1] + venuole[lenven//2])// 2
    Wb = venuole[lenven//2 - 1]
print(Wa,Wb)
CRVE = sqrt(0.72*(Wa**2) + 0.91*(Wb**2) + 450.02)
lenart = len(arteriole)

if lenart%2 == 1:
    Wa = arteriole[lenart//2]
    if lenart//2 == 0:
        Wb = arteriole[0]
    else:
        Wb = arteriole[lenart//2 - 1]
else:
    Wa = (arteriole[lenart//2 - 1] + arteriole[lenart//2])// 2
    Wb = arteriole[lenart//2 - 1]
print(arteriole)
print(Wa,Wb)
#CRAE = sqrt(0.87*(Wa**2) + 1.01*(Wb**2) - .22*Wa*Wb - 10.73)

#artervenratio = CRAE/CRVE
#print('arteriovenous ratio {}'.format(artervenratio))

cv2.waitKey(0)