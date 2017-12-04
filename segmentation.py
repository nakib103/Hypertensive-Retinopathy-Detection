import cv2, os
import numpy as np
from PIL import Image

imgDir = "E:\\library of EEE\\4-2\\eee 426\\code\\dataDRIVE\\"
imgFilenames = [f for f in os.listdir(imgDir) if f.lower().endswith("tif")]
o
for imgFilenameIndex,imgFilename in enumerate(imgFilenames):
    if imgFilenameIndex > 0:
        break
    imgPath = os.path.join(imgDir, imgFilename)
    
    image = cv2.imread(imgPath)
    cv2.imshow('original image', image)
    cv2.imwrite('C:\\Users\\Snakib\\Desktop\\picsbiomed\\orgimage.tiff',image)
    
    ############################ preprocessing ####################################
    # taking the greeen channel
    r,imageGreen,b = cv2.split(image)
    cv2.imshow('green channel image', imageGreen)
    cv2.imwrite('C:\\Users\\Snakib\\Desktop\\picsbiomed\\greenchannel.tiff',imageGreen)
    #cv2.imshow('red channel image', r)
    #cv2.imshow('blue channel image', b)
    #cv2.imshow('gray scale channel image', cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
    
    # should do histogram matching here to counter different brightness and contrast
    
    # appplying contrast limited adaptive histogram equalisation
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
    imageEqualized = clahe.apply(imageGreen)
    cv2.imshow('histogram equalized image', imageEqualized)
    cv2.imwrite('C:\\Users\\Snakib\\Desktop\\picsbiomed\\equalized.tiff',imageEqualized)
    
    # inversion 
    imageInv2 = 255 - imageEqualized
    imageInv = clahe.apply(imageInv2)
    cv2.imshow('inverted image', imageInv)
    cv2.imwrite('C:\\Users\\Snakib\\Desktop\\picsbiomed\\inverted.tiff',imageInv)    
    cv2.imwrite('C:\\Users\\Snakib\\Desktop\\inverted.tiff',imageInv)
    
    # median filter and subtraction to remove backeground did not work well
    #imageMed = cv2.medianBlur(imageInv, 33)
    #imageBackElm = imageInv - imageMed
    #cv2.imshow('background eliminated image', imageBackElm)
    
    # median filtering noise elimination
    kernel = np.ones((9,9),np.uint8)
    imageMed = cv2.medianBlur(imageInv, 5)
    cv2.imshow('median filtered image',imageMed)
    
    # top hat to remove background
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    #imageBackElm = cv2.morphologyEx(imageMed, cv2.MORPH_TOPHAT, kernel2)
    imageOpen = cv2.morphologyEx(imageMed, cv2.MORPH_OPEN, kernel2)
    imageBackElm = imageMed - imageOpen    
    cv2.imshow('opened image', imageOpen)
    cv2.imshow('background eliminated image', imageBackElm)
    cv2.imwrite('C:\\Users\\Snakib\\Desktop\\picsbiomed\\backeliminated.tiff',imageBackElm)
    
    # enhancement
    
    # adaptive thresholding
    imagethresh2 = cv2.adaptiveThreshold(imageBackElm,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,13,1)
    cv2.imshow('adaptive thresholded image', imagethresh2)
    
    # area threshholding
    imageCont, contours,hierarchy = cv2.findContours(imagethresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, contourIdx=-1, color=0, thickness=-1)    
    cv2.imshow('contour detected image', image)
    print('length of contours {}'.format(len(contours)))
    
    delete = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 50000.0:
            delete.append(i)
    print(delete)       
    for i,idx in enumerate(delete):
        del contours[idx-i]
    print('length of contours after {}'.format(len(contours)))
    
    cv2.drawContours(imagethresh2, contours, contourIdx=-1, color=0, thickness=-1)    
    cv2.imshow('contour deleted image', imagethresh2) 
    # 2nd itereation
    imageCont, contours,hierarchy = cv2.findContours(imagethresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, contourIdx=-1, color=0, thickness=-1)    
    cv2.imshow('contour detected image 2', image)
    print('length of contours {}'.format(len(contours)))
    
    delete = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 10000.0:
            delete.append(i)
    print(delete)       
    for i,idx in enumerate(delete):
        del contours[idx-i]
    print('length of contours after {}'.format(len(contours)))
    
    cv2.drawContours(imagethresh2, contours, contourIdx=-1, color=0, thickness=-1)    
    cv2.imshow('contour deleted image 2', imagethresh2) 
    cv2.imwrite('C:\\users\\Snakib\\Desktop\\image.tiff',imagethresh2)
    
    # open (did not used)    
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    imageOpen2 = cv2.morphologyEx(imageBackElm, cv2.MORPH_OPEN, kernel3)
    cv2.imshow('opened image 2', imageOpen2)
    
    imageTrain = imagethresh2*imageBackElm
    cv2.imshow('image to be trained', imageTrain)
    
    # make a copy
    img = imageTrain.copy()
    
    # training matrix
    print(img.shape)
    trainingMat = np.array(img.flatten(), np.float32) #error may occur --> then loop
    
    # labels matrix
    grountruthpath = imgPath[:-12] + "manual1.png"
    print(grountruthpath)
    groundTruth = cv2.imread(grountruthpath)
    print(groundTruth.shape)
    _,groundTruthGreen,_ = cv2.split(groundTruth)
    #groundTruthGreen = groundTruth
    cv2.imshow('ground truth green channel',groundTruthGreen)
    cv2.imwrite('C:\\Users\\Snakib\\Desktop\\picsbiomed\\groundtruth.tiff',groundTruthGreen)
    
    print(groundTruth.shape)
    labelsMat = np.array(groundTruthGreen.flatten(), np.int32)
    
        
        
#    ########################## training ###########################################
#    # creating a svm
#    if not os.path.isfile('svm_data.dat'):
#        svm = cv2.ml.SVM_create()
#    else:
#        svm = cv2.ml.SVM_load('svm_data.dat')
#    
#    # parameter
#    #svm_param:
#    #    kernel_type = cv2.SVM_LINEAR,
#    #                    svm_type = cv2.SVM_C_SVC,
#    #                    termCrit = (cv2.TERMCRIT_ITER, 100, 1e-6),
#    #                    C=2.67, 
#    #                    gamma=5.383 )                    
#    svm.setType(cv2.ml.SVM_C_SVC)
#    svm.setKernel(cv2.ml.SVM_RBF)
#    svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
#    svm.setGamma(0.2)
#    
#    # training
#    svm.train(trainingMat, cv2.ml.ROW_SAMPLE, labelsMat)
#    svm.save('svm_data.dat')
#


########################## test data ##########################################

testData = np.array(imageTrain.flatten(), np.float32)
result = svm.predict(testData)   

result2D = np.reshape(result[1],(584,565))
cv2.imshow('result',result2D)


resultprint = Image.new('L', (565,584))
resultprint.putdata(result2D.flatten().tolist())
resultprint.save('C:\\Users\\Snakib\\Desktop\\picsbiomed\\result.tiff')

print(np.count_nonzero(labelsMat))



########################## accuuracy test #####################################

count = 0;
for ind,pix in enumerate(labelsMat):
    if pix == result[1][ind]:
        count += 1
print(count*100.0/423500) #have to change 

count = 0;
for ind,pix in enumerate(labelsMat):
    if pix == result[1][ind] and pix == 255:
        count += 1
print(count*100.0/np.count_nonzero(labelsMat)) #have to change 



## waaaaaaaaaaaait       
cv2.waitKey(0)