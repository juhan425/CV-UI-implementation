from PyQt5 import QtWidgets, QtGui, QtCore
from Hw1_ui import Ui_MainWindow
from tkinter import filedialog
from tkinter import *
from torchvision import models, datasets
import torchvision.transforms as transform
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os
import sys
import cv2
import glob
import python_utils as utils
import torch
from PIL import Image 


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        #Load folder
        self.ui.LoadFolder.clicked.connect(self.LoadFolder)
        self.ui.LoadImage_L.clicked.connect(self.LoadImage_L)
        self.ui.LoadImage_R.clicked.connect(self.LoadImage_R) 
        
        #1
        self.ui.FindCorners.clicked.connect(self.FindCorners)
        self.ui.FindIntrinsic.clicked.connect(self.FindIntrinsic)
        self.ui.FindExtrinsic_button.clicked.connect(self.FindExtrinsic)
        self.ui.FindDistortion.clicked.connect(self.FindDistortion)
        self.ui.ShowResult.clicked.connect(self.ShowResult)

        #2
        self.ui.ShowWordsOnBoard.clicked.connect(self.ShowWordsOnBoard)
        self.ui.ShowWordsVertical.clicked.connect(self.ShowWordsVertical)
        
        #3
        self.ui.SteroDisparityMap_button.clicked.connect(self.StereoDisparityMap)

        #4
        self.ui.LoadImage1.clicked.connect(self.LoadImage1)
        self.ui.LoadImage2.clicked.connect(self.LoadImage2) 
        self.ui.Keypoints.clicked.connect(self.Keypoints)
        self.ui.MatchedKeypoint.clicked.connect(self.MatchedKeypoints) 

        #5
        self.ui.LoadImage_VGG19.clicked.connect(self.LoadImage_VGG19)
        self.ui.ShowAgumentedImages.clicked.connect(self.ShowAgumentedImages)
        self.ui.ShowModelStructure.clicked.connect(self.ShowModelStructure)
        self.ui.ShowAccAndLoss.clicked.connect(self.ShowAccAndLoss)
        self.ui.Inference.clicked.connect(self.Inference)

    #Load Folder
    def LoadFolder(self):
        self.LoadFolder = QtWidgets.QFileDialog.getExistingDirectory(self,'開啟資料夾',' D:/')
        print(self.LoadFolder)

    def LoadImage_L(self):
        LoadImage_L = filedialog.askopenfilename(initialdir = "/",title = "開啟檔案")
        print(self.LoadImage_L)
        self.LoadImage_L = cv2.imread(LoadImage_L)

    def LoadImage_R(self):
        LoadImage_R = filedialog.askopenfilename(initialdir = "/",title = "開啟檔案")
        print(self.LoadImage_R)
        self.LoadImage_R = cv2.imread(LoadImage_R)

    def LoadImage1(self):
        LoadImage1 = filedialog.askopenfilename(initialdir = "/",title = "開啟檔案")
        print(LoadImage1)
        self.LoadImage1 = cv2.imread(LoadImage1)

    def LoadImage2(self):
        LoadImage2 = filedialog.askopenfilename(initialdir = "/",title = "開啟檔案")
        print(LoadImage2)
        self.LoadImage2 = cv2.imread(LoadImage2)

    def LoadImage_VGG19(self):
        LoadImage_VGG19 = filedialog.askopenfilename(initialdir = "/",title = "開啟檔案")
        print(self.LoadImage_VGG19)
        self.LoadImage_VGG19 = cv2.imread(LoadImage_VGG19)


    #1. Calibration

    #1.1 Find corners
    def FindCorners(self):

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        a = 11
        b = 8
        objp = np.zeros((b*a,3), np.float32)
        objp[:,:2] = np.mgrid[0:a,0:b].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        images = glob.glob(self.LoadFolder + '/*.bmp')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (a,b), None)
            print(fname, ret)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
                imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (a,b), corners, ret)
                cv2.namedWindow('Find Corners',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Find Corners', 512,512)
                cv2.imshow('Find Corners',img)
                cv2.waitKey(500)


    #1.2 Find intrinsinc
    def FindIntrinsic(self):

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        a = 11
        b = 8
        objp = np.zeros((b*a,3), np.float32)
        objp[:,:2] = np.mgrid[0:a,0:b].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        images = glob.glob(self.LoadFolder + '/*.bmp')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (a,b), None)
            print(fname, ret)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
                imgpoints.append(corners)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            print("Intrinsic:")
            print(mtx)

    #1.3 Find extrinsinc
    def FindExtrinsic(self):
        img_idx = self.ui.comboBox.currentText()
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        a = 11
        b = 8
        objp = np.zeros((b*a,3), np.float32)
        objp[:,:2] = np.mgrid[0:a,0:b].T.reshape(-1,2)
        filename = self.LoadFolder + '/'+ img_idx +'.bmp'

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        img = cv2.imread(filename)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (a,b), None)

            # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
            imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        rotation_mat = np.zeros(shape=(3, 3))
        cv2.Rodrigues(rvecs[0], rotation_mat)
    
        Rt_matirx = np.concatenate((rotation_mat, tvecs[0]), axis=1)
        print("Extrinsic:%d" %int(img_idx))
        print(Rt_matirx)

    #1.4 Find distortion
    def FindDistortion(self):

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        a = 11
        b = 8
        objp = np.zeros((b*a,3), np.float32)
        objp[:,:2] = np.mgrid[0:a,0:b].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        images = glob.glob(self.LoadFolder + '/*.bmp')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (a,b), None)
            print(fname, ret)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
                imgpoints.append(corners)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            print("Distortion:")
            print(dist)


    #1.5 Show result
    def ShowResult(self):

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        a = 11
        b = 8
        objp = np.zeros((b*a,3), np.float32)
        objp[:,:2] = np.mgrid[0:a,0:b].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        images = glob.glob(self.LoadFolder + '/*.bmp')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (a,b), None)
            print(fname, ret)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
                imgpoints.append(corners)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            h,  w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

            # undistort
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
        
            # Draw and display the corners
            cv2.namedWindow('Distortion',0)
            cv2.resizeWindow('Distortion', 512,512)
            cv2.imshow('Distortion',img)
            cv2.namedWindow('Undistortion',0)
            cv2.resizeWindow('Undistortion', 512,512)
            cv2.imshow('Undistortion',dst)
            cv2.waitKey(500)

    #2. Augmented reality
    #2.1 Show words on board
    def ShowWordsOnBoard(self):
        corners_start = [[7,5,0],[4,5,0],[1,5,0],[7,2,0],[4,2,0],[1,2,0]]
        input = self.ui.textEdit.toPlainText()
        images = glob.glob(self.LoadFolder + '/*.bmp')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11,8),None)

            objpoints = []
            imgpoints = []
            ob_temp = np.zeros((8*11,3), np.float32)
            ob_temp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
            objpoints.append(ob_temp)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria) 
            imgpoints.append(corners)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)  
            _, rvec, tvec = cv2.solvePnP(ob_temp, corners, mtx, dist)
            
            for j in range(len(input)):
                fs = cv2.FileStorage(self.LoadFolder + '/Q2_lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
                charactor = fs.getNode(input[j]).mat()
                for k in range(len(charactor)):
                    start_point = charactor[k][0]
                    end_point = charactor[k][1]

                    start = np.array([corners_start[j][0]+start_point[0],corners_start[j][1]+start_point[1],corners_start[j][2]+start_point[2]],np.float64)
                    start_p,_ = cv2.projectPoints(start,rvec,tvec,mtx,dist,None)
                    start_p=tuple(start_p[0].ravel())
                    start_p = int(start_p[0]),int(start_p[1])

                    end_p = np.array([corners_start[j][0]+end_point[0],corners_start[j][1]+end_point[1],corners_start[j][2]+end_point[2]],np.float64)
                    end_p,_ = cv2.projectPoints(end_p,rvec,tvec,mtx,dist,None)
                    end_p=tuple(end_p[0].ravel())
                    end_p = int(end_p[0]),int(end_p[1])
                    cv2.line(img,start_p,end_p,(0,0,255),10)
                    
            cv2.namedWindow("Show Words On Board", 0)
            cv2.resizeWindow('Show Words On Board', 512,512)
            cv2.imshow('Show Words On Board',img)
            cv2.waitKey(1000)

    #2.2 Show words vertical
    def ShowWordsVertical(self):
        corners_start = [[7,5,0],[4,5,0],[1,5,0],[7,2,0],[4,2,0],[1,2,0]]
        input = self.ui.textEdit.toPlainText()
        images = glob.glob(self.LoadFolder + '/*.bmp')

        for i in range(len(images)):
            img = cv2.imread(images[i])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11,8),None)

            objpoints = []
            imgpoints = []
            ob_temp = np.zeros((8*11,3), np.float32)
            ob_temp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
            objpoints.append(ob_temp)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria) 
            imgpoints.append(corners)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)  
            _, rvec, tvec = cv2.solvePnP(ob_temp, corners, mtx, dist)
          
            
            for j in range(len(input)):
                fs = cv2.FileStorage(self.LoadFolder + '/Q2_lib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
                charactor = fs.getNode(input[j]).mat()
                for k in range(len(charactor)):
                    start_point = charactor[k][0]
                    end_point = charactor[k][1]

                    start = np.array([corners_start[j][0]+start_point[0],corners_start[j][1]+start_point[1],corners_start[j][2]+start_point[2]],np.float64)
                    start_p,_ = cv2.projectPoints(start,rvec,tvec,mtx,dist,None)
                    start_p=tuple(start_p[0].ravel())
                    start_p = int(start_p[0]),int(start_p[1])

                    end = np.array([corners_start[j][0]+end_point[0],corners_start[j][1]+end_point[1],corners_start[j][2]+end_point[2]],np.float64)
                    end_p,_ = cv2.projectPoints(end,rvec,tvec,mtx,dist,None)
                    end_p=tuple(end_p[0].ravel())
                    end_p = int(end_p[0]),int(end_p[1])
                    cv2.line(img,start_p,end_p,(0,0,255),10)
                    
            cv2.namedWindow("Show words vertical", 0)
            cv2.resizeWindow('Show words vertical', 512,512)
            cv2.imshow('Show words vertical',img)
            cv2.waitKey(1000) 



    #3. Stereo disparity map
    def StereoDisparityMap(self):
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25) #256 25 from ppt
        s_LoadImage_L = cv2.cvtColor(self.LoadImage_L,cv2.COLOR_BGR2GRAY)
        s_LoadImage_R = cv2.cvtColor(self.LoadImage_R,cv2.COLOR_BGR2GRAY)
        result_s =  stereo.compute(s_LoadImage_L,s_LoadImage_R)
        result_s = cv2.normalize(result_s, result_s, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.result_StereoDisparity = result_s

        img_l = self.LoadImage_L
        img_r = self.LoadImage_R

        def draw(event,x,y,flags,userdata):
            if event == cv2.EVENT_LBUTTONDOWN:
                img_r2 = img_r.copy()
                disparity = int(self.result_StereoDisparity[y][x])
                print('coordinate: (',x, ',', y,'), disparity: ', disparity)
                cv2.circle(img_r2, (x,y), 15, (0,255,0), -1)  
                cv2.imshow('LoadImage_R', img_r2)

        cv2.namedWindow("LoadImage_L", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow('LoadImage_L', 512,512)
        cv2.imshow('LoadImage_L', img_l)
        cv2.namedWindow("LoadImage_R", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow('LoadImage_R', 512,512)
        cv2.imshow('LoadImage_R', img_r)
        cv2.setMouseCallback('LoadImage_L', draw)
        cv2.namedWindow("Stereo Disparity", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow('Stereo Disparity', 512,512)
        cv2.imshow('Stereo Disparity', result_s)

    #4. SIFT
    #4.1 Keypoints
    def Keypoints(self):
        gray= cv2.cvtColor(self.LoadImage1,cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp = sift.detect(gray,None)
        img=cv2.drawKeypoints(gray,kp,None,color=(0,255,0))
        cv2.namedWindow("LoadImage1", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow('LoadImage1', 512,512)
        cv2.imshow('LoadImage1', self.LoadImage1)
        cv2.namedWindow("Keypoints", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow('Keypoints', 512,512)
        cv2.imshow('Keypoints',img)
           
    #4.2 Matched keypoints
    def MatchedKeypoints(self):
        img1 = cv2.cvtColor(self.LoadImage1,cv2.IMREAD_GRAYSCALE)
        img2 = cv2.cvtColor(self.LoadImage2,cv2.IMREAD_GRAYSCALE)
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.namedWindow("Keypoints", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow('Keypoints', 512,512)
        cv2.imshow('Keypoints',img3)

    #5. VGG19
    #Load Image
    def LoadImage_VGG19(self):
        img_path = filedialog.askopenfilename(initialdir = "/",title = "開啟檔案")
        print(img_path)

        self.img2 = cv2.imread(img_path)
        self.img = Image.open(img_path)
        self.img2 = cv2.resize(self.img2, (128, 128))

        x = self.img2.shape[1]
        y = self.img2.shape[0]

        im_rgb = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)
        frame = QtGui.QImage(im_rgb, x, y, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(frame)
        self.item = QtWidgets.QGraphicsPixmapItem(pix)
        self.scene = QtWidgets.QGraphicsScene() 
        self.scene.addItem(self.item)
        self.ui.InferenceImage.setScene(self.scene)
        self.ui.InferenceImage.show()

    #5.1 Show augmented images
    def ShowAgumentedImages(self):
        LoadFolder = QtWidgets.QFileDialog.getExistingDirectory(self,'開啟資料夾',' D:/')
        print(LoadFolder)
        os.chdir(LoadFolder)
        plt.figure(figsize=(10,10))

        image = Image.open('automobile.png')
        image = transform.RandomHorizontalFlip()(image)
        image = transform.RandomVerticalFlip()(image)
        image = transform.RandomRotation(30)(image)
        plt.subplot(331)     
        plt.imshow(image)                  

        image = Image.open('bird.png')
        image = transform.RandomHorizontalFlip()(image)
        image = transform.RandomVerticalFlip()(image)
        image = transform.RandomRotation(30)(image)
        plt.subplot(332)    
        plt.imshow(image)                

        image = Image.open('cat.png')
        image = transform.RandomHorizontalFlip()(image)
        image = transform.RandomVerticalFlip()(image)
        image = transform.RandomRotation(30)(image)
        plt.subplot(333)    
        plt.imshow(image)                

        image = Image.open('deer.png')
        image = transform.RandomHorizontalFlip()(image)
        image = transform.RandomVerticalFlip()(image)
        image = transform.RandomRotation(30)(image)
        plt.subplot(334)   
        plt.imshow(image) 

        image = Image.open('dog.png')
        image = transform.RandomHorizontalFlip()(image)
        image = transform.RandomVerticalFlip()(image)
        image = transform.RandomRotation(30)(image)
        plt.subplot(335) 
        plt.imshow(image)                  

        image = Image.open('frog.png')
        image = transform.RandomHorizontalFlip()(image)
        image = transform.RandomVerticalFlip()(image)
        image = transform.RandomRotation(30)(image)
        plt.subplot(336)    
        plt.imshow(image)                

        image = Image.open('horse.png')
        image = transform.RandomHorizontalFlip()(image)
        image = transform.RandomVerticalFlip()(image)
        image = transform.RandomRotation(30)(image)
        plt.subplot(337)   
        plt.imshow(image)                

        image = Image.open('ship.png')
        image = transform.RandomHorizontalFlip()(image)
        image = transform.RandomVerticalFlip()(image)
        image = transform.RandomRotation(30)(image)
        plt.subplot(338)  
        plt.imshow(image)        

        image = Image.open('truck.png')
        image = transform.RandomHorizontalFlip()(image)
        image = transform.RandomVerticalFlip()(image)
        image = transform.RandomRotation(30)(image)
        plt.subplot(339)  
        plt.imshow(image)

        plt.show()


    #5.2 Show Model Structure
    def ShowModelStructure(self):
        model = models.vgg19_bn(num_classes = 10)
        summary(model,(3, 32, 32))

    #5.3 Show accuracy and loss
    def ShowAccAndLoss(img):
        img = cv2.imread('train_val_accuracy_loss.png')
        cv2.imshow('Show accuracy and loss', img)

    #5.4 Inference
    def Inference(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        display_transform = transform.Compose([transform.Resize((128, 128))])
        display = display_transform(self.img)
        data_transform = transform.Compose([transform.ToTensor(),])
        self.img = data_transform(self.img)
        self.img = torch.unsqueeze(self.img, dim=0)
        class_indict = {0:'Airplane', 1:'Automobile', 2:'Bird', 3:'Cat', 4:'Deer', 5:'Dog', 6:'Frog', 7:'Horse', 8:'Ship', 9:'Truck'}
        model = models.vgg19_bn(num_classes=10).to(device)
        weights_path = "./best_model.pth"
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        with torch.no_grad():
            output = torch.squeeze(model(self.img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = class_indict[torch.argmax(predict).numpy().item()]
        cls = [class_indict[i] for i in range(10)]
        probabilities = predict.numpy()
        self.ui.label.setText(f'Predicted={predict_cla}')
        qimage = QtGui.QImage(display.tobytes(), display.width, display.height, QtGui.QImage.Format_RGB888)
        qpixmap = QtGui.QPixmap.fromImage(qimage)
        pixmap = QtWidgets.QGraphicsPixmapItem(qpixmap)
        scene = QtWidgets.QGraphicsScene()
        scene.addItem(pixmap)
        self.ui.InferenceImage.setScene(scene)
        plt.bar(cls, probabilities)
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title('Probability of each class')
        plt.show()
        






 

    






        

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
