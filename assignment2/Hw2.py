from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from Hw2_ui import Ui_MainWindow
from tkinter import filedialog
from tkinter import *
from torchvision import models, datasets
import torchvision.transforms as transform
from torchsummary import summary
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet50
from vggmodeltest import VGG19_BN
import torch.nn as nn
import torch.nn.functional as F
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

        self.video_path = []
        self.img_path = []
        self.folder_path = []
        self.frame = []

        #Load data
        self.ui.LoadImage.clicked.connect(self.LoadFolder)
        self.ui.LoadVideo.clicked.connect(self.LoadVideo)

        #1
        self.ui.BackgroundSubtraction.clicked.connect(self.BackgroundSubtraction)

        #2
        self.ui.Preprocessing.clicked.connect(self.Preprocessing)
        self.ui.VideoTracking.clicked.connect(self.VideoTracking)

        #3
        self.ui.DimentionReduction.clicked.connect(self.DimentionReduction)

        #4
        self.ui.ShowModelStructure.clicked.connect(self.ShowModelStructure)
        self.ui.ShowAccandLoss.clicked.connect(self.ShowAccAndLoss)
        self.ui.Predict.clicked.connect(self.predict)
        self.ui.Reset.clicked.connect(self.reset)

        ######
        # Get references to the controls from Qt Designer
        self.graphics_view = self.ui.graphicsView
 
        # Set up drawing-related properties
        self.drawing_pen = QtGui.QPen(QtCore.Qt.white, 10, QtCore.Qt.SolidLine)

        #####

        self.scene=QGraphicsScene(self)
        self.ui.graphicsView.setScene(self.scene)
  
        self.ui.graphicsView.setStyleSheet("background-color: black;")
    
        self.ui.graphicsView.mousePressEvent = self.mousePressEvent
        self.ui.graphicsView.mouseMoveEvent = self.mouseMoveEvent

        self.ui.graphicsView.show()
        
        #5
        self.ui.LoadImage2.clicked.connect(self.LoadImage2)
        self.ui.ShowModelStructure2.clicked.connect(self.ShowModelStructure2)
        self.ui.ShowImages.clicked.connect(self.ShowImages)
        self.ui.ShowComparison.clicked.connect(self.ShowComparison)
        self.ui.Inference.clicked.connect(self.Inference)


    #Load Folder
    def LoadImage(self):
        self.image_path = filedialog.askopenfilename(initialdir = "/",title = "開啟檔案")
        print(self.image_path)
        self.ui.label.setText("")

    def LoadVideo(self):
        self.video_path = filedialog.askopenfilename(initialdir = "/",title = "開啟檔案")
        print(self.video_path)
        self.ui.label2.setText("")

    def LoadFolder(self):
        self.folder_path = filedialog.askdirectory()
        print(self.folder_path)
        self.ui.label.setText("")

    #1. Background Subtraction
    def BackgroundSubtraction(self):
        cap = cv2.VideoCapture(self.video_path)
        # knn = cv2.createBackgroundSubtractorKNN(detectShadows=True)

        f_array = []
        create_model = False
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video = cv2.VideoWriter("output.avi",fourcc, fps, (width*3, height))

        while cap.isOpened():
            r, f = cap.read()
            if r:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                mask = np.zeros_like(gray)

                if len(f_array) < 25:
                    f_array.append(gray)
                else:
                    if not create_model:
                        f_array = np.array(f_array)
                        m = np.mean(f_array, axis= 0)
                        std = np.std(f_array, axis=0)
                        std[std < 5] = 5
                        create_model = True
                    else:
                        mask[np.abs(gray - m) > std*5] = 255

                foreground = cv2.bitwise_and(f, f, mask= mask)
                mask_o = np.zeros_like(f)
                mask_o[:,:,0] = mask
                mask_o[:,:,1] = mask
                mask_o[:,:,2] = mask

                o = cv2.hconcat([f, mask_o, foreground])
                video.write(o)
                cv2.imshow("Result", o)
                cv2.waitKey(30)
            else:
                break
        cap.release()
        video.release()
        cv2.destroyAllWindows()

    #2. Optical Flow
    #2.1 Preprocessing
    def Preprocessing(self):
        self.cap = cv2.VideoCapture(self.video_path)
        r, self.o_f = self.cap.read()
        self.o_gray = cv2.cvtColor(self.o_f, cv2.COLOR_BGR2GRAY)

        corners = cv2.goodFeaturesToTrack(self.o_gray,1,0.3,7,7)
        corners = np.int0(corners)

        for i in corners:
            x,y = i.ravel()
            cv2.circle(self.o_f,(x,y),3,255,-1)
            cv2.line(self.o_f,(x-20,y),(x+20,y), (0,0,255), 4)
            cv2.line(self.o_f,(x,y-20),(x,y+20), (0,0,255), 4)
 
        cv2.imshow("Circle detect", self.o_f)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #2.2 Video tracking
    def VideoTracking(self):
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 1,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        color = (0,100,255)
        color1 = (0, 0, 255)

        cap = cv2.VideoCapture(self.video_path)
        r, frame = cap.read()
        o_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(o_gray, mask = None, **feature_params)
        mask = np.zeros_like(frame)

        while True:
            ret,frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(o_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]

            # draw the tracks
            a,b = good_new.ravel()
            c,d = good_old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color, 4)
            frame = cv2.line(frame,(int(a), int(b)-20),(int(a), int(b)+20),color1, 4 )
            frame = cv2.line(frame,(int(a)-20, int(b)),(int(a)+20, int(b)),color1, 4 )

            img = cv2.add(frame,mask)
            cv2.imshow('frame', img)
            cv2.waitKey(1)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            # Now update the previous frame and previous points
            o_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
        
        cap.release()
        cv2.destroyAllWindows()

    #3. Dimention Reduction
    def DimentionReduction(self): 
        images = glob.glob(self.folder_path+ "/*.jpg")
        error = []
        for image in images:
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cv2.imshow('Original Image', img)
            cv2.imshow('Gray Image', gray)

            df_gray = gray/255

            pca_gray = PCA(n_components = 340)
            pca_gray.fit(df_gray)
            trans_pca_gray = pca_gray.transform(df_gray)

            gray_arr = pca_gray.inverse_transform(trans_pca_gray)
            
            cv2.imshow('Reconstruct Image', gray_arr)
            R_img = np.reshape(img,(350,1050)) 
            ipca = PCA(340).fit(R_img) 
            C_img = ipca.transform(R_img) 
            temp = ipca.inverse_transform(C_img) 
            temp = np.reshape(temp, (350,350,3))

            result = np.square(img - temp )
            result = np.sqrt(np.sum(result))
            error.append(result)
           
            print("n = 340")
            print(error)
            

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    #4. Training MNIST classifier Using VGG19
    #4.1 Show Model Structure
    def ShowModelStructure(self):
        model = models.vgg19_bn(num_classes = 10)
        model.features[0] = nn.Conv2d(1, 64, kernel_size = 3, stride = 1, padding = 1)
        summary(model,(1, 32, 32))

    #4.2 Show accuracy and loss
    def ShowAccAndLoss(self):
        img = cv2.imread('train_val_accuracy_loss.png')
        img = cv2.resize(img, (520, 208))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y, x = img.shape[:-1]
        frame = QImage(img, x, y, QImage.Format_RGB888)
        self.scene.clear()
        self.pix = QPixmap.fromImage(frame)
        self.scene.addPixmap(self.pix)

    def mousePressEvent(self, event):
        # Handle mouse press event
        if event.button() == QtCore.Qt.LeftButton:
            self.start_point = event.pos()
    
    def mouseMoveEvent(self, event):
        # Handle mouse move event for drawing
        if hasattr(self, 'start_point'):
            end_point = event.pos()
            line = QtCore.QLineF(self.start_point, end_point)
            self.scene.addLine(line, self.drawing_pen)
            self.start_point = end_point
   
    def predict(self):
        """Load the image and inference them with pretrained weight"""
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        weight_path = './vgg19_test.pth'
        model = VGG19_BN()
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        model.eval()
        transform_test = transform.Compose([
            transform.ToPILImage(),
            transform.Resize((32, 32)),
            transform.ToTensor(),
        ])
        pixmap = self.graphics_view.grab()
        img_path = './draw.png'
        pixmap.save(img_path, "PNG")
        image = cv2.imread(img_path)
        print(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_tensor = transform_test(image)
        image_tensor = image_tensor.unsqueeze(0)
        outputs = model(image_tensor)
        outputs = F.softmax(outputs[0], dim=0)
        probabilities = outputs.detach().numpy()
        predict = classes[np.argmax(probabilities)]
        print(probabilities)
        text = 'Predicted class = '+predict
        self.ui.label.setText(text)
       
        plt.bar(classes, probabilities)
        plt.xlabel('Classes')
        plt.ylabel('Probabilities')
        plt.title('Probability of each class')
        plt.show()
    
 
    def reset(self):
        # Clear the canvas by removing all items from QGraphicsScene
        self.scene.clear()
        # self.ui.graphicsView.scene().clear()
        # self.ui.graphicsView.setScene(DrawingScene())
        self.ui.label.setText('Predicted class =')

    #5
    # Load Image
    def LoadImage2(self):
        img_path = filedialog.askopenfilename(initialdir = "/",title = "開啟檔案")
        print(img_path)

        self.img2 = cv2.imread(img_path)
        self.img = Image.open(img_path)
        self.img2 = cv2.resize(self.img2, (224, 224))

        x = self.img2.shape[1]
        y = self.img2.shape[0]

        im_rgb = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)
        frame = QtGui.QImage(im_rgb, x, y, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(frame)
        self.item = QtWidgets.QGraphicsPixmapItem(pix)
        self.scene2 = QtWidgets.QGraphicsScene() 
        self.scene2.addItem(self.item)
        self.ui.graphicsView2.setScene(self.scene2)
        self.ui.graphicsView2.show()

    #Show Images
    def ShowImages(self):
        cat = cv2.imread('./inference_dataset/cat/8043.jpg')
        cat = cv2.resize(cat, (224, 224), interpolation=cv2.INTER_AREA)
        dog = cv2.imread('./inference_dataset/dog/12051.jpg')
        dog = cv2.resize(dog, (224, 224), interpolation=cv2.INTER_AREA)
        cat =cv2.cvtColor(cat, cv2.COLOR_BGR2RGB)  
        dog =cv2.cvtColor(dog, cv2.COLOR_BGR2RGB)  
        plt.figure()
 
        
        plt.subplot(1, 2, 1)
        plt.title('cat')
        plt.imshow(cat), plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title('dog')
        plt.imshow(dog), plt.axis('off')

        plt.show()


    #5.2 Show Model Structure
    def ShowModelStructure2(self):
        model = models.resnet50()
        #freeze all params
        for params in model.parameters():
            params.requires_grad_ = False

        #add a new final layer
        nr_filters = model.fc.in_features  #number of input features of last layer
        model.fc = nn.Sequential(
            nn.Linear(nr_filters, 1),
            nn.Sigmoid())
        summary(model,(3, 224, 224))

    def ShowComparison(self):
        img = cv2.imread('./accuracy_comparison.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        plt.imshow(img)
        plt.show()

    #5.4 Inference
    def Inference(self):
        classes = ['Cat', 'Dog']
        weight_path = './best_model_with_erasing.pth'
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
        nn.Linear(2048, 1),
        nn.Sigmoid()
        )
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        model.eval()
        transform_test = transform.Compose([
        transform.Resize((224, 224)),
        transform.ToTensor(),
        ])
        image = self.img.convert('RGB')
        image_tensor = transform_test(image)
        image_tensor = image_tensor.unsqueeze(0)
        output = model(image_tensor)
        output = output.squeeze(1)
        if output < 0.5:
            predict = 'Cat'
        elif output >= 0.5:
            predict = 'Dog'
        #print(predict)
        text = 'predict = '+predict
        self.ui.label2.setText(text)


        

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
