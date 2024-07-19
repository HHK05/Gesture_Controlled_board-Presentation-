import cv2
import os
import warnings
from cvzone.HandTrackingModule import HandDetector
import numpy as np
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress TensorFlow warnings
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except ImportError:
    pass

# Variables
width, height = 1280, 720
folderpath = "C:\\Users\\Harsh\\OneDrive\\Desktop\\hackathin\\Gesture Presentation\\Presentation"

# Camera Setup 
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Get the list of images that you need to traverse 
pathimages = sorted(os.listdir(folderpath), key=len)
print(pathimages)

# Variable
imgnumber = 0
hs, ws = int(120 * 1), int(213 * 1)  # Height and width of the small camera feed
gesturethrshold=500
buttonpressed=False
buttoncounter=0
buttondelay=10
annotation=[[]]
annotationumber=-1
annotationstart=False

detector=HandDetector(detectionCon=0.8,maxHands=1)

#  margin for the camera feed
top_margin = 30
right_margin = 20

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  
    pathfullimage = os.path.join(folderpath, pathimages[imgnumber])
    imgcurrent = cv2.imread(pathfullimage)

    hands,img=detector.findHands(img)

    cv2.line(img,(0,gesturethrshold),(width,gesturethrshold),(0,255,0),10)

    if hands and buttonpressed is False:
        hand=hands[0]
        fingers=detector.fingersUp(hand)

        cx,cy=hand['center']
        lmList = hand["lmList"]
        #constraint limit 
        indexfinger=lmList[8][0],lmList[8][1]
        xval = int(np.interp(lmList[8][0],[0,width],[0, width]))
        yval = int(np.interp(lmList[8][1],[0,height],[0, height]))
        indexfinger = xval, yval

        #print(fingers)
        if cy<=gesturethrshold:
            #Gesture - 1 print left 
            if fingers==[1,0,0,0,0]:
                print("Left")
                if imgnumber>0:
                    buttonpressed = True
                    annotation=[[]]
                    annotationumber=-1
                    annotationstart=False
                    imgnumber-=1

            #Gesture - 2 print right 
            if fingers==[0,0,0,0,1]:
                print("Right")
                if imgnumber<len(pathimages)-1:
                    buttonpressed = True
                    annotation=[[]]
                    annotationumber=-1
                    annotationstart=False
                    imgnumber+=1

        #Gesture - 3 pointing
        if fingers==[0,1,1,0,0]:
            cv2.circle(imgcurrent,indexfinger,12,(0,0,255),cv2.FILLED)
    
        #Gesture - 4 drawing
        if fingers==[0,1,0,0,0]:
            if annotationstart is False:
                annotationstart = True
                annotationumber+=1
                annotation.append([])
            cv2.circle(imgcurrent,indexfinger,12,(0,0,255),cv2.FILLED)
            annotation[annotationumber].append(indexfinger)
        else:
            annotationstart=False

        #gesture 5 erase 
        if fingers==[0,1,1,1,0]:
            if annotation:
                annotation.pop()
                annotationumber-=1
                buttonpressed=True
                

    #buttonpressed iterations 
    if buttonpressed:
        buttoncounter+=1
        if buttoncounter>buttondelay:
            buttoncounter=0
            buttonpressed = False

    for i in range (len(annotation)):
        for j in range (len(annotation[i])):
            if j!=0:
                cv2.line(imgcurrent,annotation[i][j-1],annotation[i][j],(0,0,200),12)

    # Resize  of current image
    imgcurrent = cv2.resize(imgcurrent, (width, height))

    # add slide to the camera scrren the image
    imgsmall = cv2.resize(img, (ws, hs))
    
    # Calculate position for top-right placement
    y1, y2 = top_margin, top_margin + hs
    x1, x2 = width - ws - right_margin, width - right_margin
    
    # Place the camera feed in the top-right corner
    imgcurrent[y1:y2, x1:x2] = imgsmall

    cv2.imshow('Slides', imgcurrent)
    cv2.imshow("Image",img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
