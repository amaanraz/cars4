from cgitb import grey
import cv2
import numpy as np


# Variables
min_width = 80
min_height = 80
offset = 3 # Allowable error
detect = []
cars = 0

# Start Camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
cap.set(3,720)
cap.set(4,480)


# Init algorithim from open cv
algorithm = cv2.bgsegm.createBackgroundSubtractorMOG()

# Center Circle
def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy



while True:
    ret,frame1 = cap.read()
    # Convert to greyscale
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3,),5)
    # Apply filter on each frame
    img_sub = algorithm.apply(blur)
    dilate = cv2.dilate((img_sub),np.ones((5,5)))
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilatada = cv2.morphologyEx(dilate,cv2.MORPH_CLOSE, kernal)
    dilatada_2 = cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE, kernal)
    contour,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filtered Video
    # cv2.imshow("Detecter", dilatada_2)

    # Draw line in video
    cv2.line(frame1,(0,300),(320,300), (255,127,0), 3)
    cv2.line(frame1,(330,300),(720,300), (0,0,255), 3)

    # Draw Boxes
    for(i,c) in enumerate(contour):
        (x,y,w,h) = cv2.boundingRect(c)
        counter = (w >= min_width) and (h >= min_height)
        if not counter:
            continue
        
        cv2.rectangle(frame1,(x,y),(x+w,y+h), (0,255,0), 2)
        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)

        for(x,y) in detect:
            if ((y < (300+offset)) and (y > (300-offset) and (x > 0) and (x < 320))):
                cars+=1
                cv2.line(frame1,(0,300),(320,300), (0,127,255), 3) 
                detect.remove((x,y))
            elif ((y < (300+offset)) and (y > (300-offset) and (x > 330) and (x < 720))):
                cars-=1
                cv2.line(frame1,(410,360),(660,360), (0,127,255), 3) 
                detect.remove((x,y))

    cv2.putText(frame1,"Cars: " + str(cars),(300,70),cv2.FONT_HERSHEY_SIMPLEX,2,(122,56,255),5)

                


    # Display the video
    cv2.imshow("Cars 4", frame1)

    cv2.waitKey(1) 

# Close all windows and Erase all captures
cv2.destroyAllWindows()
cap.release()