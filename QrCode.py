from flask import Flask, jsonify, render_template, request,Response
import requests
import webbrowser
import time

import cv2
import numpy as np
from pyzbar.pyzbar import decode

# Insert image to program to test
# read the image
# img = cv2.imread('1.png')
# decode the qr code
# code = decode(img)

app = Flask(__name__)



# WebCam version
cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

# Center Circle
def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy



def genFrames():
    detect = []
    cars = 0
    offset = 6 # Allowable Error
    while True:
        sucess, img = cam.read()
        # Crossing line
        # cv2.line(img,(320,0),(320,480), (255,127,0), 3)


        for qr in decode(img):
        
            # Extract data from qr code
            data = qr.data.decode('utf-8')
            # Extract dimensions and location of qr code
            points = np.array([qr.polygon],np.int32)
            points = points.reshape((-1,1,2))
            # Draw box
            cv2.polylines(img,[points],True,(255,0,255),5)
            # Draw circle in middle of bounding box
            (x,y,w,h) = qr.rect
            center = center_handle(x,y,w,h)
            detect.append(center)
            cv2.circle(img,center,4,(255,0,255),-1) 
            # print(center)

            # Increase counter based on location
            for(x,y) in detect:
                # print(x,y)
                if ((x < (320+offset)) and (x > (320-offset) and (y > 0) and (y < 480))):
                    cars+=1
                    cv2.line(img,(320,0),(320,480), (0,127,255), 3) 
                    detect.remove((x,y))
                    print(cars)
                # elif ((y < (550+offset)) and (y > (550-offset) and (x > 700) and (x < 1100))):
                #     cars-=1
                #     cv2.line(img,(700,550),(1100,550), (0,127,255), 3) 
                #     detect.remove((x,y))    


        cv2.putText(img,"Cars: " + str(cars),(250,70),cv2.FONT_HERSHEY_SIMPLEX,2,(122,56,255),5)

        # cv2.imshow('Dectect me pls',img)
        # cv2.waitKey(1)

        ret, buffer = cv2.imencode('.jpg',img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


@app.route('/video')
def video():
    return Response(genFrames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

