# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:19:12 2024

@author: sanli
"""

from ultralytics import YOLO
import cv2
import math 
import time

epochs=100
project="NewIdentities"


model = YOLO('D:/repos/PYTHON/IdentityRecognition/'+ project +'/models/_' + str(epochs) + "_" + project + ".pt")


def ModelTest(path, newpath):
    img = cv2.imread(path)

    start_time = time.time()
    results = model(img)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Kodun çalışma süresi: {execution_time} saniye")
    
    for r in results:
        boxes = r.boxes
    
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            print(box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.imwrite(newpath, img)
            
#ModelTest("D:/repos/PYTHON/IdentityRecognition/Test/test1.jpg",'D:/repos/PYTHON/IdentityRecognition/'+ project +'/results/test1.jpg')
#ModelTest("D:/repos/PYTHON/IdentityRecognition/Test/test2.jpg",'D:/repos/PYTHON/IdentityRecognition/'+ project +'/results/test2.jpg')
#ModelTest("D:/repos/PYTHON/IdentityRecognition/Test/test3.jpg",'D:/repos/PYTHON/IdentityRecognition/'+ project +'/results/test3.jpg')
#ModelTest("D:/repos/PYTHON/IdentityRecognition/Test/test4.jpg",'D:/repos/PYTHON/IdentityRecognition/'+ project +'/results/test4.jpg')
#ModelTest("D:/repos/PYTHON/IdentityRecognition/Test/test5.jpg",'D:/repos/PYTHON/IdentityRecognition/'+ project +'/results/test5.jpg')
#ModelTest("D:/repos/PYTHON/IdentityRecognition/Test/test6.jpg",'D:/repos/PYTHON/IdentityRecognition/'+ project +'/results/test6.jpg')
#ModelTest("D:/repos/PYTHON/IdentityRecognition/Test/test7.jpg",'D:/repos/PYTHON/IdentityRecognition/'+ project +'/results/test7.jpg')
#ModelTest("D:/repos/PYTHON/IdentityRecognition/Test/test8.jpg",'D:/repos/PYTHON/IdentityRecognition/'+ project +'/results/test8.jpg')
#ModelTest("D:/repos/PYTHON/IdentityRecognition/Test/test9.jpg",'D:/repos/PYTHON/IdentityRecognition/'+ project +'/results/test9.jpg')
#ModelTest("D:/repos/PYTHON/IdentityRecognition/Test/test10.jpg",'D:/repos/PYTHON/IdentityRecognition/'+ project +'/results/test10.jpg')
#ModelTest("D:/repos/PYTHON/IdentityRecognition/Test/test11.png",'D:/repos/PYTHON/IdentityRecognition/'+ project +'/results/test11.jpg')
ModelTest("D:/repos/PYTHON/IdentityRecognition/Test/test12.png",'D:/repos/PYTHON/IdentityRecognition/'+ project +'/results/test12.jpg')