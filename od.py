import cv2 as cv
cap=cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)


classNames=[]
with open ('coco.names',"rt") as f:
    classNames=f.read().rstrip('\n').split("\n")
print (classNames)
configPath='/home/inmoov/Desktop/od/yolov4.cfg'
weightsPath='/home/inmoov/Desktop/od/yolov4.weights'
net=cv.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean([127.5,127.5,127.5])
net.setInputSwapRB(True)
while True:
    success , image=cap.read()
    classids,conf,bbox=net.detect(image,confThreshold=0.5)
    if len(classids) !=0:
        print(classids,bbox)
        for classid , confidence , box in zip(classids.flatten(),conf.flatten(),bbox):
            cv.rectangle(image,box,color=(255,0,0),thickness=2)
            cv.putText(image,classNames[classid],(box[0]+10,box[1]+30),cv.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
            cv.putText(image, str(confidence) , (box[0] + 150, box[1] + 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv.imshow('image',image)
        cv.waitKey(1)
