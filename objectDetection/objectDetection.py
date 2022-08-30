import cv2 as cv
import numpy as np
from gui_buttons import Buttons

# Opencv DNN
net = cv.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Load class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)


# FULL HD 1920 x 1080
height = 720
width = 1280
frame = np.zeros((height,width,3), np.uint8)
messageColor = (255,255,255)
# Initialize camera
cap = cv.VideoCapture(0)

# Create window
cv.namedWindow("Frame")

if not cap.isOpened():
     message = "Webcam not connected"
     x = int(width / 4)
     y = int(height / 2)
     cv.putText(frame, message, (x, y), cv.FONT_HERSHEY_PLAIN, 3, messageColor, 2)
     cv.imshow("Frame", frame)

else:
     width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
     height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
     
     isCameraConnected = True
     while isCameraConnected:
          # Get frames
          isCameraConnected, frame = cap.read()
          if isCameraConnected:

               # Object Detection
               (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=.4)
               for class_id, score, bbox in zip(class_ids, scores, bboxes):
                    (x, y, w, h) = bbox
                    class_name = classes[class_id]
                    colorBottle = (255,100,200)

                    if class_name == "bottle":
                         cv.putText(frame, class_name, (x, y - 10), cv.FONT_HERSHEY_PLAIN, 3, colorBottle, 2)
                         cv.rectangle(frame, (x, y), (x + w, y + h), colorBottle, 3)

          else:
               message = "Webcam disconnect"
               x = int(width / 4)
               y = int(height / 2)
               frame = np.zeros((height,width,3), np.uint8)
               cv.putText(frame, message, (x, y), cv.FONT_HERSHEY_PLAIN, 2, messageColor, 2)

          cv.imshow("Frame", frame)
          key = cv.waitKey(1)

     cap.release()

cv.waitKey()
cv.destroyAllWindows()
