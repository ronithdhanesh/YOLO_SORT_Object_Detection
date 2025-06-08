import cv2
import cvzone
from ultralytics import YOLO

cap = cv2.VideoCapture(r'C:\Users\USER\PycharmProjects\YOLO_SORT_Object_Detection\Files\Videos\cars.mp4')

model = YOLO('../Yolo-Weights/yolov8n.pt')

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2-x1, y2-y1
            #bbox = int(x1),int(y1),int(w),int(h)
            cvzone.cornerRect(img,(x1,y1,w,h))

    cv2.imshow('Video', img)
    cv2.waitKey(1)

