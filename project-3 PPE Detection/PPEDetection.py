from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(1)
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("../Videos/ppe-1.mp4")

model = YOLO('../yolo-weights/ppe.pt')

defaultColor = (0, 0, 255)

classNames = ['Hardhat',
              'Mask',
              'NO-Hardhat',
              'NO-Mask',
              'NO-Safety Vest',
              'Person',
              'Safety Cone',
              'Safety Vest',
              'machinery',
              'vehicle']

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            cv2.rectangle(img, (x1, y1), (x2, y2), defaultColor, 3)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class name
            cls = int(box.cls[0])

            currentClass = classNames[cls]

            if conf > 0.5:
                if currentClass in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                    defaultColor = (0, 255, 0)
                elif currentClass in ['Hardhat', 'Safety Vest', 'Mask']:
                    defaultColor = (0, 0, 255)
                else:
                    defaultColor = (255, 0, 0)

            cvzone.putTextRect(img, f'{currentClass} {conf}',
                               (max(0, x1), max(35, y1)),
                               scale=1, thickness=1,
                               colorB=defaultColor, colorT=(255, 255, 255), colorR=defaultColor)

    cv2.imshow('Image', img)
    # при нуле можно на рпобел стопать
    cv2.waitKey(1)
