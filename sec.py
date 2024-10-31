

import cv2
import cvzone
from ultralytics import YOLO
import matplotlib.pyplot as plt\

# Model path
model_path = r'pt/Drones.pt'
model_path2 = r'pt/yolov8n-face.pt'
model_path3 = r'pt/knives.pt'
model_path4 = r'pt\weapons.pt'



# Initialize camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)
droneModel = YOLO(model_path)
faceModel = YOLO(model_path2)
knivesModel = YOLO(model_path3)
weaponsModel = YOLO(model_path4)



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.resize(frame, (1020, 720))
    mainframe = frame.copy()

    # Predict objects
    results = droneModel.predict(frame)

                
    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h, w = y2 - y1, x2 - x1

            # Draw rectangles
            cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)
            cvzone.cornerRect(mainframe, [x1, y1, w, h], l=9, rt=3)

            # Blur detected face area
            face = frame[y1:y1+h, x1:x1+w]
            face = cv2.blur(face, (30, 30))
            frame[y1:y1+h, x1:x1+w] = face
 # Predict objects
    results2 = faceModel.predict(frame)

                
    # Process detections
    for result2 in results2:
        boxes = result2.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h, w = y2 - y1, x2 - x1

            # Draw rectangles
            cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)
            cvzone.cornerRect(mainframe, [x1, y1, w, h], l=9, rt=3)

            # Blur detected face area
            face = frame[y1:y1+h, x1:x1+w]
            face = cv2.blur(face, (30, 30))
            frame[y1:y1+h, x1:x1+w] = face


     # Predict objects
    results3 = knivesModel.predict(frame)

                
    # Process detections
    for result3 in results3:
        boxes = result3.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h, w = y2 - y1, x2 - x1

            # Draw rectangles
            cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)
            cvzone.cornerRect(mainframe, [x1, y1, w, h], l=9, rt=3)

            # Blur detected face area
            face = frame[y1:y1+h, x1:x1+w]
            face = cv2.blur(face, (30, 30))
            frame[y1:y1+h, x1:x1+w] = face
        


    # Stack images
    all_feeds = cvzone.stackImages([mainframe, frame], 2, 0.70)

    # Display results using OpenCV
    cv2.imshow('Face Recognition', all_feeds)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
