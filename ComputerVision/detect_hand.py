import cv2
import os
from ultralytics import YOLO

# load the trained model
current_dir = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(current_dir, 'data', 'HandGesture', 'runs', 'detect', 'hand_gesture_v15', 'weights', 'best.pt')

model = YOLO(path)
cap = cv2.VideoCapture(0)

print("Searching for your hand... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    results = model(frame, conf=0.7)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # get the class id 
            cls = int(box.cls[0])
            conf = box.conf[0]

            # Get the coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 0:
                label = f"Closed {conf:.2f}"
                color = (0, 0, 255)  # red 
            elif cls == 1:
                label = f"Open: {conf:.2f}"
                color = (0, 255, 0) # green
            elif cls == 2:
                label = f"Thumb Up : {conf:.2f}"
                color = (127, 0, 255) # violet
            elif cls == 3:
                label = f"Pray : {conf:.2f}"
                color = (0, 0, 139) # dark blue

            # calculate center and radius
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            radius = int((x2 - x1) / 3)

            cv2.circle(frame, (center_x, center_y), radius, color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
    cv2.imshow("Real Time Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()