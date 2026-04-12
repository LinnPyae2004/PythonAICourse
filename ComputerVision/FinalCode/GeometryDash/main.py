import cv2
import os
import pydirectinput
import threading
from ultralytics import YOLO

# load the trained model
import os

# 1. This is: .../ComputerVision/FinalCode/GeometryDash/
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go UP one level: .../ComputerVision/FinalCode/
parent_dir = os.path.dirname(current_dir)

# 3. Go UP another level: .../ComputerVision/
grandparent_dir = os.path.dirname(parent_dir)

# 4. Now build the path from the PROJECT ROOT
model_path = os.path.join(grandparent_dir, 'runs', 'detect', 'hand_gesture_v1', 'weights', 'best.pt')

model = YOLO(model_path)
cap = cv2.VideoCapture(0)

print("Searching for your hand... Press 'q' to quit.")

hand_was_closed = False

def press_down():
    pydirectinput.press('space')

def release_up():
    pydirectinput.keyUp('space')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hand_is_currently_closed = False
    results = model(frame, conf=0.7, device=0, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # get the class id 
            cls = int(box.cls[0])
            conf = box.conf[0]

            # Get the coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 0 and conf > 0.6:
                label = f"Closed {conf:.2f}"
                hand_is_currently_closed = True
                color = (0, 0, 255)  # red 
            elif cls == 1:
                label = f"Open: {conf:.2f}"
                color = (0, 255, 0) # green

            # calculate center and radius
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            radius = int((x2 - x1) / 3)

            cv2.circle(frame, (center_x, center_y), radius, color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
    if hand_is_currently_closed and not hand_was_closed:
        threading.Thread(target=press_down, daemon=True).start()
        print("JUMP!")
        hand_was_closed = True
    elif not hand_is_currently_closed and hand_was_closed:
        threading.Thread(target=release_up, daemon=True).start()
        hand_was_closed = False

    cv2.imshow("Real Time Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()