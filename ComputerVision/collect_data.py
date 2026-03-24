import cv2
import os
import time

OUTPUT_FOLDER = "Open_Palm" # change the name as you need 
OUTPUT_DIR = f"data/HandGesture/raw_staging/{OUTPUT_FOLDER}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FRAME_INTERNAL = 10

cap = cv2.VideoCapture(0)
count = 0
saved_count = 0

print("Starting capture, move your hand around! Press 'q' to stop.")

while True:
    ret, frame= cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)

    if count % FRAME_INTERNAL == 0:
        timestamp = int(time.time() * 1000)
        file_path = os.path.join(OUTPUT_DIR, f"gesture_{timestamp}.jpg")
        cv2.imwrite(file_path, frame)
        saved_count += 1
        cv2.circle(frame, (30, 30), 10, (0, 255, 0), -1) # flash circle when saving, wait do I even need this
        print(f"Image Number: {saved_count}")

    cv2.imshow("Capturing Data, move your hand!", frame)
    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()
print(f"Done! Saved {saved_count} images to {OUTPUT_DIR}")