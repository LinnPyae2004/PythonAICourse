import cv2
import mediapipe as mp
import os
import glob

RAW_TARGET_DIR = f"Open_Palm" # put your messy photo data here  
RAW_INPUT_DIR = f"data/HandGesture/raw_staging/{RAW_TARGET_DIR}"

TARGET_DATA = "valid"
TARGET_DIR = f"data/HandGesture/{TARGET_DATA}"
IMAGE_FOLDER = f"{TARGET_DIR}/images"
LABEL_FOLDER = f"{TARGET_DIR}/labels"

# change these for each new gesture
PREFIX = "open_palm"
CLASS_ID = 1

# only cleans the gesture you are currently processing
def clean_specific_gesture(prefix):
    print(f"Syncing {prefix}: Deleting old versions from train folder...")
    # Clean images
    for img_file in glob.glob(os.path.join(IMAGE_FOLDER, f"{prefix}_*")):
        try: os.remove(img_file)
        except: pass
    # Clean labels
    for lbl_file in glob.glob(os.path.join(LABEL_FOLDER, f"{prefix}_*")):
        try: os.remove(lbl_file)
        except: pass

clean_specific_gesture(PREFIX) 

# Init MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Just making sure the directories exist 
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(LABEL_FOLDER, exist_ok=True)
os.makedirs(RAW_INPUT_DIR, exist_ok=True)

# Read from raw staging
files = [f for f in os.listdir(RAW_INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not files:
    print(f"Error: No files found in {RAW_INPUT_DIR}. Add photos first!")
else:
    print(f"Processing {len(files)} new images for {PREFIX}...")

count = 1

for filename in files:
    raw_path = os.path.join(RAW_INPUT_DIR, filename)
    image = cv2.imread(raw_path)

    if image is None:
        continue

    # Detection
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        # Standardized naming (always starts from 001)
        new_base_name = f"{PREFIX}_{count:03d}"
        new_img_name = f"{new_base_name}.jpg"
        new_txt_name = f"{new_base_name}.txt"

        # Save the image copy
        cv2.imwrite(os.path.join(IMAGE_FOLDER, new_img_name), image)

        # Save the YOLO label
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_center = (min(x_coords) + max(x_coords)) / 2
            y_center = (min(y_coords) + max(y_coords)) / 2
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)

            with open(os.path.join(LABEL_FOLDER, new_txt_name), 'w') as f:
                f.write(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        print(f"Success: {filename} -> {new_img_name}")
        count += 1
    else:
        print(f"Skipped: {filename} (Hand not detected)")

hands.close()
print(f"\nFinalized {PREFIX}. Dataset updated and clean.")