# Project Name: Hand Gesture Recognition

## 🛠 Installation

### 1. Version Requirements
To avoid compatibility issues between **Mediapipe**, **YOLO** and **Python**, ensure you are using these specific versions:

* **Python**: `3.11.x`
* **Mediapipe**: `0.10.x`
* **Ultralytics (YOLO)**: `8.4.x`
* **OpenCV (Python)**: `4.13.x` or higher

---
### 2. Setup Steps

First, create a virtual environment to keep your dependencies clean:

**Windows**
```bash
# Create environment for Windows
py -3.11 -m venv venv

# Activate (Windows)
.\venv\Scripts\activate
```

**Mac & Linux**
```bash
# Create environment for Mac and Linux
python3.11 -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate
```

---

### 3. Installing Libraries

Once your virtual environment is activated, run the following command to install all necessary dependencies:
```bash
pip install mediapipe==0.10.11 ultralytics==8.4.0 opencv-python==4.13.0.80
```

#### Verification
To ensure everything is installed correctly, run this one-liner in your terminal:
```bash
python -c "import cv2, mediapipe, ultralytics; print('OpenCV:', cv2.__version__); print('Mediapipe:', mediapipe.__version__); print('Ultralytics:', ultralytics.__version__)"
```
If you see the version numbers printed without any errors, you are ready to go!


## ⚙️ How It Works
This project follows a structured pipeline: from raw image acquisition to a fully trained model ready for real-time inference.
#### Data Splitting (70/20/10)
To ensure the model is robust and performs well on unseen data, the dataset is divided into three distinct sets:
*   **Training Set (70%)**: The primary data used by the YOLO model to learn gestures and features.
*   **Validation Set (20%)**: Used during the training process to tune hyperparameters and prevent overfitting.
*   **Test Set (10%)**: A completely **blind** set used only after training is finished to provide an unbiased evaluation of the final model's performance.

### 1. Organizing the Image Data
The first stage focuses on building a high-quality dataset. Consistent data is key to high accuracy in computer vision.

#### `collect_data.py`
* **Capture Logic**: It captures a frame every `FRAME_INTERNAL = 10` frames. This ensures the dataset has a diverse range of hand positions and backgrounds without being redundant.
* **Storage**: Images are sorted into directories based on the gesture name (e.g., `data/HandGesture/raw_staging/{FolderName}`).
* **Clean Start**: The `clean_data()` function automatically wipes the target folder before starting a new session to prevent mixing old or irrelevant data into the current set.

#### `auto_label.py`
To avoid the tedious task of manually drawing bounding boxes for thousands of images, this script automates the entire labeling process.

* **Mediapipe Integration**: The script utilizes the **Mediapipe Hands** pipeline to detect ***21*** 3D hand landmarks in real-time. It calculates the extreme points of these landmarks to create an accurate bounding box around the hand.
* **Auto-Conversion**: It automatically converts these landmark coordinates into the specific **YOLO format** required for training.
* **Output**: For every image, a corresponding `.txt` file is generated containing: `<object-class> <x_center> <y_center> <width> <height>`.

---
### 2. Training
Once the data is organized into `train`, `valid`, and `test` sets, the training phase begins.

#### `data.yaml`
This is the configuration file that tells YOLO where to find your data.
* **Path Management**: It uses relative paths to point to your image folders. This makes the project portable across different machines.
* **Global Configuration**: Note that for relative paths to resolve correctly, the **datasets_dir** must be properly configured in the Ultralytics **settings.json** file (typically located in **%AppData%/Roaming/Ultralytics**).
* **Class Definition**: It maps class IDs to human-readable names:
  * `0`: `closed_palm`
  * `1`: `open_palm`

#### `train.py`
This script executes the learning process by feeding the organized data into the YOLO architecture.
* **Hyperparameters**: It defines training settings like `epochs=50` (number of passes over the data) and `imgsz=640` (the resolution the model "sees").
* **Optimization**: It utilizes **CUDA (NVIDIA GPU)** to accelerate the complex matrix math required for deep learning. 
  * **Note**: If no compatible NVIDIA GPU is detected, the script will automatically fallback to **CPU training**, though this will be significantly slower.
* **Results**: The final weights are saved as `best.pt` in the `runs/detect/` directory, which is the file used for the actual hand tracking application.


---
### 3. Detecting Hand Gestures

Once the model is trained, this stage brings everything together for real-time application.

#### `detect_hand.py`
This is the main inference script. it acts as the bridge between your webcam and the trained AI model.

*   **Model Loading**: The script loads the `best.pt` weights generated during the training phase.
*   **Real-time Processing**: 
    *   **Frame Capture**: Uses OpenCV to stream video from your webcam.
    *   **Inference**: Each frame is passed through the YOLO model to identify gestures and calculate confidence scores.
    *   **Visualization**: The script draws a bounding box and a label (e.g., "Open Palm 95%") over the detected hand in the video window.
*   **Performance**: 
    *   **GPU (Recommended)**: On systems with a dedicated GPU, the process runs at high FPS, providing a smooth, lag-free experience.
    *   **CPU**: On systems without a dedicated GPU, the detection will still function but at a significantly lower frame rate. This may result in "choppy" video or a delay between your hand movement and the on-screen detection.

## 📝 Extra Notes

### Pre-trained Model
*   **Version 1.0**: For immediate testing, I have included a pre-trained model in the `hand_gesture_v1` directory.
*   **Ready-to-Use Classes**: This model comes pre-trained with **4 distinct classes**:
    1.  `Open Palm`
    2.  `Closed Palm`
    3.  `Thumb Up`
    4.  `Thumb Down`
*   **Usage**: If you want to skip the training process and go straight to detection, simply point your `detect_hand.py` script to use `hand_gesture_v1/weights/best.pt`.
*   **Performance**: This model was trained on a baseline dataset and serves as a reference for the accuracy you should expect after running your own custom training.

### Tips for Better Accuracy
*   **Lighting**: Ensure your hand is well-lit. Shadows can sometimes interfere with the Mediapipe landmark detection during auto-labeling.
*   **Background**: A simple, non-cluttered background will provide the best results during the data collection phase.

---
**Happy Coding!** 🚀
*Developed by Linn Pyae*