from ultralytics import YOLO
import os

def train_my_model():
    # 1. Path setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, 'data.yaml')

    # 2. Load the model
    model = YOLO('yolov8n.pt') 

    # 3. Start Training
    model.train(
        data=yaml_path,        
        epochs=50,             
        imgsz=640,            
        device=0,       
        name='hand_gesture_v1',
        workers=2  
    )

if __name__ == '__main__':
    train_my_model()