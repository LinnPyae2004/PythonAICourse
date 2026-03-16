import torch
from ultralytics import YOLO

print(f"Python Version: {torch.sys.version.split()[0]}")
print(f"Is GPU (CUDA) available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Load a nano model to test
model = YOLO("yolo11n.pt") 
print("YOLO11 is ready on your GPU!")