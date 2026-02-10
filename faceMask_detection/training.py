from ultralytics import YOLO
from config import give_datayaml, get_device

def main():
    device, dtype = get_device()
    print(f"device : {device}, dtype = {dtype}")

    model = YOLO("yolov8s.pt")
    data_yaml = give_datayaml()

    model.train(
        data=data_yaml,
        epochs=10,
        imgsz=640,
        batch=16,
        device=0,
        augment=True
    )

if __name__ == "__main__":
    main()
