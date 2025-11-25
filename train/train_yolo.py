import cv2
from ultralytics import YOLO


def train():
    #Hyperparameters
    data = "../datasets/yolo/data.yaml"
    epochs = 50
    imgsz = 2048
    device = -1

    #Load the yolo model
    model = YOLO("yolo11n.pt")

    #Finetune yolo
    model.train(data=data, epochs=epochs, imgsz=imgsz, device=device)

if __name__ == "__main__":
    train()
