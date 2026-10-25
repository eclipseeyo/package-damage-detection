from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    model.train(
        data='Package-Computer-Vision/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        patience=15,
        name='second_dataset_only'
    )