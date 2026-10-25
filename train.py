from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    model.train(
        data='Merged-Dataset-2class/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        patience=20,
        name='2class_final'
    )
