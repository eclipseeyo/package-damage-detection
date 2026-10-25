from ultralytics import YOLO
import os

if __name__ == '__main__':
    model_path = 'runs/detect/2class_final-5/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"Ошибка: модель по пути {model_path} не найдена. Сначала обучите модель.")
    else:
        model = YOLO(model_path)
        
        
        source_path = 'Merged-Dataset-2class/valid/images'
        
        results = model.predict(
            source=source_path,
            save=True,
            conf=0.25,
            project='runs/detect',
            name='predict_demo'
        )
        
        print(f"Готово. Результаты сохранены в runs/detect/predict_demo")