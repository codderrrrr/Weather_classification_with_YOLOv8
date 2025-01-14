from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load the model

model.train(data='/home/saad_waqar/PycharmProjects/Weather_classification/dataset'
            , epochs=30, imgsz=64)