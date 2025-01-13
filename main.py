from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load the model

model.train(data='/home/saad_waqar/PycharmProjects/Weather_classification/dataset'
            , epochs=1, imgsz=64)