from ultralytics import YOLO
import numpy as np

model = YOLO('./runs/classify/train3/weights/last.pt')

result = model('./prediction_dataset/img_3.png')

names_dict = result[0].names
prob = result[0].probs.data.tolist()

print(names_dict[np.argmax(prob)])
