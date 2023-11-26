from ultralytics import YOLO

#Load Model
model = YOLO("yolov8n.yaml") #Build New Model

#Use model
results = model.train(data="config.yaml", epochs=100) #train the model