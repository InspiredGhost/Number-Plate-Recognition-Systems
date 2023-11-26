import os
from ultralytics import YOLO
import cv2


VIDEOS_DIR = os.path.join('/Users/inspiredghost/Documents/ModelData', 'videos')

video_path = os.path.join(VIDEOS_DIR, '20231111_130030.mp4')
video_path_out = '{}_predicted.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train4', 'weights', 'best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model
threshold_car = 0.7
threshold_licence = 0.5
car_count = 0

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        print(f"Detected result: {result}")

        if class_id == 0:
            if score > threshold_car:
                car_count = car_count + 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, str(car_count), (int(x1), int(y1 - 60)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 4, cv2.LINE_AA)
        if class_id == 1:
            if score > threshold_licence:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 4, cv2.LINE_AA)
                cv2.putText(frame, "----------", (int(x1), int(y1 - 60)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 4, cv2.LINE_AA)





    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()