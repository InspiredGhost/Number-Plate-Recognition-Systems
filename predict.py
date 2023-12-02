import os
from ultralytics import YOLO
import cv2
import util
import query

# Initialize the camera
# video_capture = cv2.VideoCapture(0)

VIDEOS_DIR = os.path.join('/Users/inspiredghost/Documents/ModelData', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'vid.mov')

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


while ret:
    results = model(frame)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        print(f"Detected result: {result}")

        if class_id == 0:
            if score > threshold_car:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 4, cv2.LINE_AA)

        if class_id == 1:
            if score > threshold_licence:

                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Process Licence Plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_threshold = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read Licence Plate
                licence_plate_text, license_plate_score = util.read_license_plate(license_plate_crop_threshold)
                print("Licence Plate:", licence_plate_text, "at", license_plate_score, "%")

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 254), 4)
                # cv2.putText(frame, str(license_plate_score), (int(x1), int(y1 - 20)),
                            # cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 240), 4, cv2.LINE_AA)
                cv2.putText(frame, licence_plate_text, (int(x1), int(y1 - 60)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 4, cv2.LINE_AA)

                # KJ67NHGP
                result = query.query_firestore(licence_plate_text)
                print(result)
                if licence_plate_text is not None and result is True:
                    new_filename = licence_plate_text + ".png"
                    image_frame = frame.copy()
                    saved_image_path = util.save_and_return_cropped_image(image_frame, frame, x1, y1, x2, y2, "./snaps", new_filename)

                    util.open_image_dialog(licence_plate_text, saved_image_path)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()