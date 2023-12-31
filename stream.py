import os
from ultralytics import YOLO
import cv2
import util

# Initialize the camera

VIDEOS_DIR = os.path.join('/Users/inspiredghost/Documents/ModelData', 'videos')
video_capture = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
# Load a model
model_path = os.path.join('.', 'runs', 'detect', 'train4', 'weights', 'best.pt')
model = YOLO(model_path)  # load a custom model
threshold_car = 0.7
threshold_licence = 0.5
car_count = 0

while True:

    video_path = os.path.join(VIDEOS_DIR, '20231111_131128.mp4')
    video_path_out = 'stream.mp4'.format(video_path)

    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

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

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 4, cv2.LINE_AA)
                cv2.putText(frame, licence_plate_text, (int(x1), int(y1 - 60)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 4, cv2.LINE_AA)

                alert = False
                # Query Firebase to check
                if alert is True:
                    new_filename = licence_plate_text + ".png"
                    image_frame = frame.copy()
                    saved_image_path = util.save_and_return_cropped_image(image_frame, frame, x1, y1, x2, y2, "./snaps", new_filename)

                    util.open_image_dialog(licence_plate_text, saved_image_path)

    # Display the frame in a window
    cv2.imshow('Camera Feed', frame)
    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()