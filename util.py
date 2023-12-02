
import easyocr
import tkinter as tk
from PIL import Image, ImageTk
import pyaudio
import numpy as np
import threading
import time


import cv2
import os

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                        'license_plate' in results[frame_nmr][car_id].keys() and \
                        'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    return True
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False
    """


def format_license(text):
    allowed_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                     'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # Filter out characters not in allowed_chars
    filtered_string = ''.join(char for char in text if char.upper() in allowed_chars)

    return filtered_string


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1


def open_image_dialog(licence_plate, image_path):
    def play_beep():
        p = pyaudio.PyAudio()

        volume = 0.5  # Range: 0.0 - 1.0
        fs = 44100  # Sampling frequency
        duration = 1.0  # Duration in seconds

        f = 440  # Frequency in Hz (adjust as needed)
        t = np.linspace(0, duration, int(fs * duration), False)
        data = volume * np.sin(2 * np.pi * f * t)

        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=fs,
                        output=True)

        start_time = time.time()
        # Play the beep sound for the first 10 seconds
        while time.time() - start_time < 10:
            stream.write(data.astype(np.float32).tobytes())

        stream.stop_stream()
        stream.close()
        p.terminate()

    def on_closing():
        root.destroy()


    root = tk.Tk()
    root.title(licence_plate)

    # Set a fixed size for the dialog window
    dialog_width = 1200  # Replace with your desired width
    dialog_height = 1000  # Replace with your desired height
    root.geometry(f"{dialog_width}x{dialog_height}")

    # Load the original image
    original_image = Image.open(image_path)

    # Calculate the initial size of the image
    initial_width, initial_height = original_image.size

    # Calculate the scaling factor to fit the image within the dialog
    width_ratio = dialog_width / initial_width
    height_ratio = dialog_height / initial_height
    scaling_factor = min(width_ratio, height_ratio)

    # Resize the image to fit within the dialog
    resized_width = int(initial_width * scaling_factor)
    resized_height = int(initial_height * scaling_factor)

    # Resize the image with anti-aliasing using the 'ANTIALIAS' method
    resized_image = original_image.resize((resized_width, resized_height), Image.LANCZOS)

    # Create the ImageTk object for the resized image
    image = ImageTk.PhotoImage(resized_image)

    # Create a label with the resized image
    label = tk.Label(root, image=image)
    label.pack(fill="both", expand=True)

    # Start a separate thread to play the beep sound continuously
    threading.Thread(target=play_beep).start()

    # Call on_closing function when the dialog window is closed
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Run the dialog window
    root.mainloop()

def save_and_return_cropped_image(big_frame, frame, x1, y1, x2, y2, output_directory, filename):
    # Get the dimensions of the smaller frame
    small_frame = frame[int(y1):int(y2), int(x1): int(x2), :]
    # Resize the smaller frame to three times its original size
    small_frame_resized = cv2.resize(small_frame, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)

    # Get the dimensions of the resized smaller frame
    small_height, small_width, _ = small_frame_resized.shape

    # Define the region of interest (ROI) for the smaller frame on the larger frame
    top_left_x, top_left_y = 10, 10  # Define the top-left corner coordinates for placement
    bottom_right_x, bottom_right_y = top_left_x + small_width, top_left_y + small_height

    # Ensure the region in the big frame matches the size of the resized smaller frame
    big_frame_roi = big_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    big_frame_roi_resized = cv2.resize(big_frame_roi, (small_width, small_height))

    # Place the resized smaller frame onto the resized region in the larger frame
    big_frame_copy = big_frame.copy()  # Make a copy to avoid modifying the original frame
    big_frame_copy[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = small_frame_resized

    # Save the resulting frame with the smaller frame appended
    output_path = os.path.join(output_directory, filename)
    resulting_image_path = output_path  # Replace with desired path and filename
    cv2.imwrite(resulting_image_path, big_frame_copy)

    return resulting_image_path





