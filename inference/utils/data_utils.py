import json
import cv2
import numpy as np




def load_inf_data(file_path):
    """
    Load the data from the inference.

    Parameters:
    file_path (str): The path to the file that are to be opened.

    Returns:
    JSON object: The loaded JSON-file as a JSON object.
    """
    # Read and prepare the data
    with open(file_path) as f:
        return json.load(f)


def prep_inf_data(data):
    """
    Preprocessing of the data including:
    - Shifting the data so that the low point is 0.
    - Convert the data to meter (from pixels)
    - Extract the FPS rate
    - Creating the time series given the FPS and frame count

    Parameters:
    data (dict): The positional (y-coordinate) data of the center of the barbell as well as the meta data of the video

    Returns:
    list: The extracted meta data of the video
    list: The preprocessed positional data as a list
    list: The time series of the video, in seconds
    int: The FPS rate of the video    
    """


    
    meta_data = data.pop("meta") # Meta data format: [model_name, w, h, fps, meter_per_pixel, inference_time]
    fps = meta_data[-3]
    meter_per_pixel = meta_data[-2]

    time, y_data = [], []
    for frame, y in list(data.items()):
        y_data.append(y) # Add positional data to list
        time.append(int(frame)/fps) # Add time stamp of the frame

    min_y = min(y_data) # Extract the minimum point of the motion.

    data_list = [(elem-min_y) * meter_per_pixel for elem in y_data] # Convert to meter and shift data

    return meta_data, data_list, time, fps

def extract_exercise_number(file):
    """
    Extract the exercise repetition number from the name of a file.

    Parameters:
    file (str): The name of the file of the positional data

    Returns:
    str: The repetition number of an exericse
    """
    return file.split("_")[1]


def extract_exercise(file):
    """
    Extract the exercise of the name from a file.

    Parameters:
    file (str): The name of the file of the positional data

    Returns:
    str: The name of the exercise in the video
    """
    return file.split("_")[0].lower().split("/")[-1]

def get_meter_per_pixel(img):
    """
    Extract the ratio between meter and pixels in a video.

    Parameters:
    img (list): The first frame of an video

    Returns:
    float: The ratio between meter and pixels in the video
    """
    radius_plate = 0.45/2 # The radius of the plate in meter

    # Preprocess image (frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gauss_img = cv2.GaussianBlur(gray, (7,7), 1.5)

    # Use Hough Circle Transform to approximate a circle (the barbell plate)
    circles = cv2.HoughCircles(gauss_img, cv2.HOUGH_GRADIENT, dp=1.3, minDist=30, param1=150, param2=70, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles)) 

    # Get the radius of the approximated circle
    radius = circles[0, :][0][2]

    return radius_plate/radius