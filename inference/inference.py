
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
import os
import json
from time import time
import utils.data_utils as utils
import utils.vbt_utils as vbt

MODELS = ["best_v8n.pt"]
VIDEOS = ["squat_1.mp4"]

def create_models(names_models=MODELS, path_to_weights="training_weights/"):
    """
    Create YOLO models.
    
    Parameters:
    names_models (list): A list of the names of the model weight files that will be used.
    path_to_weights (str): The path to the directory where the weights for the models are stored.
    
    
    Returns:
    dict: A dictionary where the keys are the name of the model and the value are the YOLO model instance.
    """
    models = {} # Initialize dictionary where the models are stored

    for model in names_models:
        models[model.split("_")[1].split(".")[0]] = YOLO(path_to_weights+model)

    return models

def inference(models, videos=VIDEOS, video_path="./videos/"):
    """
    Perform instance segmentation of frames in vidoes and store the centroid data points to track the barbell.
    The data points are stored in a json file.

    Parameters:
    models (dict): A dictionary of models instances.
    videos (list): A list of video names used in inference
    """

    

    for model_name, model in models.items():
        for video in videos:
            
            # Get information of and from video
            video_name = video.split(".")[0]
            exercise = video.split("_")[0]  

            # Open video file
            cap = cv2.VideoCapture(video_path + video)
            w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
            
            # Prepare video writer
            out = cv2.VideoWriter(video_name + ".avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

            centroid_data = {"meta": [model_name, w, h, fps]} # Store relevant meta data
            
            frame_number = -1 # Frame counter (start at -1 to make 0 the first frame number)
            names = model.model.names
            converted = False # Flag for conversion between meter and pixels in video
            time_list = []
            while True:
                ret, im0 = cap.read() # Read the video capture.
                
                if not ret:
                    print("Video frame is empty or video processing has been successfully completed.")
                    break

                frame_number += 1  # Current frame number

                # Perform prediction using the model
                cls = [0] # 0 = Barbell
                
                start_time = time()
                results = model.predict(im0, classes=cls[0], max_det=1) 
                end_time = time()
                time_list.append(end_time-start_time)
                annotator = Annotator(im0, line_width=2) # Initialize annotator

                # Get the meter per pixel ratio (necessary one time, first frame)
                if not converted:
                    meter_per_pixel = utils.get_meter_per_pixel(im0)
                    centroid_data["meta"].append(meter_per_pixel)
                    converted = True

                # Process results if masks are present
                if results[0].masks is not None:

                    boxes = results[0].boxes
                    #coords = boxes.xywh.cpu().tolist()[0]
                    masks = results[0].masks.xy

                    for mask, box in zip(masks, boxes.xywh):

                        # Get the coordinates of the centre of the box/barbell
                        coords = box.cpu().tolist()
                        x, y = coords[0], coords[1]

                        # Save the centroids
                        centroid_data[frame_number] = h-y # Subtract the position from the height of the video

                        # Draw a circle at the centroid of the largest contour
                        cv2.circle(im0, (int(x), int(y)), 4, colors(cls[0], True), -1)  # radius=10, filled circle

                        # Annotate segmentation and bounding box
                        annotator.seg_bbox(mask=mask, mask_color=colors(cls[0], True), det_label=names[cls[0]])                

                # Write the frame with annotations
                out.write(im0)
                cv2.imshow(exercise, im0)

                # Exit loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            # Cleanup
            out.release()
            cap.release()
            cv2.destroyAllWindows()
 
            # Summarize inference time
            centroid_data["meta"].append(np.mean(time_list))

            # Store the result
            json.dump(centroid_data, open(f"data/{video_name}_{model_name}.json", "w"))


def main():
    """
    The main function where inference, velocity approximation and plotting are done.
    """

    models = create_models() # Create the models.

    inference(models) # Perform inference on the videos.

    path = "./data/"
    files = os.listdir(path) # Extract the files in the directory "data"
    for file in files:

        if "json" not in file: # Error handling if file not in json-format is present
            continue

        # Get exercise information
        exercise = utils.extract_exercise(path + file) 
        exercise_number = utils.extract_exercise_number(file)
        
        # Load the data
        data = utils.load_inf_data(path + file)
        
        # Preprocess and prepare the data
        meta_data, y, time, fps = utils.prep_inf_data(data) 
        model_name = meta_data[0]    

        # Numerically approximate the velocities
        delta_time = 1/fps
        all_velocities, concentric_velocities, peak_velocity, mean_velocity = vbt.calculate_velocity(y, delta_time)
        
        # Calculate the Range of Motion
        RoM = vbt.calculate_rom(y)

        # Create the directory where the results are stored
        dir_name = f"Inference/{model_name}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # Plot and save the velocities
        vbt.plot_positions_and_velocities(time, y, all_velocities, dir_name, exercise, exercise_number, model_name)

        # Save the velocity data
        output_data = {"all_velocities": all_velocities, 
                         "concentric_velocities": concentric_velocities,
                         "peak_velocity": peak_velocity, 
                         "mean_velocity": mean_velocity,
                         "RoM": RoM, 
                         "mean_inference_time": meta_data[-1]}
        
        json.dump(output_data, open(f"{dir_name}/{exercise}_{model_name}_{exercise_number}.json", "w"))

main()