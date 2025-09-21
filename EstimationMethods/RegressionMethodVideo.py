import tensorflow as tf
import cv2
import numpy as np
import json
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import torch
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from collections import deque, Counter
import joblib
import math
import pandas as pd
import os

############################## Important Functions ########################################

# Load models globally to avoid reloading for each frame
ENV_MODEL = tf.lite.Interpreter(model_path="regressionModels/EnvNewDataWithAug.tflite")
VISIBILITY_MODEL = tf.lite.Interpreter(model_path="regressionModels/visibility.model.tflite")
DAYNIGHT_MODEL = load_model('regressionModels/daynight_classification_resnet50.h5')
WEATHER_MODEL = load_model('regressionModels/weather_classification_resnet50.h5')
CLASSIFICATION_MODEL = tf.lite.Interpreter(model_path="regressionModels/TurnTypeClassifierTrimmedLined.tflite")
YOLO_MODEL = YOLO('regressionModels/TrafficAndPedestrianbest.pt')

# Initialize models
for model in [ENV_MODEL, VISIBILITY_MODEL, CLASSIFICATION_MODEL]:
    model.allocate_tensors()

# Set up model details
ENV_INPUT_DETAILS = ENV_MODEL.get_input_details()
ENV_OUTPUT_DETAILS = ENV_MODEL.get_output_details()
VISIBILITY_INPUT_DETAILS = VISIBILITY_MODEL.get_input_details()
VISIBILITY_OUTPUT_DETAILS = VISIBILITY_MODEL.get_output_details()
CLASSIFICATION_INPUT_DETAILS = CLASSIFICATION_MODEL.get_input_details()
CLASSIFICATION_OUTPUT_DETAILS = CLASSIFICATION_MODEL.get_output_details()

ROAD_LENGTH_DATA_DQ = deque(maxlen=10)

# Resize input tensors
ENV_MODEL.resize_tensor_input(ENV_INPUT_DETAILS[0]['index'], (1, 224, 224, 3))
VISIBILITY_MODEL.resize_tensor_input(VISIBILITY_INPUT_DETAILS[0]['index'], (1, 224, 224, 3))
CLASSIFICATION_MODEL.resize_tensor_input(CLASSIFICATION_INPUT_DETAILS[0]['index'], (1, 224, 224, 3))

# Reallocate tensors after resizing
for model in [ENV_MODEL, VISIBILITY_MODEL, CLASSIFICATION_MODEL]:
    model.allocate_tensors()

# Define classes globally
ENV_CLASSES = ["City", "Highway", "Residential"]
DAYNIGHT_CLASSES = ['Day', 'Night']
WEATHER_CLASSES = ['Clear', 'Fogg', 'Rain']
VISIBILITY_CLASSES = ['High', 'Low']
CLASSIFICATION_CLASSES = ["Gentle turn", "Intersection", "Sharp turn", "Straight"]

def create_segmentation_model():
    model = tf.lite.Interpreter(model_path=r"regressionModels/roadsegmentation2.tflite")
    model.allocate_tensors()
    model.resize_tensor_input(model.get_input_details()[0]['index'], (1, 256, 256, 3))
    model.allocate_tensors()
    return model

def process_frame_to_environment_daynight_weather(frame):
    # Convert frame to RGB and resize once
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_224 = cv2.resize(img_rgb, (224, 224))
    img_np = np.array(img_224)[None].astype('float32')
    
    # Environment prediction
    ENV_MODEL.set_tensor(ENV_INPUT_DETAILS[0]['index'], img_np)
    ENV_MODEL.invoke()
    env_scores = ENV_MODEL.get_tensor(ENV_OUTPUT_DETAILS[0]['index'])
    env_pred = ENV_CLASSES[env_scores.argmax()]

    # Visibility prediction
    VISIBILITY_MODEL.set_tensor(VISIBILITY_INPUT_DETAILS[0]['index'], img_np)
    VISIBILITY_MODEL.invoke()
    visibility_scores = VISIBILITY_MODEL.get_tensor(VISIBILITY_OUTPUT_DETAILS[0]['index'])
    visibility_pred = VISIBILITY_CLASSES[visibility_scores.argmax()]

    # Day/Night and Weather prediction
    img_preprocessed = preprocess_input(img_np)
    daynight_pred = DAYNIGHT_CLASSES[np.argmax(DAYNIGHT_MODEL.predict(img_preprocessed, verbose=0))]
    weather_pred = WEATHER_CLASSES[np.argmax(WEATHER_MODEL.predict(img_preprocessed, verbose=0))]

    return json.dumps({
        "environment": env_pred,
        "daynight": daynight_pred,
        "weather": weather_pred,
        "visibility": visibility_pred
    })

def process_frame_to_turn_type(frame):
    # Convert frame to PIL image and resize
    original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    original_size = original_image.size
    
    # Road Segmentation
    img_256 = original_image.resize((256, 256))
    img_np = np.array(img_256)[None].astype('float32')

    segmentation_model = create_segmentation_model()
    segmentation_model.set_tensor(segmentation_model.get_input_details()[0]['index'], img_np)
    segmentation_model.invoke()
    segmentation_output = segmentation_model.get_tensor(segmentation_model.get_output_details()[0]['index'])[0].argmax(-1)
    
    # Calculate class proportions
    total_pixels = segmentation_output.size
    class_counts = np.bincount(segmentation_output.flatten(), minlength=3)
    class_proportions = class_counts / total_pixels

    # Resize segmentation for classification
    segmentation_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    segmentation_output_resized = cv2.resize(segmentation_output.astype('uint8'), (224, 224), interpolation=cv2.INTER_NEAREST)
    
    for class_idx, color in enumerate([(0, 0, 0), (128, 64, 128), (50, 234, 157)]):
        segmentation_rgb[segmentation_output_resized == class_idx] = color

    # Classification
    CLASSIFICATION_MODEL.set_tensor(CLASSIFICATION_INPUT_DETAILS[0]['index'], segmentation_rgb[None].astype('float32'))
    CLASSIFICATION_MODEL.invoke()
    class_scores = CLASSIFICATION_MODEL.get_tensor(CLASSIFICATION_OUTPUT_DETAILS[0]['index'])
    predicted_class = CLASSIFICATION_CLASSES[class_scores.argmax()]

    return json.dumps({
        "segmentation": {
            "nothing": class_proportions[0],
            "road": class_proportions[1],
            "marking": class_proportions[2]
        },
        "classification": predicted_class
    })

def process_frame_to_road_type(frame, segmentation_model):
    original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    original_size = original_image.size

    # Resize for segmentation model
    img_256 = original_image.resize((256, 256))
    img_np = np.array(img_256)[None].astype('float32')

    segmentation_model.set_tensor(segmentation_model.get_input_details()[0]['index'], img_np)
    segmentation_model.invoke()
    
    segmentation_output = segmentation_model.get_tensor(segmentation_model.get_output_details()[0]['index'])[0].argmax(-1)
    segmentation_output_resized = cv2.resize(segmentation_output.astype('uint8'), original_size, interpolation=cv2.INTER_NEAREST)

    # Convert to RGB for visualization
    segmentation_rgb = np.zeros((*original_size[::-1], 3), dtype=np.uint8)
    for class_idx, color in enumerate([(0, 0, 0), (128, 64, 128), (50, 234, 157)]):
        segmentation_rgb[segmentation_output_resized == class_idx] = color

    # Road width calculation
    road_mask = segmentation_output_resized == 1
    line_y_position = int(380)
    road_pixels = np.where(road_mask[line_y_position, :] == True)[0]
    
    if len(road_pixels) > 0:
        left_x = road_pixels[0]
        right_x = road_pixels[-1]
        min_x, max_x = left_x, right_x
        real_hood_length_in_meters = 1.85
        real_hood_length_in_pixels = 772
        pixel_to_meter_converter = real_hood_length_in_meters / real_hood_length_in_pixels
        distance_of_lane_width_line_in_pixels = max_x - min_x
        road_width_in_meters = (distance_of_lane_width_line_in_pixels * pixel_to_meter_converter * 12) / 1.5
        ROAD_LENGTH_DATA_DQ.append(road_width_in_meters)

    # Determine road type based on median road width
    median_width = calculate_median(ROAD_LENGTH_DATA_DQ)
    if median_width >= 12.5:
        road_type = 'Wide'
    elif 7 <= median_width < 12.5:
        road_type = 'Medium'
    else:
        road_type = 'Narrow'

    return json.dumps({
        "road_width_median": median_width,
        "road_type": road_type
    })

def detect_traffic_and_pedestrians(frame):
    results = YOLO_MODEL(frame)
    
    effective_pedestrians = 0
    effective_traffic = 0
    safe_distance_threshold = 100

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.conf[0].item() >= 0.4:
                class_id = int(box.cls[0])
                if class_id == 0:  # Pedestrian
                    effective_pedestrians += 1
                elif class_id == 1:  # Traffic
                    effective_traffic += 1

    return json.dumps({
        "effective_pedestrians": effective_pedestrians,
        "effective_traffic": effective_traffic
    })

def safe_speed_calculator(env_speed, road_type, turn_type, weather_type, time_of_the_day, visibility_conditions, traffic_level, pedestrian_level):
    # Define traffic and pedestrian levels
    traffic_category = 'Low' if traffic_level < 2 else 'Moderate' if traffic_level <= 5 else 'High'
    pedestrian_category = 'Low' if pedestrian_level < 2 else 'Moderate' if pedestrian_level <= 5 else 'High'

    # Define weights (unchanged)
    weights = {
        'pedestrian': {'High': 0.7, 'Moderate': 0.9, 'Low': 1, 'default': 1},
        'traffic': {'High': 0.6, 'Moderate': 0.8, 'Low': 1, 'default': 1},
        'env_speed': {'Highway': 120, 'City': 60, 'Residential': 30, 'default': 1},
        'road_type': {'Wide': 1, 'Medium': 0.8, 'Narrow': 0.3, 'default': 1},
        'turn_type': {'Intersection': 0.6, 'Sharp turn': 0.7, 'Gentle turn': 0.9, 'Straight': 1, 'default': 1},
        'weather_type': {'Rain': 0.8, 'Fogg': 0.6, 'Clear': 1, 'default': 1},
        'time_of_day': {'Day': 1, 'Night': 0.8, 'default': 1},
        'visibility': {'High': 1, 'Low': 0.8, 'Poor': 0.5, 'default': 1}
    }

    # Calculate speed using weights
    return (
        weights['env_speed'].get(env_speed, weights['env_speed']['default']) *
        weights['road_type'].get(road_type, weights['road_type']['default']) *
        weights['turn_type'].get(turn_type, weights['turn_type']['default']) *
        weights['weather_type'].get(weather_type, weights['weather_type']['default']) *
        weights['time_of_day'].get(time_of_the_day, weights['time_of_day']['default']) *
        weights['visibility'].get(visibility_conditions, weights['visibility']['default']) *
        weights['pedestrian'].get(pedestrian_category, weights['pedestrian']['default']) *
        weights['traffic'].get(traffic_category, weights['traffic']['default'])
    )

def calculate_average(dq):
    # Calculate the sum of elements
    total = sum(dq)
    # Get the number of elements in the deque
    count = len(dq)
    # Return the average
    if count > 0:
        return total / count
    else:
        return 0  # Return 0 if deque is empty
    
def calculate_median(dq):
    # Sort the deque
    sorted_dq = sorted(dq)
    
    # Calculate median
    n = len(sorted_dq)
    if n == 0:
        return 0  # Return 0 if deque is empty
    elif n % 2 == 1:
        # If odd, return the middle element
        return sorted_dq[n // 2]
    else:
        # If even, return the average of the two middle elements
        return (sorted_dq[n // 2 - 1] + sorted_dq[n // 2]) / 2
    
def mode_of_deque(d):
    counter = Counter(d)
    mode_data = counter.most_common(1)
    return mode_data[0][0] if mode_data else None

def safe_speed_calculator_with_regressor(TurnType,RoadType,EnvType,VisibilityConditions,PedestrianLevel,TrafficLevel,DayNight,WeatherConditions):

    # Define traffic and pedestrian levels
    TrafficLevel = 'Low' if TrafficLevel < 2 else 'Moderate' if TrafficLevel <= 5 else 'High'
    PedestrianLevel = 'Low' if PedestrianLevel < 2 else 'Moderate' if PedestrianLevel <= 5 else 'High'
    # Load the model from the file
    loaded_model = joblib.load(r'regressionModels/speed_limit_regression_model_normalised_upscaled_high.pkl')
    # Example: Making predictions using the loaded model
    # Assume you have new data (in the same format as the original dataset)
    new_data = pd.DataFrame({
        'TurnType': [str(TurnType)],
        'RoadType': [str(RoadType)],
        'EnvType': [str(EnvType)],
        'VisibilityConditions': [str(VisibilityConditions)],
        'PedestrianLevel': [str(PedestrianLevel)],
        'TrafficLevel': [str(TrafficLevel)],
        'DayNight': [str(DayNight)],
        'WeatherConditions': [str(WeatherConditions)]
    })

    # Predict using the loaded model
    predicted_speed_limit = loaded_model.predict(new_data)
    # print(f"Predicted Speed Limit: {predicted_speed_limit}")
    return predicted_speed_limit

############################## Usage ########################################

def process_frame(frame, frame_count, last_env_weather_json):
    # Create segmentation model once
    segmentation_model = create_segmentation_model()

    # Process environment/weather only every 30th frame
    if frame_count % 30 == 0 or frame_count % 30 == 1:
        with ThreadPoolExecutor(max_workers=4) as executor:
            env_weather_future = executor.submit(process_frame_to_environment_daynight_weather, frame)
            turntype_future = executor.submit(process_frame_to_turn_type, frame)
            traffic_future = executor.submit(detect_traffic_and_pedestrians, frame)
            roadtype_future = executor.submit(process_frame_to_road_type, frame, segmentation_model)
            
            last_env_weather_json = env_weather_future.result()
            turn_result = json.loads(turntype_future.result())
            traffic_result = json.loads(traffic_future.result())
            road_result = json.loads(roadtype_future.result())
    else:
        # For other frames, only process turn type, traffic, and road type
        with ThreadPoolExecutor(max_workers=3) as executor:
            turntype_future = executor.submit(process_frame_to_turn_type, frame)
            traffic_future = executor.submit(detect_traffic_and_pedestrians, frame)
            roadtype_future = executor.submit(process_frame_to_road_type, frame, segmentation_model)
            
            turn_result = json.loads(turntype_future.result())
            traffic_result = json.loads(traffic_future.result())
            road_result = json.loads(roadtype_future.result())
    
    env_result = json.loads(last_env_weather_json)
    
    calculated_speed = safe_speed_calculator(
        env_result['environment'], road_result['road_type'], turn_result['classification'],
        env_result['weather'], env_result['daynight'], env_result['visibility'],
        traffic_result['effective_traffic'], traffic_result['effective_pedestrians']
    )
#TurnType,RoadType,EnvType,VisibilityConditions,PedestrianLevel,TrafficLevel,DayNight,WeatherConditions
    regression_speed = safe_speed_calculator_with_regressor(
        turn_result['classification'], road_result['road_type'], env_result['environment'],
        env_result['visibility'], traffic_result['effective_pedestrians'], traffic_result['effective_traffic'],
        env_result['daynight'], env_result['weather']
    )
    
    # Define traffic and pedestrian levels
    TrafficLevel = 'Low' if traffic_result['effective_traffic'] < 2 else 'Moderate' if traffic_result['effective_traffic'] <= 5 else 'High'
    PedestrianLevel = 'Low' if traffic_result['effective_pedestrians'] < 2 else 'Moderate' if traffic_result['effective_pedestrians'] <= 5 else 'High'

    # Prepare the input values as a row to generate logs
    data = {
        'turn_type': [turn_result['classification']],
        'road_type': [road_result['road_type']],
        'environment_type': [env_result['environment']],
        'visibility': [env_result['visibility']],
        'effective_pedestrians': [PedestrianLevel],
        'effective_traffic': [TrafficLevel],
        'daynight': [env_result['daynight']],
        'weather': [env_result['weather']],
        'SafeSpeedLimit': [regression_speed[0]]
    }

    csv_filename = 'without_accident_test_video_traffic_long.csv'
    # os.remove(csv_filename)
    # Convert data into a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Check if CSV file exists
    if os.path.isfile(csv_filename):
        # Append without writing the header if file already exists
        df.to_csv(csv_filename, mode='a', header=False, index=False)
    else:
        # Write to CSV with the header if file doesn't exist
        df.to_csv(csv_filename, mode='w', header=True, index=False)

    return env_result, turn_result, traffic_result, calculated_speed, last_env_weather_json, road_result, regression_speed

def main():
    input_video_path = r'straight_2.mp4'
    output_video_path = r'output_test_video_traffic_long_without_accident.mp4'
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print('Error: Could not open video')
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    ret, first_frame = cap.read()
    if not ret:
        print('Error: Could not read first frame')
        return
    
    # Initialize with the first frame
    last_env_weather_json = process_frame_to_environment_daynight_weather(first_frame)
    # Create a deque with a maximum length of 100
    safe_speed_limit_queue = deque(maxlen=100)
    safe_speed_limit_regression_queue = deque(maxlen=100)
    
    try:
        # Wrap the video frame processing loop with tqdm for progress bar
        for frame_count in tqdm(range(1, total_frames + 1), desc="Processing Video Frames", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break
            
            env_result, turn_result, traffic_result, speed, last_env_weather_json, road_result, regression_speed = process_frame(
                frame, frame_count, last_env_weather_json
            )

            # Adding data to speed deque
            safe_speed_limit_queue.append(speed)
            # print(f"Deque after appending {speed}: {safe_speed_limit_queue}")

            safe_speed_limit_regression_queue.append(regression_speed[0])
            print(f"Deque after appending {regression_speed[0]}: {safe_speed_limit_regression_queue}")

            average_safe_speed = calculate_average(safe_speed_limit_queue)
            median_safe_speed = calculate_median(safe_speed_limit_queue)

            average_safe_speed_regression = calculate_average(safe_speed_limit_regression_queue)
            median_safe_speed_regression = calculate_median(safe_speed_limit_regression_queue)
            
            # Overlay predictions on frame (unchanged)
            cv2.putText(frame, f"Environment: {env_result['environment']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Day/Night: {env_result['daynight']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Weather: {env_result['weather']}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Visibility: {env_result['visibility']}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Turn Type: {turn_result['classification']}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Effective Traffic: {traffic_result['effective_traffic']}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Effective Pedestrians: {traffic_result['effective_pedestrians']}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Road Type: {road_result['road_type']}", (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(frame, f"Safe Speed Limit: {average_safe_speed:.2f},{median_safe_speed:.2f}", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Safe Limit By regression: {average_safe_speed_regression:.2f},{median_safe_speed_regression:.2f}", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out.write(frame)
            # cv2.imshow('Frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()