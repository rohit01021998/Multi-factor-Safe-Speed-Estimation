import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import math
import csv
import os
import json
import joblib
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from collections import deque, Counter

# Fix the current working directory issue for ultralytics
try:
    os.getcwd()
except FileNotFoundError:
    os.chdir(os.path.expanduser("~"))

# Configure MPS (Metal Performance Shaders) for faster processing on macOS
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) for acceleration")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU")

# Set PyTorch device
torch.set_default_device(device)

############################## Model Loading and Global Variables ########################################

# Load models globally to avoid reloading for each frame
ENV_MODEL = tf.lite.Interpreter(model_path="regressionModels/EnvNewDataWithAug.tflite")
VISIBILITY_MODEL = tf.lite.Interpreter(model_path="regressionModels/visibility.model.tflite")
DAYNIGHT_MODEL = load_model('regressionModels/daynight_classification_resnet50.h5')
WEATHER_MODEL = load_model('regressionModels/weather_classification_resnet50.h5')
CLASSIFICATION_MODEL = tf.lite.Interpreter(model_path="regressionModels/TurnTypeClassifierTrimmedLined.tflite")

# Load YOLO model with device specification
YOLO_MODEL = YOLO('regressionModels/TrafficAndPedestrianbest.pt')
if torch.backends.mps.is_available():
    YOLO_MODEL.to('mps')

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

# Speed averaging queue - for 60 frames average
SPEED_AVERAGING_QUEUE = deque(maxlen=60)

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

############################## Regression Method Functions (Method 1) ########################################

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
    try:
        results = YOLO_MODEL(frame)
        
        effective_pedestrians = 0
        effective_traffic = 0
        safe_distance_threshold = 100

        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                for box in boxes:
                    if box.conf[0].item() >= 0.4:
                        class_id = int(box.cls[0])
                        if class_id == 0:  # Person/Pedestrian class
                            effective_pedestrians += 1
                        else:  # Vehicle classes (cars, trucks, buses, etc.)
                            effective_traffic += 1

        return json.dumps({
            "effective_pedestrians": effective_pedestrians,
            "effective_traffic": effective_traffic
        })
    except Exception as e:
        print(f"Error in traffic detection: {e}")
        return json.dumps({
            "effective_pedestrians": 0,
            "effective_traffic": 0
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
    return predicted_speed_limit

############################## YOLO Vehicle Tracking Functions (Method 2) ########################################

class VehicleSpeedTracker:
    def __init__(self, yolo_model=None, confidence=0.2, iou=0.6):
        """
        Initialize the vehicle speed tracking system
        
        Args:
            yolo_model (str): Path to YOLO model weights
            confidence (float): Confidence threshold for detection
            iou (float): Intersection over Union threshold
        """
        if yolo_model is None:
            # Get the directory of the current script and construct the path properly
            script_dir = os.path.dirname(os.path.abspath(__file__))
            yolo_model = os.path.join(script_dir, 'ObjectDetectionRelated', 'yoloModel', 'OrientationYoloV8Model.pt')
        self.model = YOLO(yolo_model)
        
        # Set model to use MPS if available
        if torch.backends.mps.is_available():
            self.model.to('mps')
        
        self.confidence = confidence
        self.iou = iou
        
        # Tracking parameters
        self.image_width = None
        self.image_height = None
        
        # Vehicle tracking database
        self.vehicle_tracks = {}
        
        # Speed cache to avoid recalculating
        self.vehicle_speeds = {}
        
        # Safe speed cache
        self.vehicle_safe_speeds = {}
        
        # Frame-wise safe speed tracking
        self.frame_safe_speeds = {}
        
        # Vehicle size reference database
        self.vehicle_size_reference = {
            'car': {'width': 1.8, 'height': 1.5},      # Average sedan
            'truck': {'width': 2.5, 'height': 2.8},    # Large truck
            'bus': {'width': 2.6, 'height': 3.5}       # Standard bus
        }

    def estimate_distance(self, bbox, class_name='car'):
        """
        Estimate distance to vehicle using advanced heuristics
        
        Args:
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
            class_name (str): Type of vehicle for more accurate estimation
        
        Returns:
            float: Estimated distance in meters
        """
        if self.image_width is None or self.image_height is None:
            # If image dimensions are not set, use default or last known dimensions
            self.image_width = 640
            self.image_height = 480
        
        # Unpack bounding box
        x1, y1, x2, y2 = bbox
        
        # Calculate bounding box dimensions
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Get reference vehicle dimensions based on class
        ref_height = self.vehicle_size_reference.get(class_name.lower(), 
                                                     self.vehicle_size_reference['car'])['height']
        
        # Perspective-based estimation
        # Add a small check to prevent division by zero
        if bbox_height > 0:
            perspective_factor = self.image_height / bbox_height
            estimated_distance = perspective_factor * ref_height
        else:
            print('################################ Distance Calculation failed ##########################################')
            estimated_distance = 10  # Default distance if calculation fails
        
        # Add some sanity checks
        estimated_distance = max(min(estimated_distance, 1000), 1)  # Limit between 5m and 1000m
        
        return estimated_distance

    def estimated_safe_speed(self, vehicle_speed_a, distance_meter):
        if distance_meter>25:
            # Convert vehicle speed from km/h to m/s
            vehicle_speed_a = (vehicle_speed_a ) * 0.27778  # Offset by 30 km/h and convert to m/s
            
            # Constants
            reaction_time = 3  # in seconds
            braking_deceleration = 3  # in m/s^2 (braking deceleration of the vehicle)
            
            try:
                # Calculate the distance covered during reaction time (distance vehicle travels before braking starts)
                distance_reaction = vehicle_speed_a * reaction_time
                
                # Calculate the remaining distance available for braking
                remaining_distance = distance_meter - distance_reaction
                
                # Check if there's enough space for braking
                if remaining_distance <= 0:
                    print("Not enough distance for safe braking.")
                    return 0
                
                # Calculate the maximum speed that can be safely achieved based on the remaining distance
                # Using the formula v^2 = 2 * a * d where 'a' is deceleration and 'd' is the remaining distance
                max_safe_speed = math.sqrt(2 * braking_deceleration * remaining_distance)
                max_safe_speed = math.sqrt(((reaction_time*braking_deceleration)**2)+2*braking_deceleration*remaining_distance) - (reaction_time*braking_deceleration)
                
                # Convert back to km/h from m/s
                safe_speed = max_safe_speed * 3.6
                
                # Debug print for checking values
                print(f"Debug - Vehicle Speed: {vehicle_speed_a * 3.6:.2f} km/h, Distance: {distance_meter:.2f}m, Safe Speed: {safe_speed:.2f} km/h")
                
                return safe_speed
            
            except Exception as e:
                print(f"Safe speed calculation error: {e}")
                return 0
        else:
            return float('inf')

    def calculate_speed(self, track_id):
        """
        Calculate vehicle speed based on distance changes
        
        Args:
            track_id (int): Unique identifier for the vehicle track
        
        Returns:
            float: Estimated speed in km/h
        """
        # Check if speed is already calculated
        if track_id in self.vehicle_speeds:
            return self.vehicle_speeds[track_id]
        
        if track_id not in self.vehicle_tracks:
            return 0
        
        track = self.vehicle_tracks[track_id]
        
        # Require at least 10 frames of tracking
        if len(track['distances']) < 20:
            return 0
        
        # Calculate distance change
        initial_distance = track['distances'][0]
        final_distance = track['distances'][-1]
        
        # Calculate speed
        distance_change = abs(final_distance - initial_distance)

        time_elapsed = len(track['distances']) / track.get('fps', 30)  # Default to 30 fps if not specified
        
        # Convert to km/h
        speed = (distance_change / time_elapsed) * 3.6  # m/s to km/h
        
        # Cache the speed
        self.vehicle_speeds[track_id] = speed
        
        print('###############################################################################')
        print(self.vehicle_speeds[track_id])

        return speed + 30

    def process_frame_for_yolo_tracking(self, frame, fps):
        """
        Process a single frame for YOLO vehicle tracking
        
        Returns:
            dict: Safe speeds for vehicles in this frame
        """
        try:
            # Set image dimensions
            self.image_width = frame.shape[1]
            self.image_height = frame.shape[0]
            
            # Run YOLO detection and tracking
            results = self.model.track(frame, 
                                       conf=self.confidence, 
                                       iou=self.iou, 
                                       persist=True,  # Enable tracking
                                       classes=[0]  # Car, bus, truck classes in COCO
                                      )
            
            frame_safe_speeds_dict = {}
            
            if results and results[0].boxes is not None:
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            try:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = box.xyxy[0]
                                bbox = (x1, y1, x2, y2)
                                
                                # Get confidence and class
                                conf = float(box.conf[0])
                                class_id = int(box.cls[0])
                                
                                # Get class name
                                class_name = self.model.names[class_id]
                                
                                # Get track ID
                                track_id = int(box.id[0]) if box.id is not None else -1
                                
                                # Estimate distance
                                distance = self.estimate_distance(bbox, class_name)
                                
                                # Track vehicle if it has a valid track ID
                                if track_id != -1:
                                    if track_id not in self.vehicle_tracks:
                                        self.vehicle_tracks[track_id] = {
                                            'distances': [],
                                            'fps': fps,
                                            'class': class_name,
                                            'last_known_distance': None
                                        }
                                    
                                    # Update track information
                                    track_data = self.vehicle_tracks[track_id]
                                    track_data['distances'].append(distance)
                                    track_data['last_known_distance'] = distance
                                    
                                    # Limit to last 10 frames
                                    if len(track_data['distances']) > 10:
                                        track_data['distances'] = track_data['distances'][-10:]
                                    
                                    # Calculate speed if enough frames
                                    if len(track_data['distances']) >= 10:
                                        speed = self.calculate_speed(track_id)
                                        
                                        # Calculate safe speed
                                        safe_speed = self.estimated_safe_speed(speed, track_data['last_known_distance'])
                                        
                                        # Store results
                                        self.vehicle_safe_speeds[track_id] = safe_speed
                                        frame_safe_speeds_dict[track_id] = safe_speed
                            
                            except Exception as e:
                                print(f"Error processing individual vehicle box: {e}")
                                continue
            
            return frame_safe_speeds_dict
            
        except Exception as e:
            print(f"Error in YOLO tracking: {e}")
            return {}

############################## Road Curvature Functions (Method 3) ########################################

def load_tflite_model(model_path):
    model = tf.lite.Interpreter(model_path=model_path)
    model.allocate_tensors()
    return model

def preprocess_image(image, target_size=(256, 256)):
    img = cv2.resize(image, target_size)
    img_np = img.astype('float32')[np.newaxis, ...]
    return img_np

def segment_road(model, input_details, output_details, image):
    """Perform road segmentation on the input image with noise reduction."""
    model.set_tensor(input_details[0]['index'], image)
    model.invoke()
    segmentation_output = model.get_tensor(output_details[0]['index'])[0].argmax(-1)
    
    # Convert to binary mask
    road_mask = (segmentation_output == 1).astype(np.uint8)
    
    # Apply morphological operations to remove noise
    kernel_close = np.ones((5, 5), np.uint8)
    kernel_open = np.ones((3, 3), np.uint8)
    
    # Close small holes
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Remove small noise blobs
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel_open)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(road_mask, connectivity=8)
    
    # Keep only the largest connected component (the road)
    if num_labels > 1:  # If there's at least one component besides background
        # Get areas of all components (excluding background)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_component_label = np.argmax(areas) + 1
        
        # Create mask with only the largest component
        road_mask = (labels == largest_component_label).astype(np.uint8)
    
    return road_mask

def find_edge_lines(edges):
    """Find lines on the edges of the road with improved noise handling."""
    height, width = edges.shape
    
    # Apply additional noise reduction to edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find the topmost edge pixel
    topmost_edge_y = -1
    topmost_edge_x = -1

    # Scan the rows from top to bottom
    for y in range(height):
        x_coords = np.where(edges[y, :] > 0)[0]
        if x_coords.size > 0:
            topmost_edge_y = y
            topmost_edge_x = x_coords[0]
            break

    # Define regions of interest
    top_start = topmost_edge_y if topmost_edge_y != -1 else 0
    top_end = top_start + 2
    bottom_start = height - 150
    bottom_end = height

    def find_extreme_pixels(region):
        edge_pixels = np.column_stack(np.where(region > 0))
        
        if len(edge_pixels) == 0:
            return None, None
        
        # Add minimum pixel count threshold
        if len(edge_pixels) < 5:  # Ignore regions with too few pixels
            return None, None
            
        leftmost_idx = edge_pixels[:, 1].argmin()
        rightmost_idx = edge_pixels[:, 1].argmax()
        
        left_point = edge_pixels[leftmost_idx]
        right_point = edge_pixels[rightmost_idx]
        
        return left_point, right_point

    # Find top region pixels
    top_region = edges[top_start:top_end, :]
    top_left, top_right = find_extreme_pixels(top_region)
    
    # Find bottom region pixels
    bottom_region = edges[bottom_start:bottom_end, :]
    bottom_left, bottom_right = find_extreme_pixels(bottom_region)
    
    # Adjust coordinates
    if top_left is not None:
        top_left[0] += top_start
    if top_right is not None:
        top_right[0] += top_start
    if bottom_left is not None:
        bottom_left[0] += bottom_start
    if bottom_right is not None:
        bottom_right[0] += bottom_start
    
    return top_left, top_right, bottom_left, bottom_right

def line_intersection(line1, line2):
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if den == 0:
        return None
    
    x = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / den
    y = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / den
    
    return (int(x), int(y))

def calculate_angle(center, point):
    dx = point[1] - center[1]
    dy = point[0] - center[0]
    
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg + 90

def calculate_speed_from_angle(angle, width=12):
    offset=3
    angle = abs(angle) - offset
    g = 9.8
    co_efficient_of_friction = 0.7
    wheelbase = 2.57
    tireWidth = 0.205 

    if angle != 0:
        angle = math.radians(angle)
        # radius_of_curvature = width/(2*math.sin(angle/2)) # wdith is not needed
        turning_radius = (wheelbase/math.sin(angle))+(tireWidth/2)
        temp = turning_radius*g*co_efficient_of_friction
        suggested_speed = math.sqrt(abs(temp))
    else:
        suggested_speed = 0
    return suggested_speed * 3.6

def smooth_points(points_buffer):
    """Apply temporal smoothing to points."""
    if not points_buffer:
        return None, None, None, None
    
    valid_points = [(tl, tr, bl, br) for tl, tr, bl, br in points_buffer 
                    if all(p is not None for p in [tl, tr, bl, br])]
    
    if not valid_points:
        return None, None, None, None
    
    # Calculate mean positions for each point
    avg_points = []
    for i in range(4):  # For each point type (top_left, top_right, bottom_left, bottom_right)
        points = [frame_points[i] for frame_points in valid_points]
        if points:
            avg_y = int(np.mean([p[0] for p in points]))
            avg_x = int(np.mean([p[1] for p in points]))
            avg_points.append([avg_y, avg_x])
        else:
            avg_points.append(None)
    
    return tuple(avg_points)

def process_frame_for_curvature(frame, curvature_model, curvature_input_details, curvature_output_details, frame_buffer):
    """
    Process frame for road curvature analysis
    
    Returns:
        tuple: (angle, curvature_speed)
    """
    try:
        # Preprocess frame
        resized_frame = cv2.resize(frame, (256, 256))
        frame_processed = preprocess_image(resized_frame)
        
        # Perform road segmentation with noise reduction
        road_mask = segment_road(curvature_model, curvature_input_details, curvature_output_details, frame_processed)
        
        # Resize road mask back to original frame size
        road_mask_resized = cv2.resize(road_mask.astype(np.uint8) * 255, 
                                       (frame.shape[1], frame.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
        
        # Apply edge detection with additional noise reduction
        edges = cv2.Canny(road_mask_resized, 100, 200)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find edge lines
        top_left, top_right, bottom_left, bottom_right = find_edge_lines(edges)
        
        # Store current frame's points in buffer
        current_points = [top_left, top_right, bottom_left, bottom_right]
        frame_buffer.append(current_points)
        
        # Apply temporal smoothing
        if len(frame_buffer) > 30:
            frame_buffer.popleft()
        
        if len(frame_buffer) == 30:
            top_left, top_right, bottom_left, bottom_right = smooth_points(frame_buffer)
        
        # Calculate center of the image
        height, width = frame.shape[:2]
        center = (height, width // 2)
        
        angle = None
        curvature_speed = None
        
        if all(point is not None for point in [top_left, top_right, bottom_left, bottom_right]):
            # Define lines
            left_line = ((top_left[0], top_left[1]), (bottom_left[0], bottom_left[1]))
            right_line = ((top_right[0], top_right[1]), (bottom_right[0], bottom_right[1]))
            
            intersection = line_intersection(left_line, right_line)
            
            if intersection:            
                angle = calculate_angle(center, intersection)
                curvature_speed = calculate_speed_from_angle(angle=angle)
        
        return angle, curvature_speed
    
    except Exception as e:
        print(f"Error in curvature processing: {e}")
        return None, None

############################## Helper Functions ########################################

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

def calculate_60_frame_average():
    """
    Calculate the average of the last 60 frame speeds
    """
    if len(SPEED_AVERAGING_QUEUE) > 0:
        return sum(SPEED_AVERAGING_QUEUE) / len(SPEED_AVERAGING_QUEUE)
    else:
        return 0
    
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

############################## Integrated Processing Function ########################################

def process_integrated_frame(frame, frame_count, last_env_weather_json, vehicle_tracker, fps, curvature_model, curvature_input_details, curvature_output_details, frame_buffer):
    """
    Process frame using all three methods and return the lowest safe speed
    """
    try:
        # Method 1: Regression Method
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
        
        # Calculate ML regression speed only (removed formula-based calculation)
        regression_speed = safe_speed_calculator_with_regressor(
            turn_result['classification'], road_result['road_type'], env_result['environment'],
            env_result['visibility'], traffic_result['effective_pedestrians'], traffic_result['effective_traffic'],
            env_result['daynight'], env_result['weather']
        )[0]
        
        # Method 2: YOLO Vehicle Tracking
        yolo_safe_speeds = vehicle_tracker.process_frame_for_yolo_tracking(frame, fps)
        yolo_min_speed = min(yolo_safe_speeds.values()) if yolo_safe_speeds else float('inf')
        
        # Method 3: Road Curvature Analysis
        angle, curvature_speed = process_frame_for_curvature(frame, curvature_model, curvature_input_details, curvature_output_details, frame_buffer)
        if curvature_speed is None:
            curvature_speed = float('inf')
        
        # Collect all valid speeds
        all_speeds = []
        method_names = []
        
        if regression_speed > 0:
            all_speeds.append(regression_speed)
            method_names.append("ML Regression")
            
        if yolo_min_speed != float('inf') and yolo_min_speed > 0:
            all_speeds.append(yolo_min_speed)
            method_names.append("Vehicle Following")
            
        if curvature_speed != float('inf') and curvature_speed > 0:
            all_speeds.append(curvature_speed)
            method_names.append("Road Curvature")
        
        # Find the minimum safe speed
        if all_speeds:
            min_safe_speed = min(all_speeds)
            min_speed_method = method_names[all_speeds.index(min_safe_speed)]
        else:
            min_safe_speed = 0
            min_speed_method = "No Valid Data"
        
        return {
            'env_result': env_result,
            'turn_result': turn_result,
            'traffic_result': traffic_result,
            'road_result': road_result,
            'regression_speed': regression_speed,
            'yolo_min_speed': yolo_min_speed,
            'curvature_speed': curvature_speed,
            'angle': angle,
            'min_safe_speed': min_safe_speed,
            'min_speed_method': min_speed_method,
            'all_speeds': all_speeds,
            'method_names': method_names,
            'last_env_weather_json': last_env_weather_json
        }
    
    except Exception as e:
        print(f"Error in integrated frame processing: {e}")
        # Return a safe fallback result
        return {
            'env_result': {'environment': 'City', 'daynight': 'Day', 'weather': 'Clear', 'visibility': 'High'},
            'turn_result': {'classification': 'Straight'},
            'traffic_result': {'effective_pedestrians': 0, 'effective_traffic': 0},
            'road_result': {'road_type': 'Medium', 'road_width_median': 10},
            'regression_speed': 0,
            'yolo_min_speed': float('inf'),
            'curvature_speed': float('inf'),
            'angle': None,
            'min_safe_speed': 30,  # Conservative fallback speed
            'min_speed_method': "Error - Using Fallback",
            'all_speeds': [30],
            'method_names': ["Fallback"],
            'last_env_weather_json': last_env_weather_json
        }

def draw_speed_overlay(frame, results):
    """
    Draw organized speed information overlay on the frame
    """
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Define positions for different information sections
    y_start = 30
    line_height = 25
    x_left = 10
    x_right = width - 350
    
    # Background rectangle for better readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (width - 5, 220), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Main result - Current frame lowest safe speed
    cv2.putText(frame, f'CURRENT LOWEST SAFE SPEED: {results["min_safe_speed"]:.1f} km/h', 
                (x_left, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'Method: {results["min_speed_method"]}', 
                (x_left, y_start + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # 60-frame average (most prominent display)
    cv2.putText(frame, f'60-FRAME AVERAGE: {results["avg_60_frame_speed"]:.1f} km/h', 
                (x_left, y_start + 2*line_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
    
    # Individual method results
    y_methods = y_start + 80
    cv2.putText(frame, 'Individual Method Results:', 
                (x_left, y_methods), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y_methods += line_height
    if results['regression_speed'] > 0:
        cv2.putText(frame, f'ML Regression: {results["regression_speed"]:.1f} km/h', 
                    (x_left, y_methods), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_methods += line_height - 5
        
    if results['yolo_min_speed'] != float('inf') and results['yolo_min_speed'] > 0:
        cv2.putText(frame, f'Vehicle Following: {results["yolo_min_speed"]:.1f} km/h', 
                    (x_left, y_methods), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_methods += line_height - 5
        
    if results['curvature_speed'] != float('inf') and results['curvature_speed'] > 0:
        cv2.putText(frame, f'Road Curvature: {results["curvature_speed"]:.1f} km/h', 
                    (x_left, y_methods), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        if results['angle'] is not None:
            y_methods += line_height - 5
            cv2.putText(frame, f'Road Angle: {results["angle"]:.1f}°', 
                        (x_left, y_methods), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Environmental conditions on the right side
    # y_env = 30
    # cv2.putText(frame, 'Conditions:', 
    #             (x_right, y_env), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    # y_env += line_height
    # cv2.putText(frame, f'Env: {results["env_result"]["environment"]}', 
    #             (x_right, y_env), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    # y_env += line_height - 5
    # cv2.putText(frame, f'Weather: {results["env_result"]["weather"]}', 
    #             (x_right, y_env), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    # y_env += line_height - 5
    # cv2.putText(frame, f'Time: {results["env_result"]["daynight"]}', 
    #             (x_right, y_env), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    # y_env += line_height - 5
    # cv2.putText(frame, f'Visibility: {results["env_result"]["visibility"]}', 
    #             (x_right, y_env), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    # y_env += line_height - 5
    # cv2.putText(frame, f'Road: {results["road_result"]["road_type"]}', 
    #             (x_right, y_env), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    # y_env += line_height - 5
    # cv2.putText(frame, f'Turn: {results["turn_result"]["classification"]}', 
    #             (x_right, y_env), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return frame

############################## Main Function ########################################

def main():
    print("="*60)
    print("INTEGRATED SAFE SPEED SYSTEM WITH 60-FRAME AVERAGING")
    print("="*60)
    
    if torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) for GPU acceleration")
    else:
        print("MPS not available, using CPU processing")
    
    input_video_path = r'testVideos/turning_2.avi'
    output_video_path = r'integrated_safe_speed_output-2.mp4'
    output_csv_path = r'integrated_safe_speed_results-2.csv'
    
    print(f"Input video: {input_video_path}")
    print(f"Output video: {output_video_path}")
    print(f"Results CSV: {output_csv_path}")
    print("="*60)
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print('Error: Could not open video')
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize YOLO tracker
    vehicle_tracker = VehicleSpeedTracker()
    
    # Initialize road curvature model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    curvature_model_path = os.path.join(script_dir, "TurningAngleCalculation", "model.tflite")
    curvature_model = load_tflite_model(curvature_model_path)
    curvature_input_details = curvature_model.get_input_details()
    curvature_output_details = curvature_model.get_output_details()
    curvature_model.resize_tensor_input(curvature_input_details[0]['index'], (1, 256, 256, 3))
    curvature_model.allocate_tensors()
    
    # Initialize frame buffer for curvature smoothing
    frame_buffer = deque(maxlen=30)
    
    frame_count = 0
    ret, first_frame = cap.read()
    if not ret:
        print('Error: Could not read first frame')
        return
    
    # Initialize with the first frame
    last_env_weather_json = process_frame_to_environment_daynight_weather(first_frame)
    
    # Create CSV file
    csv_headers = ['Frame', 'ML_Regression_Speed', 'YOLO_Min_Speed', 
                   'Curvature_Speed', 'Road_Angle', 'Lowest_Safe_Speed', '60_Frame_Average_Speed', 'Lowest_Speed_Method',
                   'Environment', 'Weather', 'DayNight', 'Visibility', 'Road_Type', 'Turn_Type']
    
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
    
    try:
        # Wrap the video frame processing loop with tqdm for progress bar
        for frame_count in tqdm(range(1, total_frames + 1), desc="Processing Video Frames", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Process frame with all three methods
                results = process_integrated_frame(
                    frame, frame_count, last_env_weather_json, vehicle_tracker, fps,
                    curvature_model, curvature_input_details, curvature_output_details, frame_buffer
                )
                
                # Update last_env_weather_json
                last_env_weather_json = results['last_env_weather_json']
                
                # Add current frame's lowest safe speed to averaging queue
                SPEED_AVERAGING_QUEUE.append(results['min_safe_speed'])
                
                # Calculate 60-frame average using helper function
                avg_60_frame_speed = calculate_60_frame_average()
                
                # Add the 60-frame average to results
                results['avg_60_frame_speed'] = avg_60_frame_speed
                
                # Draw overlay on frame
                annotated_frame = draw_speed_overlay(frame, results)
                
                # Write frame to output video
                out.write(annotated_frame)
                
                # Save results to CSV
                csv_row = [
                    frame_count,
                    results['regression_speed'] if results['regression_speed'] > 0 else '',
                    results['yolo_min_speed'] if results['yolo_min_speed'] != float('inf') else '',
                    results['curvature_speed'] if results['curvature_speed'] != float('inf') else '',
                    results['angle'] if results['angle'] is not None else '',
                    results['min_safe_speed'],
                    results['avg_60_frame_speed'],
                    results['min_speed_method'],
                    results['env_result']['environment'],
                    results['env_result']['weather'],
                    results['env_result']['daynight'],
                    results['env_result']['visibility'],
                    results['road_result']['road_type'],
                    results['turn_result']['classification']
                ]
                
                with open(output_csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csv_row)
                    
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                # Write the original frame if processing fails
                out.write(frame)
                continue
            
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
    print(f"Processing complete! Output video saved to: {output_video_path}")
    print(f"Results saved to: {output_csv_path}")

if __name__ == "__main__":
    main()