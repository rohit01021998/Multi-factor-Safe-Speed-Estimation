import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import math
import os
from collections import deque

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

def calculate_speed(angle, width=12):
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

def process_frame(frame, model, input_details, output_details, frame_buffer, csv_file_path):
    output_frame = frame.copy()
    
    # Preprocess frame
    resized_frame = cv2.resize(frame, (256, 256))
    frame_processed = preprocess_image(resized_frame)
    
    # Perform road segmentation with noise reduction
    road_mask = segment_road(model, input_details, output_details, frame_processed)
    
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
    
    # Draw road mask overlay
    road_overlay = np.zeros_like(frame)
    road_overlay[road_mask_resized > 0] = [0, 255, 0]
    output_frame = cv2.addWeighted(output_frame, 0.7, road_overlay, 0.3, 0)
    
    intersection = None
    angle = None
    speed = None
    
    if all(point is not None for point in [top_left, top_right, bottom_left, bottom_right]):
        # Define lines
        left_line = ((top_left[0], top_left[1]), (bottom_left[0], bottom_left[1]))
        right_line = ((top_right[0], top_right[1]), (bottom_right[0], bottom_right[1]))
        
        # Draw road boundary lines
        cv2.line(output_frame, 
                 (left_line[0][1], left_line[0][0]), 
                 (left_line[1][1], left_line[1][0]), 
                 (255, 0, 0), 2)
        cv2.line(output_frame, 
                 (right_line[0][1], right_line[0][0]), 
                 (right_line[1][1], right_line[1][0]), 
                 (255, 0, 0), 2)
        
        intersection = line_intersection(left_line, right_line)
        
        if intersection:
            cv2.circle(output_frame, (intersection[1], intersection[0]), 10, (0, 255, 0), -1)
            cv2.circle(output_frame, (center[1], center[0]), 10, (0, 0, 255), -1)
            cv2.line(output_frame, 
                    (center[1], center[0]), 
                    (intersection[1], intersection[0]), 
                    (0, 255, 0), 2)
            
            angle = calculate_angle(center, intersection)
            speed = calculate_speed(angle=angle)
    
    # Append new data to CSV file (or create the file if it doesn't exist)
    if angle is not None and speed is not None:
        new_data = {'Angle': angle, 'Speed': speed}
        new_df = pd.DataFrame([new_data])

        # Append data to the CSV file without overwriting
        new_df.to_csv(csv_file_path, mode='a', header=not pd.io.common.file_exists(csv_file_path), index=False)

        # Optional: Display the results on the frame
        angle_text = f"Angle: {angle:.2f} degrees"
        cv2.putText(output_frame, angle_text, 
                   (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        speed_text = f"Speed: {speed:.2f} kmph"
        cv2.putText(output_frame, speed_text,
                   (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return output_frame


def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "TurningAngleCalculation", "model.tflite")
    model = load_tflite_model(model_path)
    
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    
    model.resize_tensor_input(input_details[0]['index'], (1, 256, 256, 3))
    model.allocate_tensors()
    
    # Input video path - use the straight_2.mp4 file from the parent directory
    parent_dir = os.path.dirname(script_dir)
    video_path = os.path.join(parent_dir, "straight_2.mp4")
    cap = cv2.VideoCapture(video_path)
    
    # Initialize frame buffer
    frame_buffer = deque(maxlen=30)
    
    # Output paths - save to parent directory
    output_video_path = os.path.join(parent_dir, "output_video.avi")
    output_csv_path = os.path.join(parent_dir, "output_video.csv")
    
    # Get video properties for output video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (800, 800))  # Using resized dimensions
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = process_frame(frame, model, input_details, output_details, frame_buffer, output_csv_path)
            resized_frame = cv2.resize(processed_frame, (800, 800))
            
            # Write frame to video file
            out.write(resized_frame)
            
            # Display frame
            cv2.imshow('Road Angle Detection', resized_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Ensure resources are properly released
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()